"""Small canvas-conditioned set Transformer; no pretrained downloads."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import FEATURE_DIM


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    feedforward: int = 512
    dropout: float = 0.1


class LayoutTransformer(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.element = nn.Linear(FEATURE_DIM, c.d_model)
        self.canvas = nn.Linear(8, c.d_model)
        self.group = nn.Linear(c.d_model, c.d_model, bias=False)
        layer = nn.TransformerEncoderLayer(c.d_model, c.nhead, c.feedforward,
                                          c.dropout, activation='gelu', batch_first=True,
                                          norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers, norm=nn.LayerNorm(c.d_model),
                                             enable_nested_tensor=False)
        self.head = nn.Linear(c.d_model, 7)
        # TransformerEncoder clones a layer: initialize each clone independently.
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, canvas, groups, valid, baseline):
        """valid=True denotes a real element. No element-order positional embedding."""
        hidden = self.element(features)
        same = (groups[:, :, None] == groups[:, None, :]) & valid[:, None, :]
        weights = same.to(hidden.dtype)
        pooled = weights @ hidden / weights.sum(-1, keepdim=True).clamp_min(1)
        hidden = hidden + self.group(pooled)
        tokens = torch.cat([self.canvas(canvas).unsqueeze(1), hidden], dim=1)
        mask = torch.cat([torch.zeros_like(valid[:, :1]), ~valid], dim=1)
        hidden = self.encoder(tokens, src_key_padding_mask=mask)[:, 1:]
        residual = self.head(hidden)
        center = torch.sigmoid(torch.logit(baseline[..., :2].clamp(1e-5, 1-1e-5)) + residual[..., :2])
        size_font = baseline[..., 2:5].clamp_min(1e-8) * residual[..., 2:5].clamp(-8, 8).exp()
        angle = F.normalize(baseline[..., 5:7] + residual[..., 5:7], dim=-1, eps=1e-6)
        return torch.cat([center, size_font, angle], dim=-1)


def per_design_mean(values, mask):
    return (values * mask).sum(1) / mask.sum(1).clamp_min(1)


def layout_loss(pred, batch):
    """Equal weight per poster, masked padding/non-text font channels.

    Overlap is deliberately not penalized globally: backgrounds/CTA groups overlap.
    A single target is one acceptable layout, not the only possible good layout.
    """
    target, valid = batch['target'], batch['valid']
    center = F.smooth_l1_loss(pred[..., :2], target[..., :2], reduction='none', beta=.1).mean(-1)
    size = F.smooth_l1_loss(pred[..., 2:4].clamp_min(1e-8).log(),
                            target[..., 2:4].clamp_min(1e-8).log(), reduction='none', beta=.1).mean(-1)
    font = F.smooth_l1_loss(pred[..., 4].clamp_min(1e-8).log(),
                            target[..., 4].clamp_min(1e-8).log(), reduction='none', beta=.1)
    angle = 1 - (pred[..., 5:7] * target[..., 5:7]).sum(-1).clamp(-1, 1)
    # Rotated extents must account for the target aspect ratio in normalized space.
    ratio = batch['aspect'][:, None]
    sn, cs = pred[..., 5].abs(), pred[..., 6].abs()
    half_w = (cs*pred[..., 2] + sn*pred[..., 3]/ratio)/2
    half_h = (sn*pred[..., 2]*ratio + cs*pred[..., 3])/2
    bounds = (F.relu(half_w-pred[..., 0]) + F.relu(pred[..., 0]+half_w-1)
              + F.relu(half_h-pred[..., 1]) + F.relu(pred[..., 1]+half_h-1))
    terms = {'center':per_design_mean(center, valid).mean(),
             'size':per_design_mean(size, valid).mean(),
             'font':per_design_mean(font, valid & batch['text']).mean(),
             'angle':per_design_mean(angle, valid).mean(),
             'bounds':per_design_mean(bounds, valid).mean()}
    total = 4*terms['center'] + terms['size'] + .5*terms['font'] + .2*terms['angle'] + .2*terms['bounds']
    return total, terms
