"""Bundled inference-only model loader; no training or dataset reads at inference."""
import torch
from .data import DEFAULT_SCHEMA,read_json,validate_design,validate_pair
from .geometry import FEATURE_VERSION,fingerprint,reconstruct
from .model import LayoutTransformer,ModelConfig
from .tensors import encode,collate,forward_batch

def select_device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name=='auto' else name)
def to_device(batch,device):
    return {k:v.to(device) for k,v in batch.items()}

class Predictor:
    """Future engine integration: Predictor(checkpoint).resize(source, target_canvas)."""
    def __init__(self, checkpoint, device='auto', schema=DEFAULT_SCHEMA):
        self.device = select_device(device); self.schema = schema
        self.checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
        c = self.checkpoint
        if c.get('format_version')!=1 or c.get('feature_version')!=FEATURE_VERSION:
            raise ValueError('Incompatible checkpoint/feature version')
        if c.get('schema_hash')!=fingerprint(read_json(schema)): raise ValueError('Checkpoint schema differs')
        self.policy = c['background_policy']
        self.model = LayoutTransformer(ModelConfig(**c['model_config']))
        self.model.load_state_dict(c['state_dict'], strict=True)
        self.model.to(self.device).eval()

    def resize(self, source, target_canvas):
        validate_design(source, self.schema)
        batch = to_device(collate([encode(source,target_canvas,self.policy)]), self.device)
        with torch.inference_mode(): rows = forward_batch(self.model,batch)[0].cpu().tolist()
        result, warnings = reconstruct(source,target_canvas,rows,self.policy)
        warnings += validate_pair(source,result,self.policy,self.schema)
        return result, {'needs_visual_review':True,'warnings':sorted(set(warnings)),
                        'background_policy':self.policy,'projection':'rotated bounds and image aspect ratio',
                        'text_fit_verified':False,'official_render_performed':False}

