"""Torch-only tensor adapter. ZIP parsing/validation also work without PyTorch."""
import torch
from torch.utils.data import Dataset
from .data import read_zip, validate_pair
from .geometry import features, baseline_rows, layout_rows


def encode(source, size, policy='cover', target=None):
    values, canvas, groups = features(source, size)
    result = {'features':torch.tensor(values, dtype=torch.float32),
              'canvas':torch.tensor(canvas, dtype=torch.float32),
              'groups':torch.tensor(groups, dtype=torch.long),
              'baseline':torch.tensor(baseline_rows(source, size, policy), dtype=torch.float32),
              'text':torch.tensor([e['type']=='text' for e in source['elements']]),
              'aspect':torch.tensor(size['width']/size['height'], dtype=torch.float32)}
    if target is not None:
        result['target'] = torch.tensor(layout_rows(target, [e['id'] for e in source['elements']]), dtype=torch.float32)
    return result


class PairDataset(Dataset):
    def __init__(self, rows, policy, schema_path, max_elements=500):
        self.items = []
        for row in rows:
            pair = read_zip(row['path'])
            source, target = pair['source'], pair['target']
            validate_pair(source, target, policy, schema_path, max_elements)
            self.items.append(encode(source, target['canvas'], policy, target))

    def __len__(self): return len(self.items)

    def __getitem__(self, index): return self.items[index]


def collate(items):
    count = max(len(item['features']) for item in items)
    result = {}
    for key in items[0]:
        if key in ('canvas','aspect'):
            result[key] = torch.stack([item[key] for item in items])
        else:
            rows = []
            for item in items:
                x = item[key]
                padded = x.new_zeros((count,) + x.shape[1:])
                padded[:len(x)] = x
                rows.append(padded)
            result[key] = torch.stack(rows)
    result['valid'] = torch.arange(count)[None, :] < torch.tensor([len(i['features']) for i in items])[:, None]
    return result


def forward_batch(model, batch):
    return model(batch['features'],batch['canvas'],batch['groups'],batch['valid'],batch['baseline'])
