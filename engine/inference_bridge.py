"""Offline subprocess bridge to the bundled inference runtime.
Uses CHAMELEON_TORCH_PYTHON, local development Torch env, or current Python.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'final-1.0'
MODEL_PATH = ROOT/'engine/models/relations_v1.pt'


def resize_many(tasks, *, checkpoint=None, device='cpu'):
    python = os.environ.get('CHAMELEON_TORCH_PYTHON')
    if not python:
        local = ROOT/'torch-py/.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        python = str(local if local.exists() else Path(sys.executable))
    model = Path(checkpoint or MODEL_PATH).resolve()
    if not model.is_file(): raise FileNotFoundError(f'final inference checkpoint missing: {model}')
    result = subprocess.run(
        [python, '-B', '-m', 'chameleon_torch.worker', '--checkpoint', str(model), '--device', device],
        input=json.dumps(tasks, ensure_ascii=False, allow_nan=False),
        text=True, encoding='utf-8', capture_output=True, cwd=ROOT/'engine',
        env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'PYTHONUTF8':'1'}, timeout=300,
    )
    if result.returncode:
        raise RuntimeError('final inference inference failed: '+result.stderr.strip())
    return json.loads(result.stdout)

