"""Configuration, reproducibility and safe atomic checkpoints."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile

import torch
from models.handcalib import HandCalib, ModelConfig


def read_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    allowed = {'model', 'optimizer', 'training', 'losses'}
    if set(cfg) - allowed:
        raise ValueError(f'Unknown config sections: {set(cfg)-allowed}')
    known_training = {'seed', 'epochs', 'max_steps', 'chunk_length', 'rollout_steps', 'grad_clip', 'cpu_threads', 'checkpoint_every', 'log_every', 'nll_warmup_steps'}
    if set(cfg.get('training', {})) - known_training:
        raise ValueError('Unknown training setting; do not silently request an unsupported feature')
    ModelConfig(**cfg['model'])
    return cfg


def new_model(config: dict, device: str | torch.device) -> HandCalib:
    return HandCalib(ModelConfig(**config['model'])).to(device)


def atomic_save(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.checkpoint-', suffix='.pt', dir=path.parent)
    os.close(fd)
    try:
        torch.save(data, temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 'unavailable'


def load_checkpoint(path: str | Path, device: str | torch.device = 'cpu') -> tuple[HandCalib, dict]:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('schema') != 'handcalib.checkpoint.v1':
        raise ValueError('Unsupported checkpoint schema')
    model = new_model(ckpt['config'], device)
    model.load_state_dict(ckpt['model'])
    return model, ckpt
