"""Observation encoders. No network access or implicit checkpoint downloads."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from .geometry import PARENTS


class SmallImageEncoder(nn.Module):
    """Trainable fallback, NOT a pretrained HaMeR/ResNet model.

    GroupNorm avoids coupling an early frame to later frames in a batch.
    For pretrained features, use visual_mode='cached' and a audited feature cache.
    """
    def __init__(self, channels: int, output: int):
        super().__init__()
        layers = []
        for width in (32, 64, 128):
            layers += [nn.Conv2d(channels, width, 3, 2, 1), nn.GroupNorm(8, width), nn.SiLU()]
            channels = width
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, output))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GloveEncoder(nn.Module):
    """Two fixed-topology graph message-passing layers, 128 channels per node."""
    def __init__(self, width: int = 128):
        super().__init__()
        adjacency = torch.eye(20)
        for j, p in enumerate(PARENTS[1:], 1):
            adjacency[j, p] = adjacency[p, j] = 1
        self.register_buffer('adjacency', adjacency / adjacency.sum(-1, keepdim=True))
        self.embed = nn.Linear(8, width)
        self.layers = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.output_dim = width

    def forward(self, joints: Tensor, valid: Tensor, age: Tensor) -> Tensor:
        parents = torch.tensor([max(0, p) for p in PARENTS], device=joints.device)
        bones = joints - joints[:, parents]
        x = torch.cat((joints / .1, bones / .1, valid[..., None].float(),
                       age[:, None, None].expand(-1, 20, 1).clamp(0, 1)), -1)
        h = torch.nn.functional.silu(self.embed(x))
        for layer, norm in zip(self.layers, self.norms):
            neighbour = torch.einsum('jk,bkc->bjc', self.adjacency, h)
            h = norm(h + torch.nn.functional.silu(layer(torch.cat((h, neighbour), -1))))
        w = valid[..., None].float()
        return (h * w).sum(1) / w.sum(1).clamp_min(1)
