"""Causal HandCalib MVP: neural motion prior, slow calibration, diagonal correction.

The correction is a learned pseudo-measurement update in SO(3) tangent space.
It is NOT a full augmented error-state EKF or a physically identified IMU filter.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import torch
from torch import Tensor, nn
from .geometry import fk, so3_exp, joint_variance, orthonormalize
from .encoders import GloveEncoder, SmallImageEncoder


@dataclass
class ModelConfig:
    visual_mode: str = 'cached'  # cached | image | none
    rgb_feature_dim: int = 256
    use_depth: bool = False
    hidden: int = 256
    slow_hidden: int = 64
    use_slow: bool = True
    use_dynamics: bool = True
    learned_noise: bool = True
    max_rotation_rate: float = 6.0
    max_translation_rate: float = 2.0
    calibration_rate_m_s: float = .01
    calibration_bound_m: float = .05

    def __post_init__(self):
        if self.visual_mode not in {'cached', 'image', 'none'}:
            raise ValueError('visual_mode must be cached, image, or none')
        if min(self.hidden, self.slow_hidden, self.rgb_feature_dim) < 1:
            raise ValueError('Network dimensions must be positive')


@dataclass
class StreamState:
    rotations: Tensor
    root: Tensor
    angular_velocity: Tensor
    root_velocity: Tensor
    variance: Tensor  # [B,63]: root xyz then 20 local rotation tangent blocks
    fast1: Tensor
    fast2: Tensor
    slow: Tensor
    bias: Tensor  # [B,20,3], observation-space glove bias, metres; NOT gyro bias

    def detach(self) -> 'StreamState':
        return StreamState(**{k: v.detach() for k, v in vars(self).items()})

    def pack(self) -> dict[str, Tensor]:
        return {k: v.detach().cpu() for k, v in vars(self).items()}

    @classmethod
    def unpack(cls, data: dict[str, Tensor], device: torch.device | str) -> 'StreamState':
        return cls(**{k: v.to(device) for k, v in data.items()})


class HandCalib(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.glove_encoder = GloveEncoder()
        if c.visual_mode == 'image':
            self.visual = SmallImageEncoder(3, 256)
        else:
            self.visual = nn.Linear(c.rgb_feature_dim, 256)
        self.visual_geometry = nn.Sequential(nn.Linear(256 + 18, 256), nn.SiLU())
        self.depth = SmallImageEncoder(2, 128)
        # RGB256 + depth128 + glove128 + three modality flags + dt.
        self.fusion = nn.Sequential(nn.Linear(516, 256), nn.LayerNorm(256), nn.SiLU())
        self.session = nn.Sequential(nn.Linear(60, 32), nn.Tanh())
        # Historical rotations(120), root3, angular velocity60, root velocity3, dt1.
        self.state_embed = nn.Sequential(nn.Linear(187, c.hidden), nn.SiLU())
        self.fast1 = nn.GRUCell(c.hidden, c.hidden)
        self.fast2 = nn.GRUCell(c.hidden, c.hidden)
        self.motion_head = nn.Linear(c.hidden, 63)
        self.process_head = nn.Linear(c.hidden, 63)
        self.slow_cell = nn.GRUCell(256 + 60 + 32, c.slow_hidden)
        self.bias_head = nn.Linear(c.slow_hidden, 60)
        self.gate_head = nn.Linear(c.slow_hidden, 1)
        self.correction = nn.Sequential(nn.Linear(256 + 120 + c.slow_hidden, 256), nn.SiLU())
        self.delta_head = nn.Linear(256, 63)
        self.noise_head = nn.Linear(256, 63)
        for layer in (self.motion_head, self.bias_head, self.delta_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.process_head.weight)
        nn.init.constant_(self.process_head.bias, -5)
        nn.init.zeros_(self.noise_head.weight)
        nn.init.constant_(self.noise_head.bias, -2)

    def initial_state(self, batch: int, device: torch.device | str) -> StreamState:
        c = self.config
        z = lambda *shape: torch.zeros(*shape, device=device)
        return StreamState(torch.eye(3, device=device).expand(batch, 20, 3, 3).clone(),
                           z(batch, 3), z(batch, 20, 3), z(batch, 3),
                           torch.full((batch, 63), .1, device=device),
                           z(batch, c.hidden), z(batch, c.hidden), z(batch, c.slow_hidden), z(batch, 20, 3))

    @staticmethod
    def _six(rotations: Tensor) -> Tensor:
        return rotations[..., :2].flatten(1)

    def predict(self, previous: StreamState, dt: Tensor) -> StreamState:
        """Uses historical state and elapsed time ONLY; shared by free forecasting."""
        c = self.config
        x = torch.cat((self._six(previous.rotations), previous.root,
                       previous.angular_velocity.flatten(1), previous.root_velocity, dt[:, None]), -1)
        h1 = self.fast1(self.state_embed(x), previous.fast1)
        h2 = self.fast2(h1, previous.fast2)
        rate = torch.tanh(self.motion_head(h2))
        if not c.use_dynamics:
            rate = rate * 0
        vr = .8 * previous.root_velocity + .2 * c.max_translation_rate * rate[:, :3]
        w = .8 * previous.angular_velocity + .2 * c.max_rotation_rate * rate[:, 3:].reshape(-1, 20, 3)
        rotations = orthonormalize(previous.rotations @ so3_exp(w * dt[:, None, None]))
        root = previous.root + vr * dt[:, None]
        q = 1e-5 + .05 * torch.sigmoid(self.process_head(h2))
        if not c.learned_noise:
            q = torch.full_like(q, .001)
        variance = (previous.variance + q * dt[:, None]).clamp(1e-7, 10)
        return StreamState(rotations, root, w, vr, variance, h1, h2, previous.slow, previous.bias)

    def decode(self, state: StreamState, offsets: Tensor) -> dict[str, Tensor]:
        joints, frames = fk(state.rotations, offsets)
        variance = joint_variance(joints, frames, state.variance[:, 3:].reshape(-1, 20, 3))
        return {'joints': joints, 'world_joints': joints + state.root[:, None], 'root': state.root,
                'rotations': state.rotations, 'frames': frames, 'joint_variance': variance,
                'state_variance': state.variance, 'glove_reconstruction': joints + state.bias,
                'calibration_bias': state.bias}

    def step(self, inputs: dict[str, Tensor], offsets: Tensor,
             previous: StreamState | None = None) -> tuple[dict[str, Tensor], StreamState]:
        """One batched event. No target or MoCap argument is accepted."""
        dt = inputs['dt'].float()
        if dt.ndim != 1 or not torch.isfinite(dt).all() or (dt <= 0).any():
            raise ValueError('dt must be finite and positive [B]')
        b = dt.shape[0]
        state = previous or self.initial_state(b, dt.device)
        prior = self.predict(state, dt)
        prior_joints, _ = fk(prior.rotations, offsets)
        gvalid = inputs['glove_valid'].bool()
        raw_glove = torch.where(gvalid[..., None], inputs['glove'].float(), torch.zeros_like(inputs['glove']))
        glove = torch.where(gvalid[..., None], raw_glove - state.bias, torch.zeros_like(raw_glove))
        fg = self.glove_encoder(glove, gvalid, inputs['glove_age'].float())
        rv = inputs['rgb_valid'].bool() & (self.config.visual_mode != 'none')
        dv = inputs['depth_valid'].bool() & self.config.use_depth
        fr = torch.zeros(b, 256, device=dt.device)
        if rv.any():
            value = inputs['rgb'] if self.config.visual_mode == 'image' else inputs['rgb_features']
            encoded = self.visual(value[rv].float())
            geometry = inputs['geometry'][rv].float()
            fr[rv] = self.visual_geometry(torch.cat((encoded, geometry), -1))
        fd = torch.zeros(b, 128, device=dt.device)
        if dv.any():
            depth = inputs['depth'][dv].float()
            mask = inputs['depth_mask'][dv].float()
            fd[dv] = self.depth(torch.cat((depth.clamp(0, 5) * mask, mask), 1))
        gv = gvalid.any(-1)
        flags = torch.stack((rv, dv, gv), -1).float()
        fused = self.fusion(torch.cat((fr, fd, fg, flags, dt[:, None]), -1))
        innovation = ((raw_glove - prior_joints - state.bias) * gvalid[..., None]).flatten(1) / .1
        session = self.session(offsets.flatten(1) / .1)
        candidate_slow = self.slow_cell(torch.cat((fused, innovation, session), -1), state.slow)
        anchor = ((rv | dv) & gv)[:, None]
        if not self.config.use_slow:
            anchor = torch.zeros_like(anchor)
        slow = torch.where(anchor, candidate_slow, state.slow)
        gate = torch.sigmoid(self.gate_head(slow)) * anchor
        bias = state.bias + (gate * dt[:, None])[:, :, None] * self.config.calibration_rate_m_s * torch.tanh(self.bias_head(slow).reshape(b, 20, 3))
        bias = bias.clamp(-self.config.calibration_bound_m, self.config.calibration_bound_m)
        # Root-relative glove wrist is identically zero, never a calibration target.
        root_mask = torch.ones(1, 20, 1, device=dt.device)
        root_mask[:, 0] = 0
        bias = bias * root_mask
        h = self.correction(torch.cat((fused, self._six(prior.rotations), state.slow), -1))
        delta = torch.tanh(self.delta_head(h))
        scale = torch.cat((torch.full((b, 3), .25, device=dt.device), torch.ones(b, 60, device=dt.device)), -1)
        delta = delta * scale
        # Learned pseudo-observation covariance: bounded and diagonal.
        r = 1e-4 + .5 * torch.sigmoid(self.noise_head(h))
        if not self.config.learned_noise:
            r = torch.full_like(r, .05)
        valid = torch.cat(((rv | dv)[:, None].expand(-1, 3), (rv | dv | gv)[:, None].expand(-1, 60)), -1)
        k = (prior.variance / (prior.variance + r)) * valid
        corrected = k * delta
        rotations = orthonormalize(prior.rotations @ so3_exp(corrected[:, 3:].reshape(b, 20, 3)))
        root = prior.root + corrected[:, :3]
        # Scalar Joseph form; not a full state covariance propagation.
        variance = ((1 - k).square() * prior.variance + k.square() * r).clamp(1e-7, 10)
        current = StreamState(rotations, root, prior.angular_velocity, prior.root_velocity,
                              variance, prior.fast1, prior.fast2, slow, bias)
        out = self.decode(current, offsets)
        out['bias_increment'] = bias - state.bias
        out['gain'] = k
        return out, current

    def forward(self, inputs: dict[str, Tensor], offsets: Tensor,
                state: StreamState | None = None) -> tuple[dict[str, Tensor], StreamState]:
        if inputs['dt'].ndim != 2 or inputs['dt'].shape[1] < 1:
            raise ValueError('Expected inputs with [B,T,...], T >= 1')
        outputs = []
        for t in range(inputs['dt'].shape[1]):
            out, state = self.step({k: v[:, t] for k, v in inputs.items()}, offsets, state)
            outputs.append(out)
        return {k: torch.stack([o[k] for o in outputs], 1) for k in outputs[0]}, state

    def rollout(self, state: StreamState, offsets: Tensor, future_dt: Tensor) -> tuple[dict[str, Tensor], StreamState]:
        """API intentionally accepts no future sensor tensors, targets or features."""
        if future_dt.ndim != 2 or future_dt.shape[1] < 1 or (future_dt <= 0).any():
            raise ValueError('future_dt must be positive [B,H], H >= 1')
        outputs = []
        for t in range(future_dt.shape[1]):
            state = self.predict(state, future_dt[:, t])
            outputs.append(self.decode(state, offsets))
        return {k: torch.stack([o[k] for o in outputs], 1) for k in outputs[0]}, state

    def specification(self) -> dict:
        return {'model': asdict(self.config), 'representation': 'native20_joint_frame_SO3',
                'correction': 'diagonal_learned_pseudo_measurement',
                'calibration': 'glove_xyz_observation_bias_not_physical_imu_bias'}
