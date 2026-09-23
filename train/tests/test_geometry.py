import torch
from models.geometry import so3_exp, fk, PARENTS, attached_markers, joint_variance
from dataset.synthetic import synthetic_offsets


def test_so3_identity_gradient():
    v = torch.zeros(4, 3, requires_grad=True)
    r = so3_exp(v)
    assert torch.allclose(r, torch.eye(3).expand(4, 3, 3))
    (r * torch.randn_like(r)).sum().backward()
    assert torch.isfinite(v.grad).all()


def test_so3_valid_large_angles():
    r = so3_exp(torch.randn(128, 3) * 5)
    assert torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3).expand_as(r), atol=3e-6)
    assert torch.allclose(torch.linalg.det(r), torch.ones(128), atol=3e-6)


def test_fk_lengths_and_marker_attachment():
    offsets = torch.from_numpy(synthetic_offsets())
    rotations = so3_exp(torch.randn(2, 20, 3) * .5)
    points, frames = fk(rotations, offsets)
    for j in range(1, 20):
        length = (points[:, j]-points[:, PARENTS[j]]).norm(dim=-1)
        assert torch.allclose(length, offsets[j].norm().expand(2), atol=1e-7)
    result = attached_markers(points, frames, torch.tensor([7, 11]), torch.zeros(2, 2, 3))
    assert torch.allclose(result, points[:, [7, 11]])
    var = joint_variance(points, frames, torch.ones(2, 20, 3) * .01)
    assert torch.isfinite(var).all() and (var > 0).all()
