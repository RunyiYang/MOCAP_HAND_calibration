from pathlib import Path
import sys

# Independent test entry point; do not require installing the delivery package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import pytest
from dataset.synthetic import generate
from dataset.sequence import read_manifest, load_sequence, chunk
from models.handcalib import HandCalib, ModelConfig


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture
def fixture_data(tmp_path):
    path = generate(tmp_path / 'data', frames=16)
    manifest = read_manifest(path, allow_synthetic=True)
    seq = load_sequence(manifest['sequences'][0])
    return path, manifest, seq


@pytest.fixture
def network():
    torch.manual_seed(123)
    model = HandCalib(ModelConfig(hidden=32, slow_hidden=16))
    # Nonzero heads make causality/state tests meaningful, not trivially identity.
    with torch.no_grad():
        for head in (model.delta_head, model.motion_head, model.bias_head):
            head.weight.normal_(0, .005)
    return model.eval()
