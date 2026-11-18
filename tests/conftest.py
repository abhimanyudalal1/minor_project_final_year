import os
from pathlib import Path
import pytest
import torch
from burgers_research.common import load_config

os.environ.setdefault("MPLCONFIGDIR", "/tmp/burgers-test-matplotlib")
torch.set_num_threads(1)

@pytest.fixture
def cfg():
    return load_config(Path(__file__).parents[1]/"configs"/"fixture.yaml")
