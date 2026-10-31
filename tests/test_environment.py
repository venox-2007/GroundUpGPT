"""Basic environment and package initialization tests."""

import sys
import src


def test_python_version() -> None:
    """Ensure Python version is 3.10 or higher."""
    assert sys.version_info >= (3, 10), "GroundUpGPT requires Python 3.10 or higher"


def test_package_metadata() -> None:
    """Ensure package root exposes valid version metadata."""
    assert hasattr(src, "__version__")
    assert isinstance(src.__version__, str)


def test_torch_import_and_cpu_tensor() -> None:
    """Verify PyTorch is importable and basic tensor operations function on CPU."""
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2.0
    assert torch.equal(y, torch.tensor([2.0, 4.0, 6.0]))
