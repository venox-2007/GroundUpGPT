"""Environment verification script for GroundUpGPT.

Inspects and reports Python runtime, operating system, PyTorch installation,
CUDA availability, and GPU hardware metrics. Handles non-CUDA and missing
dependency environments gracefully.
"""

from __future__ import annotations

import platform
import sys


def format_bytes_to_gb(bytes_val: int) -> float:
    """Convert bytes to gigabytes with 2 decimal places."""
    return round(bytes_val / (1024**3), 2)


def check_environment() -> bool:
    """Inspect local execution environment and print diagnostics.

    Returns:
        bool: True if core dependencies are met, False otherwise.
    """
    print("=" * 60)
    print(" GroundUpGPT - Environment Diagnostic Report")
    print("=" * 60)

    # 1. Python Environment
    print("\n[1] Python Runtime:")
    print(f"  - Version     : {platform.python_version()} ({sys.version})")
    print(f"  - Executable  : {sys.executable}")
    print(f"  - Architecture: {platform.architecture()[0]}")

    # 2. Operating System
    print("\n[2] Operating System:")
    print(f"  - System      : {platform.system()}")
    print(f"  - Release     : {platform.release()}")
    print(f"  - Version     : {platform.version()}")
    print(f"  - Platform    : {platform.platform()}")

    # 3. PyTorch & Acceleration Check
    print("\n[3] Deep Learning Framework (PyTorch):")
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        print("  - Status      : PyTorch is NOT installed in this Python environment.")
        print("  - Action      : Install PyTorch with CUDA support using:")
        print("                  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("\n" + "=" * 60)
        return False

    print(f"  - Version     : {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"  - CUDA Support: {'Available' if cuda_available else 'Unavailable (CPU-only build or no CUDA driver)'}")

    # 4. GPU Hardware Metrics
    print("\n[4] GPU Hardware Diagnostics:")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"  - Device Count: {device_count}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram_gb = format_bytes_to_gb(props.total_memory)
            compute_cap = f"{props.major}.{props.minor}"
            print(f"  - GPU [{i}] Name    : {props.name}")
            print(f"  - GPU [{i}] VRAM    : {total_vram_gb} GB")
            print(f"  - Compute Capability: {compute_cap}")
            print(f"  - Multi-Processors  : {props.multi_processor_count}")

        cuda_ver = torch.version.cuda or "N/A"
        cudnn_ver = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
        print(f"  - PyTorch CUDA Version : {cuda_ver}")
        print(f"  - cuDNN Version        : {cudnn_ver}")
    else:
        print("  - No CUDA-capable GPU detected by PyTorch.")
        print("  - NOTE: If an NVIDIA GPU is physically present on your machine, ensure:")
        print("          1. An updated NVIDIA display driver is installed.")
        print("          2. The CUDA-enabled PyTorch build is installed instead of the CPU wheel:")
        print("             pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124")

    print("\n" + "=" * 60)
    print(" Diagnostic Complete.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)
