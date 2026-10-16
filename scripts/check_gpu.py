"""Report CUDA availability and VRAM for the ML policy."""

from __future__ import annotations

import torch


def describe_device() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False, "device": "cpu", "torch": torch.__version__}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device": props.name,
        "vram_gb": round(props.total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> None:
    for key, value in describe_device().items():
        print(f"{key:>12}: {value}")


if __name__ == "__main__":
    main()
