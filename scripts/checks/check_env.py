"""
Environment sanity check — run with: uv run check-env
"""

import sys


def main() -> None:
    import torch

    print("=" * 50)
    print("Environment Check")
    print("=" * 50)
    print(f"Python:          {sys.version.split()[0]}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}:        {name} ({mem:.1f} GB)")
    else:
        print("  (No GPU detected — training will use CPU)")

    print("=" * 50)
    print("✓ Environment check passed.")


if __name__ == "__main__":
    main()
