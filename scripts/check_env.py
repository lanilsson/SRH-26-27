"""Check that the Python environment matches requirements.txt and report the
compute device — run once on each machine (laptop, RunPod pod) and copy the
output into the notebook.



Usage: python scripts/check_env.py
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def main():
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    ok = True
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, pinned = line.split("==")
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "MISSING"
        match = installed.split("+")[0] == pinned
        ok &= match
        print(f"  {'ok ' if match else 'BAD'} {name:16s} pinned {pinned:10s} installed {installed}")

    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"device: CUDA {torch.version.cuda}, {props.name}, {props.total_memory / 2**30:.1f} GiB")
    else:
        print("device: CPU (no CUDA; Apple MPS is not used — sparse matmul is unsupported there)")
    print("environment OK" if ok else "environment does NOT match requirements.txt")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
