"""Check required official IMUCoCo assets.

This script is intentionally strict: missing licensed/manual assets are reported
as failures instead of being silently skipped.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "IMUCoCo"

REQUIRED_FILES = {
    "official IMUCoCo checkpoint": PROJECT_ROOT / "saved_checkpoints" / "imucoco_best.pth",
    "official IMUCoCo loss map": PROJECT_ROOT / "saved_checkpoints" / "all_error_loss_map.pth",
    "official HPE checkpoint": PROJECT_ROOT / "saved_checkpoints" / "poser_dtp_best.pth",
    "SMPL male model": PROJECT_ROOT / "smpl" / "SMPL_MALE.pkl",
    "SMPL-H neutral model": PROJECT_ROOT / "smplh" / "neutral" / "model.npz",
    "upstream path config": UPSTREAM_ROOT / "path_config.py",
}

CHECKPOINT_FILES = {
    label: path
    for label, path in REQUIRED_FILES.items()
    if path.suffix == ".pth"
}


def is_readable(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def main() -> int:
    failures: list[str] = []

    if not UPSTREAM_ROOT.exists():
        failures.append(f"missing upstream repository: {UPSTREAM_ROOT}")

    for label, path in REQUIRED_FILES.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
            continue
        if not path.is_file():
            failures.append(f"not a file for {label}: {path}")
            continue
        if not is_readable(path):
            failures.append(f"not readable for {label}: {path}")

    for label, path in CHECKPOINT_FILES.items():
        if not path.is_file() or not is_readable(path):
            continue
        try:
            torch.load(path, map_location="cpu")
        except Exception as error:
            failures.append(f"invalid {label}: {path} ({error})")

    if failures:
        print("FAIL: required assets are not ready")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("PASS: all required official assets are present, readable, and valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
