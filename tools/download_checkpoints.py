"""Download public official IMUCoCo checkpoint assets."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "saved_checkpoints"

CHECKPOINTS = {
    "imucoco_best.pth": "https://synergylabs.org/haozhe/imucoco_best.pth",
    "all_error_loss_map.pth": "https://synergylabs.org/haozhe/all_error_loss_map.pth",
    "poser_dtp_best.pth": "https://synergylabs.org/haozhe/poser_dtp_best.pth",
}


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in CHECKPOINTS.items():
        out_path = CHECKPOINT_DIR / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"exists: {out_path}")
            continue
        print(f"download: {url} -> {out_path}")
        urlretrieve(url, out_path)


if __name__ == "__main__":
    main()
