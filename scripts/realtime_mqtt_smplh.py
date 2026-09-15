"""Run the working MQTT inference pipeline with a separate SMPL-H viewer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--smplh-model",
        type=Path,
        default=PROJECT_ROOT / "smplh" / "neutral" / "model.npz",
    )
    viewer_args, remaining = parser.parse_known_args()
    model_path = viewer_args.smplh_model.resolve()
    if not model_path.is_file():
        print(f"SMPL-H model not found: {model_path}", file=sys.stderr)
        return 2

    os.environ["IMUCOCO_SMPLH_MODEL"] = str(model_path)
    if "--visualize" not in remaining:
        remaining.append("--visualize")
    sys.argv = [sys.argv[0], *remaining]

    import src.visualization.live_smpl as live_smpl
    from scripts import realtime_mqtt
    from src.visualization.live_smplh import LiveSMPLHViewer

    live_smpl.LiveSMPLViewer = LiveSMPLHViewer
    return realtime_mqtt.main()


if __name__ == "__main__":
    raise SystemExit(main())
