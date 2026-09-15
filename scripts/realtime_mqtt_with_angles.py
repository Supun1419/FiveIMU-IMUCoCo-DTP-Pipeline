"""Run the unchanged MQTT pipeline and its angle exporter in one console."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION_DIR = PROJECT_ROOT / "experiments" / "live_5imu_mqtt"
DEFAULT_SMPLH_MODEL = PROJECT_ROOT / "smplh" / "neutral" / "model.npz"


def newest_prediction_since(directory: Path, started_at: float) -> Path | None:
    candidates = [
        path
        for path in directory.glob("prediction_*.csv")
        if path.is_file()
        and path.stat().st_size > 0
        and path.stat().st_mtime >= started_at
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--angle-prediction-dir",
        type=Path,
        default=DEFAULT_PREDICTION_DIR,
    )
    parser.add_argument(
        "--quiet-angles",
        action="store_true",
        help="Save angle CSV rows without printing every angle frame.",
    )
    parser.add_argument("--smplh-model", type=Path, default=DEFAULT_SMPLH_MODEL)
    parser.add_argument(
        "--smpl-renderer",
        action="store_true",
        help="Use the original SMPL renderer instead of SMPL-H.",
    )
    return parser.parse_known_args()


def stop_process(process: subprocess.Popen | None, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    own_args, pipeline_args = arguments()
    prediction_dir = own_args.angle_prediction_dir.resolve()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time() - 0.1

    pipeline_script = PROJECT_ROOT / "scripts" / "realtime_mqtt.py"
    renderer_args: list[str] = []
    if not own_args.smpl_renderer:
        smplh_model = own_args.smplh_model.resolve()
        if not smplh_model.is_file():
            print(f"SMPL-H model not found: {smplh_model}", file=sys.stderr)
            return 2
        pipeline_script = PROJECT_ROOT / "scripts" / "realtime_mqtt_smplh.py"
        renderer_args = ["--smplh-model", str(smplh_model)]

    pipeline_command = [
        sys.executable,
        str(pipeline_script),
        *renderer_args,
        *pipeline_args,
    ]
    renderer_name = "SMPL" if own_args.smpl_renderer else "SMPL-H"
    print(
        f"Starting MQTT reconstruction with {renderer_name} rendering and "
        "automatic joint-angle export."
    )
    pipeline = subprocess.Popen(pipeline_command, cwd=PROJECT_ROOT)
    angle_exporter: subprocess.Popen | None = None
    try:
        while pipeline.poll() is None and angle_exporter is None:
            prediction_path = newest_prediction_since(prediction_dir, started_at)
            if prediction_path is None:
                time.sleep(0.1)
                continue
            angle_command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "live_joint_angles.py"),
                "--prediction-file",
                str(prediction_path),
            ]
            if own_args.quiet_angles:
                angle_command.append("--quiet")
            angle_exporter = subprocess.Popen(angle_command, cwd=PROJECT_ROOT)
            print(f"Angle exporter attached to: {prediction_path.name}")

        return_code = pipeline.wait()
        return int(return_code)
    except KeyboardInterrupt:
        print("\nStopping reconstruction and angle export.")
        stop_process(pipeline)
        return 130
    finally:
        stop_process(angle_exporter, timeout=1.0)
        stop_process(pipeline, timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
