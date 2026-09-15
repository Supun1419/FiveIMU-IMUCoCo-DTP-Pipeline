"""Export comparable geometric joint angles from a live prediction CSV."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "IMUCoCo"
sys.path.insert(0, str(UPSTREAM_ROOT))

import articulate as art  # noqa: E402


ANGLE_COLUMNS = (
    "left_shoulder_deg",
    "right_shoulder_deg",
    "left_elbow_deg",
    "right_elbow_deg",
    "left_hip_deg",
    "right_hip_deg",
    "left_knee_deg",
    "right_knee_deg",
    "left_ankle_deg",
    "right_ankle_deg",
)


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def geometric_joint_angles(joints: np.ndarray) -> dict[str, float]:
    """Calculate ten position-based angles using the SMPL joint layout."""
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (24, 3):
        raise ValueError(f"Expected SMPL joints shape (24, 3), got {points.shape}")

    pelvis, chest = points[0], points[9]
    torso_up = chest - pelvis
    torso_down = -torso_up

    result = {}
    for side, hip, knee, ankle, foot, shoulder, elbow, wrist in (
        ("left", 1, 4, 7, 10, 16, 18, 20),
        ("right", 2, 5, 8, 11, 17, 19, 21),
    ):
        result[f"{side}_shoulder_deg"] = angle_degrees(
            points[elbow] - points[shoulder], torso_down
        )
        result[f"{side}_elbow_deg"] = angle_degrees(
            points[shoulder] - points[elbow], points[wrist] - points[elbow]
        )
        result[f"{side}_hip_deg"] = angle_degrees(
            torso_up, points[knee] - points[hip]
        )
        result[f"{side}_knee_deg"] = angle_degrees(
            points[hip] - points[knee], points[ankle] - points[knee]
        )
        result[f"{side}_ankle_deg"] = angle_degrees(
            points[knee] - points[ankle], points[foot] - points[ankle]
        )
    return result


class PredictionAngleCalculator:
    def __init__(self, model_path: Path) -> None:
        self.body_model = art.ParametricModel(str(model_path), device="cpu")
        self.rotation_columns = [
            f"joint{joint}_R{row}{column}"
            for joint in range(24)
            for row in range(3)
            for column in range(3)
        ]

    def calculate(self, row: dict[str, str]) -> dict[str, float]:
        global_pose = torch.tensor(
            [float(row[column]) for column in self.rotation_columns],
            dtype=torch.float32,
        ).reshape(1, 24, 3, 3)
        local_pose = self.body_model.inverse_kinematics_R(global_pose)
        with torch.inference_mode():
            _, joints = self.body_model.forward_kinematics(
                local_pose, calc_mesh=False
            )
        return geometric_joint_angles(joints[0].numpy())


def latest_prediction(directory: Path) -> Path | None:
    candidates = [
        path
        for path in directory.glob("prediction_*.csv")
        if path.is_file() and path.stat().st_size > 0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def wait_for_prediction(directory: Path) -> Path:
    print(f"Waiting for a prediction CSV in {directory}")
    while True:
        selected = latest_prediction(directory)
        if selected is not None:
            return selected
        time.sleep(0.25)


def output_path_for(prediction_path: Path) -> Path:
    suffix = prediction_path.stem.removeprefix("prediction_")
    return prediction_path.with_name(f"joint_angles_{suffix}.csv")


def format_console(row: dict[str, str], angles: dict[str, float]) -> str:
    labels = (
        ("LEFT SHOULDER", "left_shoulder_deg"),
        ("RIGHT SHOULDER", "right_shoulder_deg"),
        ("LEFT ELBOW", "left_elbow_deg"),
        ("RIGHT ELBOW", "right_elbow_deg"),
        ("LEFT HIP", "left_hip_deg"),
        ("RIGHT HIP", "right_hip_deg"),
        ("LEFT KNEE", "left_knee_deg"),
        ("RIGHT KNEE", "right_knee_deg"),
        ("LEFT ANKLE", "left_ankle_deg"),
        ("RIGHT ANKLE", "right_ankle_deg"),
    )
    values = " ".join(f"{label}: {angles[name]:05.1f}" for label, name in labels)
    return f"{row['host_time']} frame={row['frame']} {values}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-file", type=Path)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "live_5imu_mqtt",
    )
    parser.add_argument("--output-file", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "smpl" / "SMPL_MALE.pkl",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Exit at the current end of the prediction file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write the angle CSV without printing every frame.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    prediction_path = (
        args.prediction_file.resolve()
        if args.prediction_file
        else wait_for_prediction(args.prediction_dir.resolve())
    )
    if not prediction_path.is_file():
        print(f"Prediction CSV not found: {prediction_path}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"SMPL model not found: {args.model}", file=sys.stderr)
        return 2

    output_path = (
        args.output_file.resolve() if args.output_file else output_path_for(prediction_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    calculator = PredictionAngleCalculator(args.model)
    print(f"Reading predictions: {prediction_path}")
    print(f"Writing angles: {output_path}")

    with prediction_path.open("r", newline="", encoding="utf-8") as source:
        while True:
            header_line = source.readline()
            if header_line:
                break
            if args.no_follow:
                print("Prediction file has no header", file=sys.stderr)
                return 1
            time.sleep(0.1)
        fieldnames = next(csv.reader([header_line]))
        required = {"frame", "host_time", "sensor_time_ms"} | set(
            calculator.rotation_columns
        )
        missing = required - set(fieldnames)
        if missing:
            print(f"Prediction CSV is missing columns: {sorted(missing)}", file=sys.stderr)
            return 2

        with output_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(
                target,
                fieldnames=("frame", "host_time", "sensor_time_ms", *ANGLE_COLUMNS),
            )
            writer.writeheader()
            try:
                while True:
                    line = source.readline()
                    if not line:
                        if args.no_follow:
                            break
                        time.sleep(0.02)
                        continue
                    values = next(csv.reader([line]))
                    if len(values) != len(fieldnames):
                        continue
                    row = dict(zip(fieldnames, values))
                    angles = calculator.calculate(row)
                    writer.writerow(
                        {
                            "frame": row["frame"],
                            "host_time": row["host_time"],
                            "sensor_time_ms": row["sensor_time_ms"],
                            **angles,
                        }
                    )
                    target.flush()
                    if not args.quiet:
                        print(format_console(row, angles))
            except KeyboardInterrupt:
                print("\nAngle export stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
