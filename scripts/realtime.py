"""Noise-filtered five-BNO085 serial input and live IMUCoCo GPU inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.sensor_to_body import SENSOR_ORDER  # noqa: E402
from src.input.bno085_adapter import diagnose_globalpose_packet  # noqa: E402
from src.input.live_adapter import (  # noqa: E402
    FilteredSynchronizer,
    TposeCollector,
    recover_live_target,
)
from src.input.noise_filter import FiveSensorNoiseFilter, NoiseFilterConfig  # noqa: E402
from src.models.imucoco_runtime import OnlineFiveIMU  # noqa: E402


def arguments(configure_parser=None, default_output_dir: Path | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-hz", type=float, default=60.0)
    parser.add_argument("--tpose-seconds", type=float, default=2.0)
    parser.add_argument("--gravity-seconds", type=float, default=2.0)
    parser.add_argument("--bias-seconds", type=float, default=5.0)
    parser.add_argument("--calibration-gyro-limit-deg", type=float, default=3.0)
    parser.add_argument("--stale-ms", type=float, default=250.0)
    parser.add_argument("--max-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir or PROJECT_ROOT / "experiments" / "live_5imu",
    )
    parser.add_argument("--adapter-only", action="store_true", help="Record filtered frames without loading the DL models")
    parser.add_argument("--visualize", action="store_true", help="Display the predicted SMPL mesh in a live 3D window")
    if configure_parser is not None:
        configure_parser(parser)
    return parser.parse_args()


def input_header() -> list[str]:
    columns = ["frame", "host_time", "sensor_time_ms", "all_sensors_valid"]
    columns += [f"{name}_stationary" for name in SENSOR_ORDER]
    columns += [f"{name}_sequence" for name in SENSOR_ORDER]
    for name in SENSOR_ORDER:
        columns += [f"{name}_r6d_{index}" for index in range(6)]
        columns += [f"{name}_a{axis}" for axis in "xyz"]
    for name in SENSOR_ORDER:
        columns += [f"{name}_w{axis}" for axis in "xyz"]
    return columns


def prediction_header() -> list[str]:
    return ["frame", "host_time", "sensor_time_ms", "latency_ms", "tx", "ty", "tz"] + [
        f"joint{joint}_R{row}{column}"
        for joint in range(24) for row in range(3) for column in range(3)
    ]


def physics_prediction_header() -> list[str]:
    columns = (
        [f"raw_t{axis}" for axis in "xyz"]
        + [f"refined_v{axis}" for axis in "xyz"]
        + [f"acceleration_v{axis}" for axis in "xyz"]
        + [f"direction_{axis}" for axis in "xyz"]
        + ["direction_valid", "physics_state", "physics_dt"]
    )
    for name in SENSOR_ORDER:
        columns += [f"{name}_direction_{axis}" for axis in "xyz"]
        columns += [f"{name}_physics_v{axis}" for axis in "xyz"]
        columns += [f"{name}_direction_valid", f"{name}_physics_state"]
    for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
        columns += [f"{name}_refined_endpoint_v{axis}" for axis in "xyz"]
    return columns


def engine_prediction_header() -> list[str]:
    return [
        "engine_latency_ms",
        "left_foot_contact",
        "right_foot_contact",
        "engine_contact_count",
        "engine_root_speed",
        "engine_state_reset",
        "engine_raw_mean_pose_error_degrees",
        "engine_raw_max_pose_error_degrees",
        "engine_raw_translation_error",
    ]


def main(refiner_factory=None, configure_parser=None, default_output_dir: Path | None = None) -> int:
    args = arguments(configure_parser, default_output_dir)
    if min(
        args.output_hz,
        args.tpose_seconds,
        args.gravity_seconds,
        args.bias_seconds,
        args.calibration_gyro_limit_deg,
        args.stale_ms,
        args.max_skew_ms,
    ) <= 0:
        print("Timing and validity limits must be positive", file=sys.stderr)
        return 2
    try:
        import serial
    except ImportError:
        print(f'Install pyserial with: "{sys.executable}" -m pip install pyserial', file=sys.stderr)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable", file=sys.stderr)
        return 2
    device = torch.device(
        "cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    runtime = None if args.adapter_only else OnlineFiveIMU(device)
    if refiner_factory is not None and runtime is None:
        print("Physics refinement requires inference; remove --adapter-only", file=sys.stderr)
        return 2
    refiner = None if refiner_factory is None else refiner_factory(args)
    if args.visualize and runtime is None:
        print("--visualize requires inference; remove --adapter-only", file=sys.stderr)
        return 2
    viewer = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = args.output_dir / f"filtered_input_{session}.csv"
    prediction_name = "prediction_physics" if refiner is not None else "prediction"
    prediction_path = args.output_dir / f"{prediction_name}_{session}.csv"
    diagnostics_path = args.output_dir / f"calibration_{session}.json"

    try:
        connection = serial.Serial(args.port, args.baudrate, timeout=0)
        connection.reset_input_buffer()
    except serial.SerialException as error:
        print(f"Cannot open {args.port}: {error}", file=sys.stderr)
        return 1
    if args.visualize:
        from src.visualization.live_smpl import LiveSMPLViewer

        viewer = LiveSMPLViewer()

    collector = TposeCollector(args.tpose_seconds, args.calibration_gyro_limit_deg)
    noise_filter = None
    synchronizer = FilteredSynchronizer()
    latest_raw_received: dict[str, float] = {}
    observed_data_ids: dict[int, float] = {}
    packet_rejections: Counter[str] = Counter()
    serial_buffer = bytearray()
    target_time_ms = None
    step_ms = 1000.0 / args.output_hz
    frame_index = 0
    status_frames = 0
    dropped_frames = 0
    last_status = time.monotonic()
    last_warning = 0.0
    print(f"Live adapter: {args.port} @ {args.baudrate}, output {args.output_hz:g} Hz")
    print(f"Inference device: {device if runtime else 'disabled'}")
    print(f"Calibration gyro limit: {args.calibration_gyro_limit_deg:g} deg/s")
    if viewer:
        print("3D reconstruction viewer: enabled")
    if refiner is not None:
        print("Post-prediction acceleration physics refinement: enabled")
    print("Wear all sensors, hold a T-pose, and remain still for about 9 seconds.")

    input_file = input_path.open("w", newline="", encoding="utf-8")
    prediction_file = prediction_path.open("w", newline="", encoding="utf-8") if runtime else None
    input_writer = csv.writer(input_file)
    prediction_writer = csv.writer(prediction_file) if prediction_file else None
    input_writer.writerow(input_header())
    if prediction_writer:
        header = prediction_header()
        if refiner is not None:
            header += physics_prediction_header()
            if getattr(refiner, "engine_enabled", False):
                header += engine_prediction_header()
        prediction_writer.writerow(header)

    try:
        while True:
            waiting = connection.in_waiting
            if waiting:
                serial_buffer.extend(connection.read(waiting))
            while b"\n" in serial_buffer:
                raw, _, remainder = serial_buffer.partition(b"\n")
                serial_buffer[:] = remainder
                parsed = diagnose_globalpose_packet(raw.decode("utf-8", errors="replace"))
                packet = parsed.packet
                if packet is None:
                    if parsed.node_id is not None:
                        observed_data_ids[parsed.node_id] = time.monotonic() * 1000
                        detail = parsed.reason
                        if parsed.reason == "field_count":
                            detail += f"({parsed.field_count})"
                        packet_rejections[f"node{parsed.node_id}:{detail}"] += 1
                    continue
                now_ms = time.monotonic() * 1000
                observed_data_ids[packet.node_id] = now_ms
                latest_raw_received[packet.sensor_name] = now_ms
                if noise_filter is None:
                    calibration = collector.add(packet, now_ms)
                    if calibration is not None:
                        noise_filter = FiveSensorNoiseFilter(
                            calibration,
                            NoiseFilterConfig(
                                gravity_seconds=args.gravity_seconds,
                                bias_seconds=args.bias_seconds,
                                calibration_gyro_limit_degrees=args.calibration_gyro_limit_deg,
                            ),
                        )
                        print("T-pose complete. Measuring gravity, bias, and per-axis noise; remain still.")
                    continue
                filtered = noise_filter.process(packet, now_ms)
                if filtered is not None:
                    synchronizer.add(filtered)

            now = time.monotonic()
            now_ms = now * 1000
            missing_raw = [name for name in SENSOR_ORDER if name not in latest_raw_received]
            stale_raw = [
                name for name in SENSOR_ORDER
                if name in latest_raw_received and now_ms - latest_raw_received[name] > args.stale_ms
            ]
            if missing_raw or stale_raw:
                calibration_active = noise_filter is not None or collector.started_ms is not None
                if calibration_active:
                    collector = TposeCollector(
                        args.tpose_seconds,
                        args.calibration_gyro_limit_deg,
                    )
                    noise_filter = None
                    synchronizer = FilteredSynchronizer()
                    latest_raw_received.clear()
                    target_time_ms = None
                    if runtime is not None:
                        runtime.reset()
                    if refiner is not None:
                        refiner.reset()
                    print("Calibration invalidated by missing/stale sensor; restarting from T-pose.")
                if now - last_warning >= 1:
                    recent_ids = sorted(
                        node for node, seen_ms in observed_data_ids.items()
                        if now_ms - seen_ms <= 2000
                    )
                    rejection_summary = dict(packet_rejections.most_common(4))
                    print(
                        f"WAITING: missing={missing_raw or 'none'} stale={stale_raw or 'none'} "
                        f"observed_ids={recent_ids or 'none'} "
                        f"rejected={rejection_summary or 'none'}"
                    )
                    packet_rejections.clear()
                    last_warning = now
                time.sleep(0.002)
                continue
            if noise_filter is None:
                time.sleep(0.002)
                continue
            if not noise_filter.ready:
                if now - last_status >= 1:
                    print("CALIBRATING:", noise_filter.stages())
                    last_status = now
                time.sleep(0.002)
                continue

            missing, stale, skew = synchronizer.status(now_ms, args.stale_ms, args.max_skew_ms)
            if missing or stale or skew > args.max_skew_ms:
                if now - last_warning >= 1:
                    print(f"WAITING: missing={missing or 'none'} stale={stale or 'none'} newest_skew_ms={skew:.1f}")
                    last_warning = now
                time.sleep(0.002)
                continue
            common_time = synchronizer.latest_common_time()
            earliest_time = synchronizer.earliest_common_time()
            if common_time is None or earliest_time is None:
                time.sleep(0.002)
                continue
            target_time_ms, newly_dropped = recover_live_target(
                target_time_ms, earliest_time, common_time, step_ms
            )
            if newly_dropped:
                dropped_frames += newly_dropped
                if now - last_warning >= 1:
                    print(
                        f"Synchronizer: dropped {newly_dropped} obsolete frames "
                        "to recover live output."
                    )
                    last_warning = now
            produced = False
            while target_time_ms <= common_time + 1e-6:
                live_frame = synchronizer.frame(target_time_ms)
                if live_frame is None:
                    target_time_ms = common_time
                    live_frame = synchronizer.frame(target_time_ms)
                    if live_frame is None:
                        break
                host_time = datetime.now().isoformat(timespec="milliseconds")
                input_writer.writerow(
                    [frame_index, host_time, live_frame.timestamp_ms, 1]
                    + [int(value) for value in live_frame.stationary]
                    + list(live_frame.sequences)
                    + live_frame.imu.reshape(-1).tolist()
                    + live_frame.angular_velocity.reshape(-1).tolist()
                )
                if runtime is not None:
                    output = runtime.forward_frame(live_frame.imu)
                    physics_diagnostics = None
                    if refiner is not None:
                        output, physics_diagnostics = refiner.refine(
                            output,
                            live_frame.imu[:, 6:9].numpy(),
                            live_frame.stationary,
                            live_frame.timestamp_ms,
                        )
                    if viewer is not None:
                        viewer.update(output["local_pose"], output["translation"])
                    prediction_row = (
                        [frame_index, host_time, live_frame.timestamp_ms, output["latency_ms"]]
                        + output["translation"].reshape(-1).tolist()
                        + output["global_pose"].reshape(-1).tolist()
                    )
                    if physics_diagnostics is not None:
                        prediction_row += (
                            physics_diagnostics["raw_translation"].reshape(-1).tolist()
                            + physics_diagnostics["refined_velocity"].reshape(-1).tolist()
                            + physics_diagnostics["acceleration_velocity"].reshape(-1).tolist()
                            + physics_diagnostics["direction"].reshape(-1).tolist()
                            + [
                                int(physics_diagnostics["direction_valid"]),
                                physics_diagnostics["state"],
                                physics_diagnostics["dt"],
                            ]
                        )
                        for name in SENSOR_ORDER:
                            sensor = physics_diagnostics["sensor_diagnostics"][name]
                            prediction_row += (
                                sensor["direction"].reshape(-1).tolist()
                                + sensor["velocity"].reshape(-1).tolist()
                                + [int(sensor["direction_valid"]), sensor["state"]]
                            )
                        for name in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
                            prediction_row += physics_diagnostics["endpoint_velocities"][name].reshape(-1).tolist()
                        if "engine_diagnostics" in physics_diagnostics:
                            engine = physics_diagnostics["engine_diagnostics"]
                            prediction_row += [
                                engine["latency_ms"],
                                int(engine["left_foot_contact"]),
                                int(engine["right_foot_contact"]),
                                engine["contact_count"],
                                engine["root_speed"],
                                int(engine["state_reset"]),
                                engine["raw_mean_pose_error_degrees"],
                                engine["raw_max_pose_error_degrees"],
                                engine["raw_translation_error"],
                            ]
                    prediction_writer.writerow(prediction_row)
                frame_index += 1
                status_frames += 1
                target_time_ms += step_ms
                produced = True
                for name in SENSOR_ORDER:
                    while len(synchronizer.buffers[name]) > 3 and synchronizer.buffers[name][1][0] < target_time_ms:
                        synchronizer.buffers[name].popleft()
            if now - last_status >= 1:
                rate = status_frames / max(now - last_status, 1e-6)
                print(
                    f"RUNNING: frames={frame_index} output={rate:.1f} Hz "
                    f"skew={skew:.1f} ms dropped={dropped_frames}"
                )
                input_file.flush()
                if prediction_file:
                    prediction_file.flush()
                diagnostics_path.write_text(json.dumps(noise_filter.diagnostics(), indent=2), encoding="utf-8")
                status_frames = 0
                last_status = now
            if not produced:
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException as error:
        print(f"Serial error: {error}", file=sys.stderr)
        return 1
    finally:
        connection.close()
        input_file.close()
        if prediction_file:
            prediction_file.close()
        if viewer is not None:
            viewer.close()
        if noise_filter is not None:
            diagnostics_path.write_text(json.dumps(noise_filter.diagnostics(), indent=2), encoding="utf-8")
        print(f"Filtered input: {input_path}")
        if runtime:
            print(f"Predictions: {prediction_path}")
        print(f"Calibration diagnostics: {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
