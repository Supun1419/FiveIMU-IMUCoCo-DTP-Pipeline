"""Run the unchanged five-IMU DL pipeline from firmware MQTT batches."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import paho.mqtt.client as mqtt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRMWARE_DIR = PROJECT_ROOT / "firmware"
DEFAULT_BROKER = "192.168.1.155"
DEFAULT_MQTT_PORT = 1883
DEFAULT_TOPIC = "FYP_IMU_Session/#"
SOURCE_TO_NODE = {
    "Center": 0,
    "Node1": 1,
    "Node2": 2,
    "Node3": 3,
    "Node4": 4,
}
NODE_TO_SOURCE = {node_id: source for source, node_id in SOURCE_TO_NODE.items()}

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import realtime  # noqa: E402
from src.input.orientation_adapter import (  # noqa: E402
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
)


@dataclass(frozen=True)
class MQTTSettings:
    broker: str
    port: int
    topic: str
    username: str
    password: str
    queue_size: int
    sample_hz: float
    max_buffer_ms: float
    playback_mode: str
    orientation_jump_degrees: float


def canonical_quaternion(values) -> tuple[float, float, float, float]:
    """Match world_relative_viewer.canonical without importing pygame."""
    quaternion = tuple(float(value) for value in values)
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    normalized = (
        (1.0, 0.0, 0.0, 0.0)
        if magnitude < 1e-9
        else tuple(value / magnitude for value in quaternion)
    )
    return tuple(-value for value in normalized) if normalized[0] < 0 else normalized


def row_to_globalpose_line(row: dict[str, object]) -> bytes | None:
    """Adapt one parsed firmware row to the validated live packet contract."""
    node_id = SOURCE_TO_NODE.get(str(row["source"]))
    if node_id is None:
        return None
    quaternion = canonical_quaternion(
        (row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"])
    )
    fields = [
        "DATA",
        node_id,
        int(row["timestamp_ms"]),
        "mqtt",
        0,
        "mqtt",
        int(row["sequence"]),
        float(row["accel_x"]),
        float(row["accel_y"]),
        float(row["accel_z"]),
        float(row["gyro_x"]),
        float(row["gyro_y"]),
        float(row["gyro_z"]),
        0,
        0,
        0,
        *quaternion,
        "mqtt",
    ]
    return (",".join(str(value) for value in fields) + "\n").encode("utf-8")


def replace_packet_timestamp(line: bytes, timestamp_ms: int) -> bytes:
    """Replace only the GlobalPose timestamp field in an adapted packet."""
    fields = line.decode("utf-8").rstrip("\r\n").split(",")
    if len(fields) < 3 or fields[0] != "DATA":
        raise ValueError("Cannot align timestamp in malformed GlobalPose packet")
    fields[2] = str(int(timestamp_ms))
    return (",".join(fields) + "\n").encode("utf-8")


@dataclass
class OrientationState:
    correction: np.ndarray
    previous: np.ndarray


class OrientationContinuityCorrector:
    """Rebase discontinuous firmware orientations into a continuous world frame."""

    def __init__(self, jump_degrees: float, sample_hz: float = 20.0) -> None:
        self.jump_degrees = float(jump_degrees)
        self.sample_period_seconds = 1.0 / float(sample_hz)
        self.states: dict[int, OrientationState] = {}
        self.rebases: dict[int, int] = {}

    def correct(self, row: dict[str, object]) -> tuple[dict[str, object], float | None]:
        node_id = SOURCE_TO_NODE.get(str(row["source"]))
        if node_id is None or self.jump_degrees <= 0:
            return row, None

        raw = quaternion_wxyz_to_matrix(
            np.asarray(
                canonical_quaternion(
                    (row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"])
                )
            )
        )
        state = self.states.get(node_id)
        if state is None:
            output = raw
            self.states[node_id] = OrientationState(np.eye(3), output)
            return row, None

        output = state.correction @ raw
        relative = state.previous.T @ output
        angle = math.degrees(
            math.acos(float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
        )
        gyro_rate = math.sqrt(
            sum(float(row[f"gyro_{axis}"]) ** 2 for axis in ("x", "y", "z"))
        )
        plausible_degrees = gyro_rate * self.sample_period_seconds
        threshold = max(self.jump_degrees, plausible_degrees * 4.0 + 5.0)
        rebased_angle = None
        if angle > threshold:
            state.correction = state.previous @ raw.T
            output = state.previous
            rebased_angle = angle
            self.rebases[node_id] = self.rebases.get(node_id, 0) + 1

        state.previous = output
        quaternion = matrix_to_quaternion_wxyz(output)
        corrected = dict(row)
        for key, value in zip(("quat_w", "quat_x", "quat_y", "quat_z"), quaternion):
            corrected[key] = float(value)
        return corrected, rebased_angle


def load_firmware_parser(firmware_dir: Path) -> ModuleType:
    parser_path = firmware_dir / "mqtt_firmware_subscriber.py"
    if not parser_path.is_file():
        raise FileNotFoundError(f"Firmware MQTT parser not found: {parser_path}")
    spec = importlib.util.spec_from_file_location("fyp_mqtt_firmware_subscriber", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load firmware MQTT parser: {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enqueue_latest(samples: queue.Queue[bytes], item: bytes) -> bool:
    """Use the same bounded newest-sample behavior as the firmware viewer."""
    try:
        samples.put_nowait(item)
        return False
    except queue.Full:
        pass
    try:
        samples.get_nowait()
    except queue.Empty:
        pass
    try:
        samples.put_nowait(item)
    except queue.Full:
        pass
    return True


class FiveNodeBatchGate:
    """Release MQTT samples only when every sensor has supplied a batch."""

    def __init__(self, capacity: int) -> None:
        per_node_capacity = max(1, capacity // len(SOURCE_TO_NODE))
        self.pending = {
            node_id: deque(maxlen=per_node_capacity)
            for node_id in SOURCE_TO_NODE.values()
        }

    def add(
        self, node_id: int, timestamp_ms: int, line: bytes
    ) -> list[bytes]:
        return self.add_batch(node_id, [(timestamp_ms, line)])

    def add_batch(
        self, node_id: int, items: list[tuple[int, bytes]]
    ) -> list[bytes]:
        self.pending[node_id].extend(items)
        if not all(self.pending.values()):
            return []
        complete_samples = min(len(node_items) for node_items in self.pending.values())
        ready = []
        for _ in range(complete_samples):
            for node_id_in_order in SOURCE_TO_NODE.values():
                node_items = self.pending[node_id_in_order]
                _, line = node_items.popleft()
                ready.append(line)
        return ready

    def pending_counts(self) -> dict[str, int]:
        return {
            NODE_TO_SOURCE[node_id]: len(node_items)
            for node_id, node_items in self.pending.items()
        }

    def missing_sources(self) -> list[str]:
        return [
            NODE_TO_SOURCE[node_id]
            for node_id, node_items in self.pending.items()
            if not node_items
        ]


class MQTTSerialBridge:
    """Expose MQTT firmware rows through the interface consumed by realtime.py."""

    def __init__(self, settings: MQTTSettings, firmware: ModuleType) -> None:
        self.settings = settings
        self.firmware = firmware
        live_groups = max(
            2, math.ceil(settings.sample_hz * settings.max_buffer_ms / 1000.0)
        )
        self.samples: queue.Queue[bytes] = queue.Queue(maxsize=live_groups)
        self.batch_gate = FiveNodeBatchGate(settings.queue_size)
        self.buffer = bytearray()
        self.stopping = False
        self.connected = False
        self.loop_started = False
        self.next_release_at = 0.0
        self.batches = 0
        self.sample_count = 0
        self.invalid = 0
        self.released_cycles = 0
        self.dropped_groups = 0
        self.reported_dropped_groups = 0
        self.last_gate_status = 0.0
        self.last_release_at = time.monotonic()
        self.aligned_timestamp_ms: float | None = None
        self.orientation_corrector = OrientationContinuityCorrector(
            settings.orientation_jump_degrees, settings.sample_hz
        )

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="python_imucoco_five_imu_dl",
        )
        if settings.username:
            self.client.username_pw_set(settings.username, settings.password)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect(settings.broker, settings.port, keepalive=30)

    def _ensure_loop_started(self) -> None:
        if not self.loop_started:
            self.client.loop_start()
            self.loop_started = True

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            client.subscribe(self.settings.topic, qos=0)
            self.connected = True
            print(
                f"MQTT connected to {self.settings.broker}:{self.settings.port}; "
                f"subscribed to {self.settings.topic}"
            )
            print(f"MQTT playback mode: {self.settings.playback_mode}")
        else:
            self.connected = False
            print(f"MQTT connection failed: {reason_code}", file=sys.stderr)

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ) -> None:
        self.connected = False
        if not self.stopping:
            print(f"MQTT disconnected: {reason_code}; reconnecting", file=sys.stderr)

    def _on_message(self, client, userdata, message) -> None:
        try:
            rows = self.firmware.parse_batch(message.topic, message.payload)
            accepted = 0
            released = 0
            grouped: dict[int, list[tuple[int, bytes]]] = {}
            for row in rows:
                row, rebased_angle = self.orientation_corrector.correct(row)
                if rebased_angle is not None:
                    node_id = SOURCE_TO_NODE[str(row["source"])]
                    print(
                        f"MQTT ORIENTATION REBASE: {row['source']} node={node_id} "
                        f"jump={rebased_angle:.1f} deg "
                        f"count={self.orientation_corrector.rebases[node_id]}"
                    )
                line = row_to_globalpose_line(row)
                if line is None:
                    continue
                node_id = SOURCE_TO_NODE[str(row["source"])]
                grouped.setdefault(node_id, []).append(
                    (int(row["timestamp_ms"]), line)
                )
                accepted += 1
            for node_id, items in grouped.items():
                ready = self.batch_gate.add_batch(node_id, items)
                for start in range(0, len(ready), len(SOURCE_TO_NODE)):
                    sample_group = ready[start : start + len(SOURCE_TO_NODE)]
                    if len(sample_group) == len(SOURCE_TO_NODE):
                        if self.aligned_timestamp_ms is None:
                            self.aligned_timestamp_ms = time.monotonic() * 1000.0
                        else:
                            self.aligned_timestamp_ms += 1000.0 / self.settings.sample_hz
                        aligned_group = [
                            replace_packet_timestamp(line, round(self.aligned_timestamp_ms))
                            for line in sample_group
                        ]
                        dropped = enqueue_latest(
                            self.samples, b"".join(aligned_group)
                        )
                        self.dropped_groups += int(dropped)
                        released += len(sample_group)
            self.batches += 1
            self.sample_count += accepted
            now = time.monotonic()
            if released:
                self.released_cycles += 1
                self.last_release_at = now
            if self.dropped_groups > self.reported_dropped_groups:
                print(
                    f"MQTT LIVE BUFFER: dropped_groups={self.dropped_groups} "
                    f"queued={self.samples.qsize()}/{self.samples.maxsize}"
                )
                self.reported_dropped_groups = self.dropped_groups
            missing = self.batch_gate.missing_sources()
            gate_stalled = now - self.last_release_at >= 2.0
            if missing and gate_stalled and now - self.last_gate_status >= 1.0:
                print(
                    f"MQTT GATE: waiting for {missing}; "
                    f"pending={self.batch_gate.pending_counts()}"
                )
                self.last_gate_status = now
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            self.invalid += 1
            print(f"Invalid MQTT payload on {message.topic}: {error}", file=sys.stderr)

    def _fill_buffer(self) -> None:
        now = time.monotonic()
        paced = self.settings.playback_mode == "paced"
        if self.buffer or (
            paced and self.next_release_at and now < self.next_release_at
        ):
            return
        try:
            self.buffer.extend(self.samples.get_nowait())
        except queue.Empty:
            self.next_release_at = 0.0
            return
        if paced:
            if not self.next_release_at:
                self.next_release_at = now
            self.next_release_at += 1.0 / self.settings.sample_hz

    @property
    def in_waiting(self) -> int:
        self._ensure_loop_started()
        self._fill_buffer()
        return len(self.buffer)

    def read(self, size: int) -> bytes:
        count = min(max(int(size), 0), len(self.buffer))
        data = bytes(self.buffer[:count])
        del self.buffer[:count]
        return data

    def reset_input_buffer(self) -> None:
        self.buffer.clear()
        self.next_release_at = 0.0
        while True:
            try:
                self.samples.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        if self.stopping:
            return
        self.stopping = True
        if self.loop_started:
            self.client.loop_stop()
        self.client.disconnect()
        print(
            f"MQTT stopped: batches={self.batches} samples={self.sample_count} "
            f"released_cycles={self.released_cycles} "
            f"dropped_groups={self.dropped_groups} invalid={self.invalid}"
        )


def mqtt_preparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--queue-size", type=int, default=1000)
    parser.add_argument("--mqtt-sample-hz", type=float, default=20.0)
    parser.add_argument("--mqtt-max-buffer-ms", type=float, default=1000.0)
    parser.add_argument("--mqtt-orientation-jump-deg", type=float, default=30.0)
    parser.add_argument(
        "--mqtt-playback-mode",
        choices=("catch-up", "paced"),
        default="catch-up",
    )
    parser.add_argument("--firmware-dir", type=Path, default=DEFAULT_FIRMWARE_DIR)
    return parser


def main() -> int:
    mqtt_args, _ = mqtt_preparser().parse_known_args()
    if min(
        mqtt_args.mqtt_port,
        mqtt_args.queue_size,
        mqtt_args.mqtt_sample_hz,
        mqtt_args.mqtt_max_buffer_ms,
        mqtt_args.mqtt_orientation_jump_deg,
    ) <= 0:
        print(
            "MQTT port, queue, sample rate, buffer duration, and orientation "
            "jump threshold must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        firmware = load_firmware_parser(mqtt_args.firmware_dir)
    except (FileNotFoundError, ImportError) as error:
        print(error, file=sys.stderr)
        return 2

    settings = MQTTSettings(
        broker=mqtt_args.broker,
        port=mqtt_args.mqtt_port,
        topic=mqtt_args.topic,
        username=mqtt_args.username,
        password=mqtt_args.password,
        queue_size=mqtt_args.queue_size,
        sample_hz=mqtt_args.mqtt_sample_hz,
        max_buffer_ms=mqtt_args.mqtt_max_buffer_ms,
        playback_mode=mqtt_args.mqtt_playback_mode,
        orientation_jump_degrees=mqtt_args.mqtt_orientation_jump_deg,
    )

    def configure_parser(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--broker", default=DEFAULT_BROKER)
        parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
        parser.add_argument("--topic", default=DEFAULT_TOPIC)
        parser.add_argument("--username", default="")
        parser.add_argument("--password", default="")
        parser.add_argument("--queue-size", type=int, default=1000)
        parser.add_argument("--mqtt-sample-hz", type=float, default=20.0)
        parser.add_argument("--mqtt-max-buffer-ms", type=float, default=1000.0)
        parser.add_argument("--mqtt-orientation-jump-deg", type=float, default=30.0)
        parser.add_argument(
            "--mqtt-playback-mode",
            choices=("catch-up", "paced"),
            default="catch-up",
        )
        parser.add_argument("--firmware-dir", type=Path, default=DEFAULT_FIRMWARE_DIR)
        parser.set_defaults(
            port=settings.broker,
            baudrate=settings.port,
            stale_ms=2500.0,
            max_skew_ms=1200.0,
            output_hz=settings.sample_hz,
        )
        parser._option_string_actions["--port"].help = argparse.SUPPRESS
        parser._option_string_actions["--baudrate"].help = argparse.SUPPRESS

    import serial

    original_serial = serial.Serial

    def open_mqtt_bridge(*args, **kwargs):
        try:
            return MQTTSerialBridge(settings, firmware)
        except (ConnectionError, OSError) as error:
            raise serial.SerialException(
                f"Cannot connect to MQTT broker {settings.broker}:{settings.port}: {error}"
            ) from error

    serial.Serial = open_mqtt_bridge
    try:
        return realtime.main(
            configure_parser=configure_parser,
            default_output_dir=PROJECT_ROOT / "experiments" / "live_5imu_mqtt",
        )
    finally:
        serial.Serial = original_serial


if __name__ == "__main__":
    raise SystemExit(main())
