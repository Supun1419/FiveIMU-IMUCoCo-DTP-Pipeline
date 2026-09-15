#!/usr/bin/env python3
"""Subscribe to ESP32 firmware MQTT batches and save IMU samples to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Missing dependency. Install it with: pip install paho-mqtt", file=sys.stderr)
    raise SystemExit(1)


DEFAULT_BROKER = "127.0.0.1"
DEFAULT_PORT = 1883
DEFAULT_TOPIC = "FYP_IMU_Session/#"
CSV_FIELDS = [
    "received_time",
    "topic",
    "source",
    "batch",
    "batch_t0",
    "sequence",
    "timestamp_ms",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]


def source_from_topic(topic: str) -> str:
    leaf = topic.rsplit("/", 1)[-1]
    if leaf == "Center_Batch":
        return "Center"
    if leaf.startswith("Node") and leaf.endswith("_Batch"):
        return leaf.removesuffix("_Batch")
    return leaf


def parse_batch(topic: str, payload: bytes) -> list[dict[str, object]]:
    data = json.loads(payload.decode("utf-8"))
    batch = data["b"]
    batch_t0 = data["t0"]
    samples = data["d"]
    source = source_from_topic(topic)
    received_time = datetime.now().isoformat(timespec="milliseconds")

    rows = []
    for sample in samples:
        if len(sample) != 12:
            raise ValueError(f"Expected 12 values per sample, got {len(sample)}")

        rows.append(
            {
                "received_time": received_time,
                "topic": topic,
                "source": source,
                "batch": batch,
                "batch_t0": batch_t0,
                "sequence": sample[0],
                "timestamp_ms": sample[1],
                "accel_x": sample[2],
                "accel_y": sample[3],
                "accel_z": sample[4],
                "gyro_x": sample[5],
                "gyro_y": sample[6],
                "gyro_z": sample[7],
                "quat_w": sample[8],
                "quat_x": sample[9],
                "quat_y": sample[10],
                "quat_z": sample[11],
            }
        )

    return rows


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read IMU batches published by the ESP32 firmware over MQTT."
    )
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Default: mqtt_imu_capture_YYYYMMDD_HHMMSS.csv",
    )
    parser.add_argument("--username", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--raw", action="store_true", help="Print full JSON payloads.")
    return parser.parse_args()


class App:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop = False
        self.received_batches = 0
        self.received_samples = 0
        self.invalid = 0
        self.started = time.monotonic()

        if args.output is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = Path(__file__).resolve().parent / f"mqtt_imu_capture_{stamp}.csv"

        self.csv_file = args.output.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.writer.writeheader()

    def close(self) -> None:
        self.csv_file.flush()
        self.csv_file.close()

    def handle_message(self, message: mqtt.MQTTMessage) -> None:
        try:
            rows = parse_batch(message.topic, message.payload)
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            self.invalid += 1
            print(f"Invalid payload on {message.topic}: {error}", file=sys.stderr)
            return

        self.writer.writerows(rows)
        self.csv_file.flush()
        self.received_batches += 1
        self.received_samples += len(rows)

        source = rows[0]["source"] if rows else source_from_topic(message.topic)
        batch = rows[0]["batch"] if rows else "?"
        print(
            f"{source} batch {batch}: {len(rows)} samples "
            f"({self.received_samples} total) -> {self.args.output}"
        )

        if self.args.raw:
            print(message.payload.decode("utf-8", errors="replace"))


def create_client(app: App) -> mqtt.Client:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="python_imu_subscriber")

    if app.args.username:
        client.username_pw_set(app.args.username, app.args.password)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(app.args.topic)
            print(
                f"Connected to {app.args.broker}:{app.args.port}; "
                f"subscribed to {app.args.topic}"
            )
        else:
            print(f"MQTT connection failed: {reason_code}", file=sys.stderr)

    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
        if not app.stop:
            print(f"MQTT disconnected: {reason_code}", file=sys.stderr)

    def on_message(client, userdata, message):
        app.handle_message(message)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    return client


def main() -> int:
    args = arguments()
    app = App(args)

    def request_stop(signum, frame):
        app.stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    client = create_client(app)

    try:
        print(f"Saving data to: {args.output}")
        client.connect(args.broker, args.port, keepalive=30)
        client.loop_start()

        while not app.stop:
            time.sleep(0.2)
    finally:
        app.stop = True
        client.loop_stop()
        client.disconnect()
        app.close()

    elapsed = max(time.monotonic() - app.started, 0.001)
    print(
        f"Stopped. Batches: {app.received_batches}, samples: {app.received_samples}, "
        f"invalid: {app.invalid}, rate: {app.received_samples / elapsed:.1f} samples/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
