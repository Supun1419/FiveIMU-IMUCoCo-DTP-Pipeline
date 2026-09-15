"""Non-blocking CPU-rendered viewer for live SMPL reconstructions."""

from __future__ import annotations

import multiprocessing as mp
import queue
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _viewer_process(frames: mp.Queue, stop: mp.Event) -> None:
    import os
    import sys
    import time

    import matplotlib

    matplotlib.use(os.environ.get("IMUCOCO_VIEWER_BACKEND", "TkAgg"), force=True)
    import matplotlib.pyplot as plt
    import torch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    upstream_root = PROJECT_ROOT / "third_party" / "IMUCoCo"
    sys.path.insert(0, str(upstream_root))
    import articulate as art

    body_model = art.ParametricModel(str(PROJECT_ROOT / "smpl" / "SMPL_MALE.pkl"))
    identity_pose = torch.eye(3).expand(1, 24, 3, 3)
    _, _, vertices = body_model.forward_kinematics(identity_pose, calc_mesh=True)
    points = vertices[0].numpy()
    faces = np.asarray(body_model.face, dtype=np.int32)

    # Matplotlib treats Z as vertical; swap SMPL's Y-up coordinates for display.
    display_points = points[:, (0, 2, 1)]
    figure = plt.figure(figsize=(8.5, 8.5), facecolor="#16191d")
    figure.canvas.manager.set_window_title("IMUCoCo Live Reconstruction")
    axes = figure.add_subplot(111, projection="3d", facecolor="#20262c")
    surface = Poly3DCollection(
        display_points[faces],
        facecolor="#59b7c9",
        edgecolor="none",
        linewidth=0,
        antialiased=False,
    )
    axes.add_collection3d(surface)
    axes.set_xlabel("X", color="white")
    axes.set_ylabel("Z", color="white")
    axes.set_zlabel("Y", color="white")
    axes.tick_params(colors="#b8c0c8")
    axes.grid(True, color="#59636d", alpha=0.4)
    axes.set_box_aspect((1.3, 1.3, 2.2))
    axes.view_init(elev=8, azim=-90)
    figure.suptitle("Live SMPL reconstruction", color="white", fontsize=13)
    figure.text(0.02, 0.02, "Drag: rotate   Scroll: zoom   Q: close", color="#cbd2d9")
    closed = False

    def close_viewer(_event=None) -> None:
        nonlocal closed
        closed = True

    def key_pressed(event) -> None:
        if event.key and event.key.lower() == "q":
            close_viewer()
            plt.close(figure)

    figure.canvas.mpl_connect("close_event", close_viewer)
    figure.canvas.mpl_connect("key_press_event", key_pressed)
    plt.show(block=False)
    last_draw = 0.0

    def follow_root(translation: np.ndarray) -> None:
        center = translation.reshape(3)[[0, 2, 1]]
        axes.set_xlim(center[0] - 1.25, center[0] + 1.25)
        axes.set_ylim(center[1] - 1.25, center[1] + 1.25)
        axes.set_zlim(center[2] - 1.15, center[2] + 1.15)

    follow_root(np.zeros(3, dtype=np.float32))

    try:
        while not stop.is_set() and not closed:
            newest = None
            try:
                newest = frames.get(timeout=0.01)
                while True:
                    newest = frames.get_nowait()
            except queue.Empty:
                pass
            except (EOFError, OSError):
                break
            if newest is not None:
                pose, translation = newest
                pose_tensor = torch.from_numpy(pose).float().reshape(1, 24, 3, 3)
                tran_tensor = torch.from_numpy(translation).float().reshape(1, 3)
                _, _, vertices = body_model.forward_kinematics(
                    pose_tensor, tran=tran_tensor, calc_mesh=True
                )
                display_points = vertices[0].numpy()[:, (0, 2, 1)]
                surface.set_verts(display_points[faces], closed=True)
                follow_root(translation)
            now = time.monotonic()
            if newest is not None and now - last_draw >= 1.0 / 15.0:
                figure.canvas.draw_idle()
                last_draw = now
            figure.canvas.flush_events()
            plt.pause(0.001)
    finally:
        plt.close(figure)


class LiveSMPLViewer:
    """Own a separate GUI process and retain only the latest predicted frame."""

    def __init__(self) -> None:
        context = mp.get_context("spawn")
        self._frames = context.Queue(maxsize=2)
        self._stop = context.Event()
        self._process = context.Process(
            target=_viewer_process,
            args=(self._frames, self._stop),
            name="imucoco-smpl-viewer",
            daemon=True,
        )
        self._process.start()

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def update(self, local_pose, translation) -> None:
        if not self.alive:
            return
        item = (
            local_pose.detach().cpu().numpy().astype(np.float32, copy=False),
            translation.detach().cpu().numpy().astype(np.float32, copy=False),
        )
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop.set()
        if self._process.is_alive():
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._frames.close()
