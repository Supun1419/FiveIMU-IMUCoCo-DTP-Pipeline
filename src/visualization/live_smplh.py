"""Non-blocking SMPL-H mesh viewer for IMUCoCo body predictions."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "smplh" / "neutral" / "model.npz"


def smpl_pose_to_smplh_axis_angles(local_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map SMPL's root and first 21 body joints to SMPL-H axis angles."""
    from scipy.spatial.transform import Rotation

    pose = np.asarray(local_pose, dtype=np.float32)
    if pose.shape != (24, 3, 3):
        raise ValueError(f"Expected SMPL pose shape (24, 3, 3), got {pose.shape}")
    rotation_vectors = Rotation.from_matrix(pose[:22]).as_rotvec().astype(np.float32)
    return rotation_vectors[0], rotation_vectors[1:22].reshape(63)


def _load_smplh_model(model_path: Path):
    import smplx
    import torch
    from smplx.utils import Struct

    if model_path.suffix.lower() == ".npz":
        model_data = dict(np.load(model_path, allow_pickle=True))
        model_data.setdefault("hands_componentsl", np.zeros((6, 45), dtype=np.float32))
        model_data.setdefault("hands_componentsr", np.zeros((6, 45), dtype=np.float32))
        model_data.setdefault("hands_meanl", np.zeros(45, dtype=np.float32))
        model_data.setdefault("hands_meanr", np.zeros(45, dtype=np.float32))
        return smplx.SMPLH(
            model_path="",
            data_struct=Struct(**model_data),
            gender="neutral",
            use_pca=False,
            flat_hand_mean=True,
            batch_size=1,
        ).to(torch.device("cpu"))
    return smplx.SMPLH(
        model_path=str(model_path),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(torch.device("cpu"))


def _viewer_process(frames: mp.Queue, stop: mp.Event, model_path_text: str) -> None:
    import time

    import matplotlib

    matplotlib.use(os.environ.get("IMUCOCO_VIEWER_BACKEND", "TkAgg"), force=True)
    import matplotlib.pyplot as plt
    import torch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    torch.set_num_threads(1)
    model_path = Path(model_path_text)
    body_model = _load_smplh_model(model_path)
    faces = np.asarray(body_model.faces, dtype=np.int32)
    zeros3 = torch.zeros(1, 3)
    zeros45 = torch.zeros(1, 45)
    zeros63 = torch.zeros(1, 63)

    def vertices_for(
        global_orient: np.ndarray | None = None,
        body_pose: np.ndarray | None = None,
        translation: np.ndarray | None = None,
    ) -> np.ndarray:
        with torch.inference_mode():
            output = body_model(
                betas=torch.zeros(1, 10),
                global_orient=zeros3 if global_orient is None else torch.from_numpy(global_orient).reshape(1, 3),
                body_pose=zeros63 if body_pose is None else torch.from_numpy(body_pose).reshape(1, 63),
                left_hand_pose=zeros45,
                right_hand_pose=zeros45,
                transl=zeros3 if translation is None else torch.from_numpy(translation).reshape(1, 3),
            )
        return output.vertices[0].numpy()

    points = vertices_for()
    display_points = points[:, (0, 2, 1)]
    figure = plt.figure(figsize=(8.5, 8.5), facecolor="#16191d")
    figure.canvas.manager.set_window_title("IMUCoCo Live SMPL-H Reconstruction")
    axes = figure.add_subplot(111, projection="3d", facecolor="#20262c")
    surface = Poly3DCollection(
        display_points[faces],
        facecolor="#68b7a2",
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
    figure.suptitle("Live SMPL-H reconstruction", color="white", fontsize=13)
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
                global_orient, body_pose = smpl_pose_to_smplh_axis_angles(pose)
                points = vertices_for(global_orient, body_pose, translation)
                display_points = points[:, (0, 2, 1)]
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


class LiveSMPLHViewer:
    """Render the latest body prediction as an SMPL-H mesh in another process."""

    def __init__(self, model_path: Path | None = None) -> None:
        selected = model_path or Path(
            os.environ.get("IMUCOCO_SMPLH_MODEL", str(DEFAULT_MODEL_PATH))
        )
        if not selected.is_file():
            raise FileNotFoundError(f"SMPL-H model not found: {selected}")
        context = mp.get_context("spawn")
        self._frames = context.Queue(maxsize=2)
        self._stop = context.Event()
        self._process = context.Process(
            target=_viewer_process,
            args=(self._frames, self._stop, str(selected)),
            name="imucoco-smplh-viewer",
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
