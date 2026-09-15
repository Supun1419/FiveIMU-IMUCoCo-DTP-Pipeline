"""GPU-capable runtime wrapper for the official IMUCoCo and DTP checkpoints."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = PROJECT_ROOT / "third_party" / "IMUCoCo"
if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

import articulate as art  # noqa: E402
from models.dtp import Poser  # noqa: E402
from models.imucoco import IMUCoCo  # noqa: E402
from utils import imu_config  # noqa: E402


SENSOR_NAMES = (
    "left_wrist",
    "right_wrist",
    "torso",
    "left_ankle",
    "right_ankle",
)
SENSOR_VERTEX_IDS = (1961, 5424, 3021, 1176, 4662)


def placement_coordinates(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        imu_config.vertex_coordinates[list(SENSOR_VERTEX_IDS)],
        dtype=torch.float32,
        device=device,
    )


def build_models(
    device: torch.device, *, online: bool
) -> tuple[IMUCoCo, Poser]:
    vertex_coordinates = torch.tensor(
        imu_config.vertex_coordinates_with_category, dtype=torch.float32, device=device
    )
    joint_coordinates = torch.tensor(
        imu_config.joint_coordinates_with_category, dtype=torch.float32, device=device
    )
    imucoco = IMUCoCo(
        coordinate_origins=joint_coordinates,
        coordinate_max=vertex_coordinates[:, 1:].max(dim=0).values,
        coordinate_min=vertex_coordinates[:, 1:].min(dim=0).values,
        smpl_mesh_coordinates=vertex_coordinates,
        n_hidden=128,
        n_kr_hidden=32,
        n_mfe_layers=2,
        n_jnm_layers=3,
        n_sce_freq=4,
        n_sce_emb=40,
        online_mode=online,
        joint_node_allocation_map=str(
            PROJECT_ROOT / "saved_checkpoints" / "all_error_loss_map.pth"
        ),
        joint_node_max_err_tolerance=-1,
    ).to(device)
    state = torch.load(
        PROJECT_ROOT / "saved_checkpoints" / "imucoco_best.pth",
        map_location=device,
    )
    if online:
        imucoco.load_offline_state_dict_to_online_model(state)
        imucoco.prepare_parallel_sce_implementation()
        imucoco.prepare_parallel_joint_node_implementation()
    else:
        imucoco.load_state_dict(state, strict=False)
    imucoco.freeze()
    imucoco.eval()

    poser = Poser(
        joint_feature_dim=128,
        n_hidden=300,
        n_glb=40,
        num_layer=3,
        n_total_devices=24,
        load_tran_module=True,
    ).to(device)
    poser.load_state_dict(
        torch.load(
            PROJECT_ROOT / "saved_checkpoints" / "poser_dtp_best.pth",
            map_location=device,
        ),
        strict=False,
    )
    poser.body_model = art.ParametricModel(
        str(PROJECT_ROOT / "smpl" / "SMPL_MALE.pkl"), device=device
    )
    poser.eval()

    imucoco.set_current_device_coordinates(placement_coordinates(device))
    imucoco.buffer_placement_codes_with_current_devices(parallel=online)
    return imucoco, poser


class OnlineFiveIMU:
    """Stateful single-frame inference for `[5, 9]` calibrated inputs."""

    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.imucoco, self.poser = build_models(self.device, online=True)
        self.reset()

    def reset(self) -> None:
        """Reset recurrent and translation state after recalibration/reconnection."""
        initial_rotation = torch.eye(3, device=self.device).expand(1, 24, 3, 3)
        initial_r6d = initial_rotation[:, :, :, :2].transpose(2, 3).flatten(2)
        self.mfe_hidden = None
        self.jnm_hidden = None
        self.pose_hidden = self.poser.init_hidden_states(
            v_init=torch.zeros(1, 24, 3, device=self.device),
            glb_init=initial_r6d,
        )
        self.translation = None
        self.poser.saved_previous_joint_pos = None

    def forward_frame(self, frame: torch.Tensor) -> dict[str, torch.Tensor | float]:
        if tuple(frame.shape) != (5, 9):
            raise ValueError(f"Expected frame shape (5, 9), got {tuple(frame.shape)}")
        if not torch.isfinite(frame).all():
            raise ValueError("Input frame contains NaN or infinite values")
        started = time.perf_counter()
        with torch.inference_mode():
            features, self.mfe_hidden, self.jnm_hidden = (
                self.imucoco.inference_time_forward_mesh_online(
                    frame.to(self.device).view(1, 1, 5, 9),
                    self.mfe_hidden,
                    self.jnm_hidden,
                )
            )
            local_pose, global_pose, translation, self.pose_hidden = (
                self.poser.forward_online(
                    features,
                    self.pose_hidden,
                    current_tran=self.translation,
                    compute_tran="transpose",
                )
            )
        self.translation = translation
        return {
            "local_pose": local_pose[0, -1].detach().cpu(),
            "global_pose": global_pose[0, -1].detach().cpu(),
            "translation": translation[-1].detach().cpu(),
            "features": features[0, -1].detach().cpu(),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
