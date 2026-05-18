# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion script for OpenArm Dataset to robot_eval_logger format.

The on-disk layout produced here is the one documented in
robot_eval_logger's ``DATA_FORMAT.md``: each run is a directory
``<output>/<eval_id>/`` containing one ``metadata.json`` plus one
``traj_{i}.pkl`` per episode (an lz4-compressed pickle of an object
with the documented attributes). The files are written directly from
the spec, so this package has no runtime dependency on the
``robot_eval_logger`` package itself.
"""

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import lz4.frame
import numpy as np

from .dataset import Dataset

# Preferred camera name to map to the eval-logger ``image_primary`` key
# (the FrameVisualizer default). Checked in order.
_PRIMARY_CAMERA_PREFERENCE = ("head", "ceiling", "base", "front")

# End-effector pose is not stored in the OpenArm dataset. Emit a zero vector
# of this length as a placeholder so downstream consumers still get a 1-D array.
_EE_POSE_DIM = 7

# Allowed enum values per robot_eval_logger/DATA_FORMAT.md.
_CONTROL_MODES = ("joint_velocity", "joint_position", "end_effector")


@dataclass
class _TrajData:
    """Per-episode payload pickled into ``traj_{i}.pkl``.

    robot_eval_logger's loader checks attribute names and types only
    (DATA_FORMAT.md, "The object's class name does not matter for
    loading"), so this class is intentionally local.
    """

    language_command: str
    success: bool
    episode_length: int
    duration_seconds: float
    collection_time: str
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    end_effector_pose: np.ndarray
    gripper: np.ndarray


def _write_metadata_json(
    path: Path,
    *,
    eval_id: int,
    robot_name: str,
    robot_type: str,
    control_mode: str,
    action_frequency_hz: float,
    time_iso: str,
    location: str | None,
) -> None:
    payload = {
        "eval_id": eval_id,
        "robot_name": robot_name,
        "robot_type": robot_type,
        "control_mode": control_mode,
        "action_frequency_hz": action_frequency_hz,
        "time": time_iso,
        "location": location,
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def _write_traj(path: Path, traj: _TrajData) -> None:
    raw = pickle.dumps(traj, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path, "wb") as f:
        f.write(lz4.frame.compress(raw))


def _make_eval_id(time_iso: str, robot_type: str) -> int:
    """Return a positive int identifying this run.

    Mirrors robot_eval_logger's ``EvalID.create`` semantics (Python
    ``hash`` of a time + robot_type string, made positive); the value
    is non-deterministic across Python processes because string
    hashing is salted via PYTHONHASHSEED.
    """
    return abs(hash(f"{time_iso}{robot_type}"))


def _earliest_collection_timestamp(dataset: Dataset):
    earliest = None
    for i in range(dataset.num_episodes):
        for cam in dataset.load_cameras(i).values():
            if cam.num_frames == 0:
                continue
            t = cam.get_frame(0).timestamp
            if earliest is None or t < earliest:
                earliest = t
    return earliest


def _measure_action_frequency_hz(dataset: Dataset):
    if dataset.num_episodes == 0:
        return None
    rates = []
    for i in range(dataset.num_episodes):
        for df in dataset.load_action(i, use_unixtime=True).values():
            if len(df.index) < 2:
                continue
            mean_dt = float(np.mean(np.diff(df.index.to_numpy())))
            if mean_dt > 0:
                rates.append(1.0 / mean_dt)
    if not rates:
        return None
    rates = np.asarray(rates)
    mean_hz = float(rates.mean())
    if len(rates) > 1 and mean_hz > 0:
        rel_std = float(rates.std() / mean_hz)
        if rel_std > 0.05:
            print(
                f"[openarm_dataset] warning: action rate varies across parquets "
                f"(mean={mean_hz:.2f} Hz, rel_stdev={rel_std:.1%}); "
                f"single action_frequency_hz value may misrepresent the run."
            )
    return mean_hz


def _joint_index_map(dataset: Dataset, arm: str):
    mapping = {}
    for name, embodiment in dataset.meta.equipment.embodiments.items():
        if embodiment.components:
            if arm not in embodiment.components:
                raise ValueError(
                    f"Embodiment {name!r} has components {embodiment.components}; "
                    f"requested arm {arm!r} not found"
                )
            keys = [f"{name}/{arm}/qpos"]
        else:
            keys = [f"{name}/qpos"]
        pos_idxs = [j for j, jn in enumerate(embodiment.joints) if jn != "gripper"]
        grip_idxs = [j for j, jn in enumerate(embodiment.joints) if jn == "gripper"]
        for key in keys:
            mapping[key] = (pos_idxs, grip_idxs)
    return mapping


def _camera_belongs_to_other_arm(camera_name: str, arm: str) -> bool:
    """True if the camera is the opposite arm's wrist camera."""
    opposite = "left" if arm == "right" else "right"
    return opposite in camera_name.split("_")


def _resolve_image_key_map(dataset: Dataset, arm: str, user_map):
    if user_map is not None:
        return dict(user_map)
    cameras = [
        c for c in dataset.camera_names
        if not _camera_belongs_to_other_arm(c, arm)
    ]
    primary = None
    for preferred in _PRIMARY_CAMERA_PREFERENCE:
        if preferred in cameras:
            primary = preferred
            break
    if primary is None and cameras:
        primary = cameras[0]
    return {
        name: ("image_primary" if name == primary else f"image_{name}")
        for name in cameras
    }


def _derive_robot_name(dataset: Dataset) -> str:
    for embodiment in dataset.meta.equipment.embodiments.values():
        if embodiment.id == "OpenArm":
            return f"OpenArm_{embodiment.version}"
    return "OpenArm"


def _episode_to_traj_data(
    dataset: Dataset,
    episode_index: int,
    fps: int,
    joint_index_map: dict,
    image_key_map: dict,
    tasks: list,
):
    samples = dataset.sample(hz=fps, episode_index=episode_index)
    if not samples:
        return None

    timestamps = np.asarray([s.timestamp for s in samples], dtype=np.float64)
    obs_image_lists: Dict[str, list] = {}
    joint_position_list = []
    gripper_list = []
    action_list = []
    obs_keys_sorted = sorted(joint_index_map)

    for s in samples:
        for cam_name, frame in s.cameras.items():
            if cam_name not in image_key_map:
                continue
            mapped = image_key_map[cam_name]
            obs_image_lists.setdefault(mapped, []).append(np.asarray(frame.load()))

        pos_parts = []
        grip_parts = []
        action_parts = []
        for key in obs_keys_sorted:
            pos_idxs, grip_idxs = joint_index_map[key]
            obs_vec = np.asarray(s.obs[key], dtype=np.float32)
            if pos_idxs:
                pos_parts.append(obs_vec[pos_idxs])
            if grip_idxs:
                grip_parts.append(obs_vec[grip_idxs])
            action_parts.append(np.asarray(s.action[key], dtype=np.float32))

        joint_position_list.append(
            np.concatenate(pos_parts) if pos_parts else np.zeros(0, dtype=np.float32)
        )
        gripper_list.append(
            np.concatenate(grip_parts) if grip_parts else np.zeros(1, dtype=np.float32)
        )
        action_list.append(np.concatenate(action_parts))

    joint_position_arr = np.stack(joint_position_list).astype(np.float32)
    if len(joint_position_arr) > 1:
        joint_velocity_arr = np.gradient(
            joint_position_arr, timestamps, axis=0
        ).astype(np.float32)
    else:
        joint_velocity_arr = np.zeros_like(joint_position_arr, dtype=np.float32)

    action_arr = np.stack([a.astype(np.float32) for a in action_list])
    gripper_arr = np.stack([g.astype(np.float32) for g in gripper_list])
    ee_pose_arr = np.zeros((len(samples), _EE_POSE_DIM), dtype=np.float32)
    obs_arr: Dict[str, np.ndarray] = {
        k: np.stack(v).astype(np.uint8) for k, v in obs_image_lists.items()
    }

    episode_meta = dataset.meta.episodes[episode_index]
    task_index = episode_meta.get("task_index", 0)
    if tasks and 0 <= task_index < len(tasks):
        language_command = tasks[task_index].get("prompt", "")
    else:
        language_command = ""

    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    return _TrajData(
        language_command=language_command,
        success=bool(episode_meta.get("success", False)),
        episode_length=len(samples),
        duration_seconds=duration,
        collection_time=datetime.fromtimestamp(float(timestamps[0])).isoformat(),
        obs=obs_arr,
        action=action_arr,
        joint_position=joint_position_arr,
        joint_velocity=joint_velocity_arr,
        end_effector_pose=ee_pose_arr,
        gripper=gripper_arr,
    )


def to_eval_logger(
    dataset: Dataset,
    output_dir: str | Path,
    arm: str,
    control_mode: str = "joint_position",
    fps: int = 30,
    image_key_map: dict | None = None,
) -> None:
    """Convert the given dataset to robot_eval_logger format and save under output_dir/<eval_id>/.

    robot_eval_logger is single-arm only, so ``arm`` ("left" or "right") selects which
    half of the bimanual data to export; the other arm is dropped entirely.

    Metadata (robot_name, location) is derived from the dataset's metadata.yaml;
    robot_type is fixed to "openarm". action_frequency_hz is measured from the data.
    """
    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    if control_mode not in _CONTROL_MODES:
        raise ValueError(
            f"control_mode must be one of {_CONTROL_MODES}, got {control_mode!r}"
        )

    kept_cameras = [
        c for c in dataset.camera_names
        if not _camera_belongs_to_other_arm(c, arm)
    ]
    dropped_cameras = [
        c for c in dataset.camera_names
        if _camera_belongs_to_other_arm(c, arm)
    ]
    banner = "!" * 78
    print(banner)
    print(f"!! SINGLE-ARM EXPORT: keeping {arm.upper()} arm only, DROPPING the other arm.")
    print("!! robot_eval_logger is single-arm only; bimanual data cannot be represented.")
    print(f"!! Cameras kept:    {kept_cameras}")
    print(f"!! Cameras dropped: {dropped_cameras}")
    print(banner)

    output_dir = Path(output_dir)
    robot_type = "openarm"

    # Collection time is the earliest frame on disk, not datetime.now().
    earliest_unix = _earliest_collection_timestamp(dataset)
    run_dt = (
        datetime.fromtimestamp(earliest_unix)
        if earliest_unix is not None
        else datetime.now()
    )
    time_iso = run_dt.isoformat()

    action_frequency_hz = _measure_action_frequency_hz(dataset) or fps

    eval_id = _make_eval_id(time_iso, robot_type)
    run_dir = output_dir / str(eval_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Eval data saving to {run_dir}")

    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        raise ValueError(f"metadata already exists at {metadata_path}")
    _write_metadata_json(
        metadata_path,
        eval_id=eval_id,
        robot_name=_derive_robot_name(dataset),
        robot_type=robot_type,
        control_mode=control_mode,
        action_frequency_hz=float(action_frequency_hz),
        time_iso=time_iso,
        location=dataset.meta.location,
    )

    joint_index_map = _joint_index_map(dataset, arm)
    resolved_image_map = _resolve_image_key_map(dataset, arm, image_key_map)
    tasks = list(dataset.meta.tasks or [])

    for i in range(dataset.num_episodes):
        traj = _episode_to_traj_data(
            dataset,
            i,
            fps,
            joint_index_map,
            resolved_image_map,
            tasks,
        )
        if traj is None:
            continue
        _write_traj(run_dir / f"traj_{i}.pkl", traj)
