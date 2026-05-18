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

"""Conformance test for the eval_logger converter output.

Re-implements the validation logic from
``robot_eval_logger/tests/test_data_format_validator.py`` inline (that script
ships in the upstream repo's ``tests/`` dir, which is not included in the
installed package). The checks here mirror what is documented in
``robot_eval_logger/DATA_FORMAT.md``.
"""

import json
import pickle
import re
from pathlib import Path

import lz4.frame
import numpy as np
import pytest

from openarm_dataset import Dataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
ARM = "right"
FPS = 30

_LZ4_MAGIC = b"\x04\x22\x4d\x18"
_VALID_ROBOT_TYPES = {"franka", "widowx", "openarm"}
_VALID_CONTROL_MODES = {"joint_velocity", "joint_position", "end_effector"}
_REQUIRED_METADATA_FIELDS = (
    "eval_id",
    "robot_name",
    "robot_type",
    "control_mode",
    "action_frequency_hz",
    "time",
)
_NUMERIC_STEP_FIELDS = (
    "action",
    "joint_position",
    "joint_velocity",
    "end_effector_pose",
    "gripper",
)
_CONTROL_MODE_REQUIRED_STATE = {
    "joint_velocity": "joint_velocity",
    "joint_position": "joint_position",
    "end_effector": "end_effector_pose",
}
_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


@pytest.fixture
def converted_run_dir(tmp_path):
    """Convert the 0.3.0 fixture to eval_logger format and return the run dir."""
    Dataset(DATASET_0_3_0_PATH).write(
        tmp_path,
        format="eval_logger",
        arm=ARM,
        fps=FPS,
    )
    children = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(children) == 1, f"expected exactly one <eval_id> dir, found {children}"
    return children[0]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        header = f.read(4)
        f.seek(0)
        if header == _LZ4_MAGIC:
            return pickle.loads(lz4.frame.decompress(f.read()))
        return pickle.load(f)


def _validate_metadata(metadata: dict) -> list[str]:
    errs = []
    for field in _REQUIRED_METADATA_FIELDS:
        if field not in metadata:
            errs.append(f"missing required field '{field}'")

    if "eval_id" in metadata and not isinstance(metadata["eval_id"], int):
        errs.append(f"eval_id must be int, got {type(metadata['eval_id']).__name__}")
    if "robot_type" in metadata and metadata["robot_type"] not in _VALID_ROBOT_TYPES:
        errs.append(f"robot_type {metadata['robot_type']!r} not in {_VALID_ROBOT_TYPES}")
    if "control_mode" in metadata and metadata["control_mode"] not in _VALID_CONTROL_MODES:
        errs.append(
            f"control_mode {metadata['control_mode']!r} not in {_VALID_CONTROL_MODES}"
        )
    if "action_frequency_hz" in metadata:
        v = metadata["action_frequency_hz"]
        if not isinstance(v, (int, float)) or v <= 0:
            errs.append(f"action_frequency_hz must be a positive number, got {v!r}")
    if "time" in metadata:
        v = metadata["time"]
        if not isinstance(v, str) or not _ISO8601_RE.match(v):
            errs.append(f"time must be ISO 8601 string, got {v!r}")
    return errs


def _validate_trajectory(traj, control_mode: str) -> list[str]:
    errs = []
    _MISSING = object()

    lang = getattr(traj, "language_command", _MISSING)
    if lang is _MISSING:
        errs.append("missing 'language_command'")
    elif not isinstance(lang, str) or not lang.strip():
        errs.append(f"'language_command' must be non-empty string, got {lang!r}")

    success = getattr(traj, "success", _MISSING)
    if success is _MISSING:
        errs.append("missing 'success'")
    elif not isinstance(success, (bool, np.bool_)):
        errs.append(f"'success' must be bool, got {type(success).__name__}")

    for required in ("obs", "action", "gripper"):
        if getattr(traj, required, None) is None:
            errs.append(f"missing required step-level field '{required}'")

    required_state = _CONTROL_MODE_REQUIRED_STATE.get(control_mode)
    if required_state and getattr(traj, required_state, None) is None:
        errs.append(
            f"control_mode is {control_mode!r} but required state "
            f"field {required_state!r} is missing"
        )

    # obs must be dict of camera_name -> (T, H, W, 3) uint8 ndarray
    obs = getattr(traj, "obs", None)
    if isinstance(obs, dict):
        for cam, arr in obs.items():
            if not isinstance(arr, np.ndarray):
                errs.append(
                    f"obs[{cam!r}] must be ndarray (T,H,W,3) uint8, "
                    f"got {type(arr).__name__}"
                )
                continue
            if arr.ndim != 4 or arr.shape[3] != 3:
                errs.append(f"obs[{cam!r}] must have shape (T,H,W,3), got {arr.shape}")
            if arr.dtype != np.uint8:
                errs.append(f"obs[{cam!r}] dtype must be uint8, got {arr.dtype}")
    elif obs is not None:
        errs.append(f"'obs' must be dict, got {type(obs).__name__}")

    # Numeric step fields must be stacked (T, D) float32 ndarrays
    for field in _NUMERIC_STEP_FIELDS:
        val = getattr(traj, field, None)
        if val is None:
            continue
        if isinstance(val, list):
            errs.append(f"'{field}' must be stacked ndarray, got list")
            continue
        if not isinstance(val, np.ndarray):
            errs.append(f"'{field}' must be ndarray, got {type(val).__name__}")
            continue
        if val.ndim != 2:
            errs.append(f"'{field}' must be 2-D (T,D), got shape {val.shape}")
        if val.dtype != np.float32:
            errs.append(f"'{field}' dtype must be float32, got {val.dtype}")
        if field == "gripper" and val.ndim == 2 and val.shape[1] != 1:
            errs.append(f"'gripper' must have 1 column, got {val.shape[1]}")
        if field == "end_effector_pose" and val.ndim == 2 and val.shape[1] != 7:
            errs.append(f"'end_effector_pose' must have 7 columns, got {val.shape[1]}")

    return errs


def test_run_dir_structure(converted_run_dir):
    """The run dir contains metadata.json and at least one traj_<i>.pkl."""
    assert (converted_run_dir / "metadata.json").is_file()
    trajs = sorted(converted_run_dir.glob("traj_*.pkl"))
    assert trajs, f"no traj_*.pkl files in {converted_run_dir}"
    # Indices must be contiguous from 0
    indices = [int(re.search(r"\d+", p.name).group()) for p in trajs]
    assert indices == list(range(len(indices))), f"non-contiguous traj indices: {indices}"


def test_metadata_matches_spec(converted_run_dir):
    """metadata.json conforms to robot_eval_logger DATA_FORMAT.md."""
    with open(converted_run_dir / "metadata.json") as f:
        meta = json.load(f)
    errs = _validate_metadata(meta)
    assert not errs, "metadata.json violations:\n  - " + "\n  - ".join(errs)


def test_trajectories_match_spec(converted_run_dir):
    """Each traj_<i>.pkl conforms to DATA_FORMAT.md."""
    with open(converted_run_dir / "metadata.json") as f:
        control_mode = json.load(f)["control_mode"]

    all_errs = []
    for traj_path in sorted(converted_run_dir.glob("traj_*.pkl")):
        traj = _load_pickle(traj_path)
        for e in _validate_trajectory(traj, control_mode):
            all_errs.append(f"{traj_path.name}: {e}")

    assert not all_errs, "trajectory violations:\n  - " + "\n  - ".join(all_errs)
