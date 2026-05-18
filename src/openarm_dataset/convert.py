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

"""Convert OpenArm dataset format."""

import argparse
import openarm_dataset
import pathlib


def main():
    """Convert OpenArm dataset format."""
    parser = argparse.ArgumentParser(description="Convert OpenArm dataset format")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to be converted",
        type=pathlib.Path,
    )
    parser.add_argument(
        "output",
        help="Path of converted OpenArm dataset",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--format",
        help="Format of the output dataset (default: openarm)",
        default="openarm",
        choices=["openarm", "lerobot_v2.1", "eval_logger"],
    )
    parser.add_argument(
        "--fps",
        help="Frames per second for the output dataset (default: 30) if the output format is lerobot_v2.1 or eval_logger",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--smoothing-cutoff",
        help="Cutoff frequency for smoothing (default: 1.0) if the output format is lerobot_v2.1",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--train-split",
        help="Split ratio for training dataset (default: 0.8) if the output format is lerobot_v2.1",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--success-only",
        help="Include only successful episodes in the output dataset (default: False) if the output format is lerobot_v2.1",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--control-mode",
        help="Robot control mode if the output format is eval_logger (default: joint_position)",
        choices=("joint_velocity", "joint_position", "end_effector"),
        default="joint_position",
    )
    parser.add_argument(
        "--arm",
        help="Which arm to export if the output format is eval_logger (required; robot_eval_logger is single-arm only, so the other arm is dropped)",
        choices=("left", "right"),
        default=None,
    )

    args = parser.parse_args()
    write_kwargs = {"format": args.format}
    if args.format == "lerobot_v2.1":
        write_kwargs["fps"] = args.fps
        write_kwargs["smoothing_cutoff"] = args.smoothing_cutoff
        write_kwargs["train_split"] = args.train_split
        write_kwargs["success_only"] = args.success_only
    elif args.format == "eval_logger":
        if args.arm is None:
            parser.error("--format eval_logger requires --arm {left,right}")
        write_kwargs["fps"] = args.fps
        write_kwargs["control_mode"] = args.control_mode
        write_kwargs["arm"] = args.arm

    old_dataset = openarm_dataset.Dataset(args.input)
    old_dataset.write(args.output, **write_kwargs)


if __name__ == "__main__":
    main()
