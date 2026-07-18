#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import time
from pathlib import Path

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Path to a precomputed leader-arm calibration JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip="localhost", id="didi")
    teleop_arm_config = SO100LeaderConfig(port="/dev/ttyACM1", id="di")  # tty.usbmodem5AB01813381
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    should_calibrate = True
    if args.calibration_file is not None:
        if not args.calibration_file.is_file():
            raise FileNotFoundError(f"Calibration file not found: {args.calibration_file}")
        leader_arm._load_calibration(args.calibration_file)
        leader_arm.bus.calibration = leader_arm.calibration
        should_calibrate = False

    # Connect to the robot and teleoperator
    # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`
    robot.connect()
    leader_arm.connect(calibrate=should_calibrate)
    keyboard.connect()

    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Arm
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        # Keyboard
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Send action to robot
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
