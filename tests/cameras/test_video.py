#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import cv2

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a live stream from an OpenCV camera.")
    parser.add_argument("--path", default="/dev/video5", help="Camera device path or index.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--width", type=int, default=640, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera frame height.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. Use 0 to stream until keypress.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Read frames without opening a GUI window (useful for headless environments).",
    )
    return parser.parse_args()


def _parse_path(path: str) -> str | int:
    return int(path) if path.isdigit() else path


def main() -> None:
    args = parse_args()
    index_or_path = _parse_path(args.path)

    config = OpenCVCameraConfig(
        index_or_path=index_or_path,
        fps=args.fps,
        width=args.width,
        height=args.height,
        color_mode=ColorMode.BGR,
    )
    camera = OpenCVCamera(config)

    frame_count = 0
    display_enabled = not args.no_display
    display_warning_printed = False
    window_name = f"LeRobot Live Stream ({args.path})"
    try:
        camera.connect()
        while True:
            frame = camera.read()
            frame_count += 1

            if display_enabled:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error:
                    display_enabled = False
                    if not display_warning_printed:
                        print("OpenCV HighGUI is unavailable; continuing without display. Press Ctrl+C to stop.")
                        display_warning_printed = True

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        if camera.is_connected:
            camera.disconnect()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
