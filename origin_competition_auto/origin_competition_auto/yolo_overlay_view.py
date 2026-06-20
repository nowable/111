#!/usr/bin/env python3
"""Capture one camera frame and render current YOLO detections on top."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    try:
        from ai_msgs.msg import PerceptionTargets
    except ImportError:
        PerceptionTargets = None
except ImportError:
    rclpy = None
    CompressedImage = None
    PerceptionTargets = None

    class Node:  # type: ignore[no-redef]
        pass

try:
    from origin_competition_auto.vision_detector import (
        DetectorConfig,
        VisionDetector,
        VisionDecision,
        parse_ai_targets_msg,
        load_detector_config,
    )
except ImportError:
    from vision_detector import (  # type: ignore[no-redef]
        DetectorConfig,
        VisionDetector,
        VisionDecision,
        parse_ai_targets_msg,
        load_detector_config,
    )


class OverlayNode(Node):
    def __init__(
        self,
        image_topic: str,
        yolo_topic: str,
        detector_config: DetectorConfig,
        yolo_min_confidence: float,
    ) -> None:
        super().__init__('yolo_overlay_view')
        self.vision = VisionDetector(detector_config)
        self.yolo_min_confidence = yolo_min_confidence
        self.latest_image: Optional[np.ndarray] = None
        self.latest_detections = []
        self.latest_detection_time = 0.0
        self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        if PerceptionTargets is not None:
            self.create_subscription(
                PerceptionTargets,
                yolo_topic,
                self._yolo_callback,
                qos_profile_sensor_data,
            )

    def _image_callback(self, msg: CompressedImage) -> None:
        array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is not None:
            self.latest_image = image

    def _yolo_callback(self, msg: object) -> None:
        self.latest_detections = parse_ai_targets_msg(
            msg,
            min_confidence=self.yolo_min_confidence,
        )
        self.latest_detection_time = time.monotonic()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Save one camera frame with current YOLO detections overlaid.'
    )
    parser.add_argument('--image-topic', default='/image')
    parser.add_argument('--yolo-topic', default='/hobot_dnn_detection')
    parser.add_argument('--config', help='Optional mission config JSON path.')
    parser.add_argument('--output', required=True, help='Output JPG/PNG path.')
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--yolo-min-confidence', type=float, default=0.45)
    parser.add_argument('--require-yolo', action='store_true',
                        help='Fail if no YOLO detections were received.')
    return parser


def main(argv=None) -> int:
    if rclpy is None:
        print('rclpy is not available', flush=True)
        return 2

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    detector_config = load_detector_config(args.config)
    rclpy.init(args=None)
    node = OverlayNode(
        image_topic=args.image_topic,
        yolo_topic=args.yolo_topic,
        detector_config=detector_config,
        yolo_min_confidence=args.yolo_min_confidence,
    )
    try:
        deadline = time.monotonic() + max(0.5, args.timeout)
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_image is not None:
                if not args.require_yolo or node.latest_detection_time > 0.0:
                    break

        if node.latest_image is None:
            print('OVERLAY_CAPTURE_FAIL: no image received', flush=True)
            return 1
        if args.require_yolo and node.latest_detection_time <= 0.0:
            print('OVERLAY_CAPTURE_FAIL: no yolo detections received', flush=True)
            return 1

        decision = VisionDecision(detections=node.latest_detections)
        overlay = node.vision.draw_debug(node.latest_image, decision)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)
        print(f'SAVED_OVERLAY_IMAGE: {output_path}', flush=True)
        print(f'YOLO_DETECTIONS: {len(node.latest_detections)}', flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
