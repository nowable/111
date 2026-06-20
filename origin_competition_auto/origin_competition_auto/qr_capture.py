#!/usr/bin/env python3
"""Capture one camera frame and try to decode a QR code."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from pyzbar import pyzbar as _pyzbar
except ImportError:
    _pyzbar = None

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage


DEFAULT_OUTPUT = '/root/dev_ws/debug/qr_capture/latest.jpg'


class QrCaptureNode(Node):
    def __init__(self, topic: str, output: str, save: bool) -> None:
        super().__init__('qr_capture')
        self.output = Path(output)
        self.save = save
        cv2.setNumThreads(1)
        self.detector = cv2.QRCodeDetector()
        self.last_result: Optional[str] = None
        self.frame_received = False
        self.finished = False
        self.subscription = self.create_subscription(
            CompressedImage,
            topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f'subscribed to {topic}')

    def _image_callback(self, msg: CompressedImage) -> None:
        self.frame_received = True
        image = self._decode_image(msg)
        if image is None:
            self.get_logger().error('failed to decode CompressedImage')
            self.finished = True
            return

        if self.save:
            self._save_image(image)

        decoded, points, method = self._detect_qr(image)
        if decoded:
            self.last_result = decoded
            self.get_logger().info(f'QR_DETECTED: {decoded} ({method})')
            print(f'QR_DETECTED: {decoded}', flush=True)
            print(f'QR_METHOD: {method}', flush=True)
        else:
            self.get_logger().warn('QR_NOT_FOUND: frame saved for inspection')
            print('QR_NOT_FOUND', flush=True)
        if points is not None:
            print(f'QR_POINTS: {points.reshape(-1, 2).tolist()}', flush=True)
        self.finished = True

    def _decode_image(self, msg: CompressedImage) -> Optional[np.ndarray]:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def _save_image(self, image: np.ndarray) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(self.output), image)
        if ok:
            height, width = image.shape[:2]
            self.get_logger().info(
                f'saved frame to {self.output} ({width}x{height})'
            )
            print(f'SAVED_IMAGE: {self.output}', flush=True)
        else:
            self.get_logger().error(f'failed to save frame to {self.output}')

    def _detect_qr(self, image: np.ndarray) -> Tuple[str, Optional[np.ndarray], str]:
        gray_first = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if _pyzbar is not None:
            try:
                symbols = _pyzbar.decode(gray_first)
            except Exception:
                symbols = []
            for sym in symbols:
                if not sym.data:
                    continue
                text = sym.data.decode('utf-8', errors='replace')
                points = np.array(
                    [[p.x, p.y] for p in sym.polygon], dtype=np.float32
                )
                return text, points, 'pyzbar'
        candidates = [('bgr', image)]
        gray = gray_first
        candidates.append(('gray', gray))
        for scale in (2, 3):
            resized = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            candidates.append((f'gray_x{scale}', resized))

        best_points = None
        for name, candidate in candidates:
            decoded, points, _ = self.detector.detectAndDecode(candidate)
            if decoded:
                return decoded, self._scale_points_back(points, name), name
            if points is not None and best_points is None:
                best_points = self._scale_points_back(points, name)

        # OpenCV often detects a small QR region but cannot decode it until the
        # region is cropped and enlarged. This is common with phone-screen QR.
        if best_points is not None:
            for warp_name, warped in self._qr_warps_from_points(gray, best_points):
                decoded, points, _ = self.detector.detectAndDecode(warped)
                if decoded:
                    return decoded, best_points, warp_name

            for crop_name, crop in self._qr_crops_from_points(gray, best_points):
                for scale in (6,):
                    enlarged = cv2.resize(
                        crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                    )
                    decoded, points, _ = self.detector.detectAndDecode(enlarged)
                    method = f'{crop_name}_gray_x{scale}'
                    if decoded:
                        return decoded, best_points, method

                    _, binary = cv2.threshold(
                        enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    decoded, points, _ = self.detector.detectAndDecode(binary)
                    method = f'{crop_name}_otsu_x{scale}'
                    if decoded:
                        return decoded, best_points, method

        return '', best_points, 'not_found'

    def _scale_points_back(
        self, points: Optional[np.ndarray], method: str
    ) -> Optional[np.ndarray]:
        if points is None:
            return None
        if '_x' not in method:
            return points
        try:
            scale = float(method.rsplit('_x', 1)[1])
        except ValueError:
            return points
        return points / scale

    def _qr_crops_from_points(
        self, gray: np.ndarray, points: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        pts = points.reshape(-1, 2)
        min_x, min_y = np.floor(pts.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(pts.max(axis=0)).astype(int)
        width = max_x - min_x
        height = max_y - min_y
        crops = []
        for ratio in (0.15, 0.3):
            pad = int(max(width, height) * ratio)
            x1 = max(0, min_x - pad)
            y1 = max(0, min_y - pad)
            x2 = min(gray.shape[1], max_x + pad)
            y2 = min(gray.shape[0], max_y + pad)
            if x2 > x1 and y2 > y1:
                crops.append((f'crop_pad_{ratio:.2f}', gray[y1:y2, x1:x2]))
        return crops

    def _qr_warps_from_points(
        self, gray: np.ndarray, points: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        pts = self._order_points(points.reshape(4, 2).astype(np.float32))
        warps = []
        for size in (600, 800):
            dst = np.array(
                [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(gray, matrix, (size, size))
            border = int(size * 0.15)
            bordered = cv2.copyMakeBorder(
                warped,
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            warps.append((f'perspective_border_{size}', bordered))
        return warps

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Capture a frame from /image and decode a QR code.'
    )
    parser.add_argument('--topic', default='/image',
                        help='CompressedImage topic to subscribe to.')
    parser.add_argument('--input-file',
                        help='Decode an existing image file instead of subscribing.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Path for the saved debug image.')
    parser.add_argument('--timeout', type=float, default=5.0,
                        help='Seconds to wait for one frame.')
    parser.add_argument('--once', action='store_true',
                        help='Capture one frame and exit. This is the default.')
    parser.add_argument('--save', action='store_true',
                        help='Save the captured frame.')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the captured frame.')
    return parser


def run(args: argparse.Namespace) -> int:
    save = True
    if args.no_save:
        save = False
    elif args.save:
        save = True

    if args.input_file:
        image = cv2.imread(args.input_file)
        if image is None:
            print(f'INPUT_IMAGE_ERROR: {args.input_file}', flush=True)
            return 2
        cv2.setNumThreads(1)
        detector_shell = type('DetectorShell', (), {})()
        detector_shell.detector = cv2.QRCodeDetector()
        detector_shell._scale_points_back = QrCaptureNode._scale_points_back.__get__(
            detector_shell
        )
        detector_shell._qr_crops_from_points = (
            QrCaptureNode._qr_crops_from_points.__get__(detector_shell)
        )
        detector_shell._qr_warps_from_points = (
            QrCaptureNode._qr_warps_from_points.__get__(detector_shell)
        )
        detector_shell._order_points = QrCaptureNode._order_points.__get__(
            detector_shell
        )
        detector_shell._detect_qr = QrCaptureNode._detect_qr.__get__(detector_shell)
        decoded, points, method = detector_shell._detect_qr(image)
        if save:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output), image)
            print(f'SAVED_IMAGE: {output}', flush=True)
        if decoded:
            print(f'QR_DETECTED: {decoded}', flush=True)
            print(f'QR_METHOD: {method}', flush=True)
            if points is not None:
                print(f'QR_POINTS: {points.reshape(-1, 2).tolist()}', flush=True)
            return 0
        print('QR_NOT_FOUND', flush=True)
        if points is not None:
            print(f'QR_POINTS: {points.reshape(-1, 2).tolist()}', flush=True)
        return 1

    rclpy.init(args=None)
    node = QrCaptureNode(args.topic, args.output, save)
    try:
        deadline = node.get_clock().now().nanoseconds / 1e9 + args.timeout
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = node.get_clock().now().nanoseconds / 1e9
            if now >= deadline:
                node.get_logger().error(
                    f'timeout waiting for image on {args.topic}'
                )
                print('QR_CAPTURE_TIMEOUT', flush=True)
                return 2
        return 0 if node.last_result else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
