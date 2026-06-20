#!/usr/bin/env python3
"""Capture camera frames for OpenCV tuning and YOLO dataset preparation."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CompressedImage
except ImportError:
    rclpy = None
    CompressedImage = None
    qos_profile_sensor_data = None

    class Node:  # type: ignore[no-redef]
        pass

    def remove_ros_args(args: List[str]) -> List[str]:
        return args

try:
    from origin_competition_auto.vision_detector import VisionDetector
except ImportError:
    from vision_detector import VisionDetector  # type: ignore[no-redef]


DEFAULT_CLASSES = ('black_obstacle', 'qr_board', 'image_card', 'base', 'marker')


class CameraFrameBuffer(Node):
    def __init__(self, topic: str) -> None:
        super().__init__('dataset_capture')
        self.latest_image: Optional[np.ndarray] = None
        self.latest_time: Optional[float] = None
        self.frame_count = 0
        self.create_subscription(
            CompressedImage,
            topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

    def _image_callback(self, msg: CompressedImage) -> None:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn('failed to decode camera frame')
            return
        self.latest_image = image
        self.latest_time = time.monotonic()
        self.frame_count += 1


def parse_classes(raw: str) -> List[str]:
    classes = [item.strip() for item in raw.split(',') if item.strip()]
    if not classes:
        raise ValueError('classes must not be empty')
    return classes


def dataset_paths(output_dir: Path, split: str, stem: str) -> Dict[str, Path]:
    return {
        'image': output_dir / 'images' / split / f'{stem}.jpg',
        'label': output_dir / 'labels' / split / f'{stem}.txt',
        'overlay': output_dir / 'debug' / split / f'{stem}_overlay.jpg',
        'metadata': output_dir / 'metadata.jsonl',
        'classes': output_dir / 'classes.txt',
        'readme': output_dir / 'README.md',
    }


def ensure_dataset(output_dir: Path, split: str, classes: Sequence[str]) -> None:
    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    (output_dir / 'debug' / split).mkdir(parents=True, exist_ok=True)
    classes_path = output_dir / 'classes.txt'
    if not classes_path.exists():
        classes_path.write_text('\n'.join(classes) + '\n', encoding='utf-8')
    readme_path = output_dir / 'README.md'
    if not readme_path.exists():
        readme_path.write_text(
            '# OriginCar Competition Dataset\n\n'
            'Images are saved for OpenCV threshold tuning and YOLO annotation.\n'
            'YOLO label files are created only when an auto-label mode is enabled; '
            'otherwise images remain unlabeled and should be annotated manually.\n',
            encoding='utf-8',
        )


def make_stem(prefix: str, index: int) -> str:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f'{prefix}_{stamp}_{index:04d}'


def bbox_to_yolo_line(
    class_id: int,
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int, int],
) -> str:
    height, width = image_shape[:2]
    x, y, w, h = bbox
    cx = (x + w / 2.0) / max(1, width)
    cy = (y + h / 2.0) / max(1, height)
    nw = w / max(1, width)
    nh = h / max(1, height)
    return f'{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}'


def auto_label_image(
    image: np.ndarray,
    mode: str,
    classes: Sequence[str],
) -> Tuple[List[str], Dict[str, object], Optional[np.ndarray]]:
    if mode == 'none':
        return [], {'mode': mode, 'detections': []}, None

    detector = VisionDetector()
    decision = detector.analyze(image)
    selected = []
    if mode in ('opencv-obstacle', 'opencv-all') and decision.obstacle is not None:
        selected.append(('black_obstacle', decision.obstacle))
    if mode in ('opencv-target', 'opencv-all') and decision.target_card is not None:
        selected.append(('image_card', decision.target_card))

    lines = []
    for class_name, detection in selected:
        if class_name not in classes:
            continue
        lines.append(
            bbox_to_yolo_line(classes.index(class_name), detection.bbox, image.shape)
        )

    metadata = decision.to_dict()
    metadata['mode'] = mode
    overlay = detector.draw_debug(image, decision)
    return lines, metadata, overlay


def save_sample(
    image: np.ndarray,
    output_dir: Path,
    split: str,
    stem: str,
    classes: Sequence[str],
    scene_label: str,
    source_note: str,
    auto_label_mode: str,
    save_overlay: bool,
) -> Dict[str, object]:
    paths = dataset_paths(output_dir, split, stem)
    paths['image'].parent.mkdir(parents=True, exist_ok=True)
    paths['label'].parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(paths['image']), image)
    if not ok:
        raise RuntimeError(f'failed to save image: {paths["image"]}')

    label_lines, auto_metadata, overlay = auto_label_image(
        image, auto_label_mode, classes
    )
    if label_lines:
        paths['label'].write_text('\n'.join(label_lines) + '\n', encoding='utf-8')
    elif auto_label_mode != 'none':
        paths['label'].write_text('', encoding='utf-8')

    overlay_path = ''
    if save_overlay and overlay is not None:
        ok = cv2.imwrite(str(paths['overlay']), overlay)
        if ok:
            overlay_path = str(paths['overlay'])

    height, width = image.shape[:2]
    record: Dict[str, object] = {
        'time': datetime.now().isoformat(timespec='milliseconds'),
        'image': str(paths['image']),
        'label_file': str(paths['label']) if paths['label'].exists() else '',
        'overlay': overlay_path,
        'split': split,
        'scene_label': scene_label,
        'source_note': source_note,
        'width': width,
        'height': height,
        'auto_label_mode': auto_label_mode,
        'auto_label_count': len(label_lines),
        'auto_label': auto_metadata,
    }
    with paths['metadata'].open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    return record


def capture_from_input_file(args: argparse.Namespace, classes: Sequence[str]) -> int:
    image = cv2.imread(args.input_file)
    if image is None:
        print(f'INPUT_IMAGE_ERROR: {args.input_file}', flush=True)
        return 2
    ensure_dataset(Path(args.output_dir), args.split, classes)
    stem = make_stem(args.prefix, 1)
    record = save_sample(
        image=image,
        output_dir=Path(args.output_dir),
        split=args.split,
        stem=stem,
        classes=classes,
        scene_label=args.label,
        source_note=args.source_note or args.input_file,
        auto_label_mode=args.auto_label_mode,
        save_overlay=args.save_overlay,
    )
    print(f'SAVED_DATASET_IMAGE: {record["image"]}', flush=True)
    if record.get('label_file'):
        print(f'SAVED_DATASET_LABEL: {record["label_file"]}', flush=True)
    if record.get('overlay'):
        print(f'SAVED_DATASET_OVERLAY: {record["overlay"]}', flush=True)
    return 0


def capture_from_ros(args: argparse.Namespace, classes: Sequence[str]) -> int:
    if rclpy is None:
        print('ROS_ERROR: rclpy is not available', flush=True)
        return 2
    output_dir = Path(args.output_dir)
    ensure_dataset(output_dir, args.split, classes)

    rclpy.init(args=None)
    node = CameraFrameBuffer(args.topic)
    saved = 0
    last_saved_frame = -1
    next_save_time = time.monotonic() + max(0.0, args.warmup)
    deadline = time.monotonic() + max(0.1, args.timeout)
    try:
        while rclpy.ok() and saved < args.count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.latest_image is None:
                continue
            now = time.monotonic()
            if now < next_save_time:
                continue
            if node.frame_count == last_saved_frame:
                continue
            saved += 1
            stem = make_stem(args.prefix, saved)
            record = save_sample(
                image=node.latest_image,
                output_dir=output_dir,
                split=args.split,
                stem=stem,
                classes=classes,
                scene_label=args.label,
                source_note=args.source_note,
                auto_label_mode=args.auto_label_mode,
                save_overlay=args.save_overlay,
            )
            last_saved_frame = node.frame_count
            print(
                f'SAVED_DATASET_IMAGE: {record["image"]} '
                f'auto_labels={record["auto_label_count"]}',
                flush=True,
            )
            next_save_time = now + max(0.0, args.interval)
        if saved < args.count:
            print(f'CAPTURE_INCOMPLETE saved={saved} requested={args.count}', flush=True)
            return 1
        print(f'CAPTURE_DONE saved={saved} output_dir={output_dir}', flush=True)
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Capture OriginCar camera frames for tuning and YOLO annotation.'
    )
    parser.add_argument('--topic', default='/image',
                        help='CompressedImage topic to capture from.')
    parser.add_argument('--output-dir',
                        default='/root/dev_ws/datasets/origin_competition',
                        help='Dataset root directory.')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test', 'unlabeled'])
    parser.add_argument('--label', default='unlabeled',
                        help='Scene tag stored in metadata; not a YOLO box label.')
    parser.add_argument('--classes', default=','.join(DEFAULT_CLASSES),
                        help='Comma-separated YOLO class names.')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of frames to save from ROS topic.')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Seconds between saved frames.')
    parser.add_argument('--warmup', type=float, default=0.5,
                        help='Seconds to wait before first ROS capture.')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Maximum ROS capture time.')
    parser.add_argument('--prefix', default='capture')
    parser.add_argument('--source-note', default='',
                        help='Free-form note stored in metadata.')
    parser.add_argument('--input-file',
                        help='Save one local image instead of subscribing to ROS.')
    parser.add_argument('--auto-label-mode', default='none',
                        choices=['none', 'opencv-obstacle', 'opencv-target', 'opencv-all'],
                        help='Optional rough YOLO labels from OpenCV detections.')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save OpenCV debug overlays when auto-labeling.')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    try:
        classes = parse_classes(args.classes)
        if args.count < 1:
            raise ValueError('count must be >= 1')
        if args.input_file:
            return capture_from_input_file(args, classes)
        return capture_from_ros(args, classes)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
