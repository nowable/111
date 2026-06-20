#!/usr/bin/env python3
"""Audit OriginCar YOLO-style datasets and generate training artifacts."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = ('train', 'val', 'test', 'unlabeled')


@dataclass
class LabelBox:
    class_id: int
    x: float
    y: float
    w: float
    h: float
    line_no: int


@dataclass
class ImageRecord:
    split: str
    image: str
    label_file: str = ''
    width: int = 0
    height: int = 0
    labels: List[LabelBox] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    root: str
    classes: List[str]
    split_image_counts: Dict[str, int]
    split_label_counts: Dict[str, int]
    class_counts: Dict[str, int]
    warnings: List[str]
    errors: List[str]
    records: List[ImageRecord]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return {
            'root': self.root,
            'ok': self.ok,
            'classes': self.classes,
            'split_image_counts': self.split_image_counts,
            'split_label_counts': self.split_label_counts,
            'class_counts': self.class_counts,
            'warnings': self.warnings,
            'errors': self.errors,
            'records': [
                {
                    **asdict(record),
                    'labels': [asdict(label) for label in record.labels],
                }
                for record in self.records
            ],
        }


def load_classes(root: Path) -> List[str]:
    classes_path = root / 'classes.txt'
    if not classes_path.exists():
        raise ValueError(f'missing classes.txt: {classes_path}')
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f'classes.txt is empty: {classes_path}')
    return classes


def iter_images(root: Path, split: str) -> Iterable[Path]:
    image_dir = root / 'images' / split
    if not image_dir.exists():
        return []
    return sorted(
        path for path in image_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(root: Path, split: str, image_path: Path) -> Path:
    return root / 'labels' / split / f'{image_path.stem}.txt'


def parse_label_file(
    label_path: Path,
    class_count: int,
) -> Tuple[List[LabelBox], List[str], List[str]]:
    labels: List[LabelBox] = []
    warnings: List[str] = []
    errors: List[str] = []
    if not label_path.exists():
        return labels, warnings, errors

    for index, raw_line in enumerate(
        label_path.read_text(encoding='utf-8').splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'{label_path}:{index}: expected 5 fields, got {len(parts)}')
            continue
        try:
            class_id = int(parts[0])
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(f'{label_path}:{index}: non-numeric YOLO label')
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f'{label_path}:{index}: class_id out of range: {class_id}')
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            errors.append(f'{label_path}:{index}: center out of range: {x:.6f},{y:.6f}')
        if not 0.0 < w <= 1.0 or not 0.0 < h <= 1.0:
            errors.append(f'{label_path}:{index}: size out of range: {w:.6f},{h:.6f}')
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        if x1 < -0.001 or y1 < -0.001 or x2 > 1.001 or y2 > 1.001:
            warnings.append(
                f'{label_path}:{index}: box extends outside image bounds'
            )
        labels.append(LabelBox(class_id, x, y, w, h, index))
    return labels, warnings, errors


def read_image_size(image_path: Path) -> Tuple[int, int, Optional[str]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0, f'cannot read image: {image_path}'
    height, width = image.shape[:2]
    return width, height, None


def audit_dataset(root: Path, require_labels: bool = False) -> DatasetReport:
    classes = load_classes(root)
    records: List[ImageRecord] = []
    warnings: List[str] = []
    errors: List[str] = []
    split_image_counts = {split: 0 for split in SPLITS}
    split_label_counts = {split: 0 for split in SPLITS}
    class_counts = {name: 0 for name in classes}

    for split in SPLITS:
        for image_path in iter_images(root, split):
            split_image_counts[split] += 1
            label_path = label_path_for(root, split, image_path)
            width, height, image_error = read_image_size(image_path)
            record = ImageRecord(
                split=split,
                image=str(image_path),
                label_file=str(label_path) if label_path.exists() else '',
                width=width,
                height=height,
            )
            if image_error:
                record.errors.append(image_error)

            labels, label_warnings, label_errors = parse_label_file(
                label_path,
                len(classes),
            )
            record.labels = labels
            record.warnings.extend(label_warnings)
            record.errors.extend(label_errors)

            if label_path.exists():
                split_label_counts[split] += 1
            elif split != 'unlabeled' or require_labels:
                record.warnings.append(f'missing label file: {label_path}')

            for label in labels:
                if 0 <= label.class_id < len(classes):
                    class_counts[classes[label.class_id]] += 1

            warnings.extend(record.warnings)
            errors.extend(record.errors)
            records.append(record)

    if not any(split_image_counts.values()):
        errors.append(f'no images found under {root / "images"}')
    if split_image_counts['train'] == 0:
        warnings.append('train split has no images')
    if split_image_counts['val'] == 0:
        warnings.append('val split has no images')

    return DatasetReport(
        root=str(root),
        classes=classes,
        split_image_counts=split_image_counts,
        split_label_counts=split_label_counts,
        class_counts=class_counts,
        warnings=warnings,
        errors=errors,
        records=records,
    )


def write_yolo_yaml(report: DatasetReport, output_path: Path) -> None:
    root = Path(report.root)
    train_split = 'images/train'
    val_split = 'images/val'
    if report.split_image_counts.get('train', 0) == 0:
        train_split = 'images/unlabeled'
    if report.split_image_counts.get('val', 0) == 0:
        val_split = train_split

    lines = [
        f'path: {root.as_posix()}',
        f'train: {train_split}',
        f'val: {val_split}',
    ]
    if report.split_image_counts.get('test', 0) > 0:
        lines.append('test: images/test')
    lines.append(f'nc: {len(report.classes)}')
    lines.append('names:')
    for index, name in enumerate(report.classes):
        escaped = name.replace("'", "''")
        lines.append(f"  {index}: '{escaped}'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def draw_label_overlay(
    image_path: Path,
    labels: Sequence[LabelBox],
    classes: Sequence[str],
    output_path: Path,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    height, width = image.shape[:2]
    for label in labels:
        x1 = int((label.x - label.w / 2.0) * width)
        y1 = int((label.y - label.h / 2.0) * height)
        x2 = int((label.x + label.w / 2.0) * width)
        y2 = int((label.y + label.h / 2.0) * height)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        class_name = (
            classes[label.class_id]
            if 0 <= label.class_id < len(classes)
            else str(label.class_id)
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            image,
            class_name,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def write_previews(report: DatasetReport, preview_dir: Path, limit: int) -> int:
    count = 0
    for record in report.records:
        if count >= limit:
            break
        if not record.labels:
            continue
        image_path = Path(record.image)
        output_path = preview_dir / record.split / f'{image_path.stem}_labels.jpg'
        if draw_label_overlay(image_path, record.labels, report.classes, output_path):
            count += 1
    return count


def print_summary(report: DatasetReport) -> None:
    print('DATASET_AUDIT', flush=True)
    print(f'  root: {report.root}', flush=True)
    print(f'  ok: {report.ok}', flush=True)
    print(f'  classes: {len(report.classes)} {report.classes}', flush=True)
    print(f'  images: {report.split_image_counts}', flush=True)
    print(f'  label_files: {report.split_label_counts}', flush=True)
    print(f'  class_counts: {report.class_counts}', flush=True)
    print(f'  warnings: {len(report.warnings)}', flush=True)
    print(f'  errors: {len(report.errors)}', flush=True)
    for warning in report.warnings[:20]:
        print(f'WARNING: {warning}', flush=True)
    if len(report.warnings) > 20:
        print(f'WARNING: ... {len(report.warnings) - 20} more', flush=True)
    for error in report.errors[:20]:
        print(f'ERROR: {error}', flush=True)
    if len(report.errors) > 20:
        print(f'ERROR: ... {len(report.errors) - 20} more', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Audit an OriginCar YOLO dataset and generate helper artifacts.'
    )
    parser.add_argument('--dataset-dir',
                        default='/root/dev_ws/datasets/origin_competition')
    parser.add_argument('--write-yaml', action='store_true',
                        help='Write dataset.yaml under dataset dir unless --yaml-output is set.')
    parser.add_argument('--yaml-output',
                        help='Output path for YOLO dataset YAML.')
    parser.add_argument('--report-json',
                        help='Optional JSON report output path.')
    parser.add_argument('--preview-dir',
                        help='Optional directory for label overlay previews.')
    parser.add_argument('--preview-limit', type=int, default=20)
    parser.add_argument('--require-labels', action='store_true',
                        help='Warn on missing labels even under unlabeled split.')
    parser.add_argument('--strict', action='store_true',
                        help='Exit non-zero when warnings exist.')
    parser.add_argument('--json', action='store_true',
                        help='Print full JSON report to stdout.')
    return parser


def run(args: argparse.Namespace) -> int:
    root = Path(args.dataset_dir)
    report = audit_dataset(root, require_labels=args.require_labels)

    if args.write_yaml or args.yaml_output:
        yaml_output = Path(args.yaml_output) if args.yaml_output else root / 'dataset.yaml'
        write_yolo_yaml(report, yaml_output)
        print(f'WROTE_DATASET_YAML: {yaml_output}', flush=True)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f'WROTE_DATASET_REPORT: {report_path}', flush=True)

    if args.preview_dir:
        count = write_previews(report, Path(args.preview_dir), args.preview_limit)
        print(f'WROTE_LABEL_PREVIEWS: {count}', flush=True)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), flush=True)
    else:
        print_summary(report)

    if report.errors:
        return 2
    if args.strict and report.warnings:
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.preview_limit < 0:
        raise SystemExit('preview-limit must be >= 0')
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'DATASET_AUDIT_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
