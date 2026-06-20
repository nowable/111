#!/usr/bin/env python3
"""Offline parameter sweep for OpenCV vision thresholds."""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2

try:
    from origin_competition_auto.vision_detector import DetectorConfig, VisionDetector
except ImportError:
    from vision_detector import DetectorConfig, VisionDetector  # type: ignore


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
SPLITS = ('train', 'val', 'test', 'unlabeled')
OBSTACLE_LABELS = {'black_obstacle', 'obstacle'}
TARGET_LABELS = {'image_card', 'target_card', 'marker'}


@dataclass
class ImageTruth:
    obstacle_known: bool = False
    obstacle_expected: bool = False
    target_known: bool = False
    target_expected: bool = False


@dataclass
class ImageSample:
    path: Path
    truth: ImageTruth


def parse_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(',') if item.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(',') if item.strip()]


def iter_images(path: Path, recursive: bool = False) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if not path.exists():
        return []
    pattern = '**/*' if recursive else '*'
    return sorted(
        item for item in path.glob(pattern)
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_classes(dataset_dir: Optional[Path]) -> List[str]:
    if dataset_dir is None:
        return []
    path = dataset_dir / 'classes.txt'
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def label_path_for(dataset_dir: Path, split: str, image_path: Path) -> Path:
    return dataset_dir / 'labels' / split / f'{image_path.stem}.txt'


def truth_from_label_file(label_path: Path, classes: Sequence[str]) -> ImageTruth:
    truth = ImageTruth()
    if not label_path.exists():
        return truth
    truth.obstacle_known = True
    truth.target_known = True
    for raw_line in label_path.read_text(encoding='utf-8').splitlines():
        parts = raw_line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        if not 0 <= class_id < len(classes):
            continue
        class_name = classes[class_id]
        if class_name in OBSTACLE_LABELS:
            truth.obstacle_expected = True
        if class_name in TARGET_LABELS:
            truth.target_expected = True
    return truth


def collect_samples(args: argparse.Namespace) -> List[ImageSample]:
    samples: List[ImageSample] = []
    seen = set()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    classes = load_classes(dataset_dir)
    if dataset_dir is not None:
        for split in SPLITS:
            for image_path in iter_images(dataset_dir / 'images' / split, recursive=True):
                key = str(image_path)
                if key in seen:
                    continue
                seen.add(key)
                truth = (
                    truth_from_label_file(label_path_for(dataset_dir, split, image_path), classes)
                    if classes
                    else ImageTruth()
                )
                samples.append(ImageSample(image_path, truth))

    for raw in args.input_file or []:
        path = Path(raw)
        key = str(path)
        if key not in seen:
            seen.add(key)
            samples.append(ImageSample(path, ImageTruth()))

    for raw in args.input_dir or []:
        for path in iter_images(Path(raw), recursive=args.recursive):
            key = str(path)
            if key not in seen:
                seen.add(key)
                samples.append(ImageSample(path, ImageTruth()))

    if args.limit > 0:
        samples = samples[:args.limit]
    return samples


def config_from_values(values: Dict[str, object]) -> DetectorConfig:
    return DetectorConfig(
        obstacle_roi_y_ratio=float(values['obstacle_roi_y_ratio']),
        obstacle_min_area_ratio=float(values['obstacle_min_area_ratio']),
        obstacle_danger_area_ratio=float(values['obstacle_danger_area_ratio']),
        obstacle_center_band_ratio=float(values['obstacle_center_band_ratio']),
        black_v_max=int(values['black_v_max']),
        black_s_min=int(values['black_s_min']),
        target_min_area_ratio=float(values['target_min_area_ratio']),
        target_white_s_max=int(values['target_white_s_max']),
        target_white_v_min=int(values['target_white_v_min']),
    )


def config_to_dict(config: DetectorConfig) -> Dict[str, object]:
    return {
        'obstacle_roi_y_ratio': config.obstacle_roi_y_ratio,
        'obstacle_min_area_ratio': config.obstacle_min_area_ratio,
        'obstacle_danger_area_ratio': config.obstacle_danger_area_ratio,
        'obstacle_center_band_ratio': config.obstacle_center_band_ratio,
        'black_v_max': config.black_v_max,
        'black_s_min': config.black_s_min,
        'target_min_area_ratio': config.target_min_area_ratio,
        'target_white_s_max': config.target_white_s_max,
        'target_white_v_min': config.target_white_v_min,
    }


def make_grid(args: argparse.Namespace) -> Iterable[DetectorConfig]:
    keys = [
        'black_v_max',
        'black_s_min',
        'obstacle_roi_y_ratio',
        'obstacle_min_area_ratio',
        'obstacle_danger_area_ratio',
        'obstacle_center_band_ratio',
        'target_white_v_min',
        'target_white_s_max',
        'target_min_area_ratio',
    ]
    values = {
        'black_v_max': parse_int_list(args.black_v_max),
        'black_s_min': parse_int_list(args.black_s_min),
        'obstacle_roi_y_ratio': parse_float_list(args.obstacle_roi_y_ratio),
        'obstacle_min_area_ratio': parse_float_list(args.obstacle_min_area_ratio),
        'obstacle_danger_area_ratio': parse_float_list(args.obstacle_danger_area_ratio),
        'obstacle_center_band_ratio': parse_float_list(args.obstacle_center_band_ratio),
        'target_white_v_min': parse_int_list(args.target_white_v_min),
        'target_white_s_max': parse_int_list(args.target_white_s_max),
        'target_min_area_ratio': parse_float_list(args.target_min_area_ratio),
    }
    for key, items in values.items():
        if not items:
            raise ValueError(f'empty sweep list for {key}')
    for combo in itertools.product(*(values[key] for key in keys)):
        yield config_from_values(dict(zip(keys, combo)))


def update_binary_counts(
    counts: Dict[str, int],
    predicted: bool,
    known: bool,
    expected: bool,
) -> None:
    if predicted:
        counts['positive'] += 1
    if not known:
        counts['unknown'] += 1
        return
    counts['known'] += 1
    if predicted and expected:
        counts['tp'] += 1
    elif predicted and not expected:
        counts['fp'] += 1
    elif (not predicted) and expected:
        counts['fn'] += 1
    else:
        counts['tn'] += 1


def metrics(counts: Dict[str, int], total: int) -> Dict[str, float]:
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        'known': counts['known'],
        'unknown': counts['unknown'],
        'positive': counts['positive'],
        'positive_rate': counts['positive'] / max(1, total),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': counts['tn'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def evaluate_config(
    config: DetectorConfig,
    images: Sequence[Tuple[ImageSample, object]],
) -> Dict[str, object]:
    detector = VisionDetector(config)
    obstacle_counts = {
        'known': 0,
        'unknown': 0,
        'positive': 0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'tn': 0,
    }
    target_counts = dict(obstacle_counts)
    failures = 0
    total_area = 0.0

    for sample, image in images:
        if image is None:
            failures += 1
            continue
        decision = detector.analyze(image)
        update_binary_counts(
            obstacle_counts,
            decision.obstacle_danger,
            sample.truth.obstacle_known,
            sample.truth.obstacle_expected,
        )
        update_binary_counts(
            target_counts,
            decision.target_card is not None,
            sample.truth.target_known,
            sample.truth.target_expected,
        )
        total_area += decision.obstacle_area_ratio

    total = len(images)
    obstacle = metrics(obstacle_counts, total)
    target = metrics(target_counts, total)
    known_scores = []
    if obstacle['known'] > 0:
        known_scores.append(obstacle['f1'])
    if target['known'] > 0:
        known_scores.append(target['f1'])
    if known_scores:
        score = sum(known_scores) / len(known_scores)
    else:
        score = 0.0

    return {
        'score': score,
        'config': config_to_dict(config),
        'obstacle': obstacle,
        'target': target,
        'image_failures': failures,
        'avg_obstacle_area_ratio': total_area / max(1, total),
    }


def load_images(samples: Sequence[ImageSample]) -> List[Tuple[ImageSample, object]]:
    loaded = []
    for sample in samples:
        loaded.append((sample, cv2.imread(str(sample.path))))
    return loaded


def write_best_config(path: Path, result: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result['config'], ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def write_report(path: Path, report: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def print_results(report: Dict[str, object], top_n: int) -> None:
    print('VISION_TUNE', flush=True)
    print(f'  images: {report["image_count"]}', flush=True)
    print(f'  configs: {report["config_count"]}', flush=True)
    print(f'  obstacle_known: {report["truth"]["obstacle_known"]}', flush=True)
    print(f'  target_known: {report["truth"]["target_known"]}', flush=True)
    print(f'  warning: {report.get("warning", "")}', flush=True)
    for index, result in enumerate(report['top_results'][:top_n], start=1):
        obstacle = result['obstacle']
        target = result['target']
        print(
            f'TOP {index} score={result["score"]:.3f} '
            f'obstacle_f1={obstacle["f1"]:.3f} '
            f'obstacle_rate={obstacle["positive_rate"]:.3f} '
            f'target_f1={target["f1"]:.3f} '
            f'target_rate={target["positive_rate"]:.3f} '
            f'config={result["config"]}',
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Sweep OpenCV vision thresholds over saved images.'
    )
    parser.add_argument('--dataset-dir')
    parser.add_argument('--input-file', action='append')
    parser.add_argument('--input-dir', action='append')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir', default='/root/dev_ws/debug/vision_tune')
    parser.add_argument('--report-json')
    parser.add_argument('--write-best-config', action='store_true')
    parser.add_argument('--best-config-output')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--black-v-max', default='65,85,105')
    parser.add_argument('--black-s-min', default='0,20,40')
    parser.add_argument('--obstacle-roi-y-ratio', default='0.45,0.55,0.65')
    parser.add_argument('--obstacle-min-area-ratio', default='0.015,0.03,0.06')
    parser.add_argument('--obstacle-danger-area-ratio', default='0.035,0.06,0.10')
    parser.add_argument('--obstacle-center-band-ratio', default='0.30,0.36,0.45')
    parser.add_argument('--target-white-v-min', default='140,160,190')
    parser.add_argument('--target-white-s-max', default='60,90,130')
    parser.add_argument('--target-min-area-ratio', default='0.015,0.025,0.04')
    return parser


def run(args: argparse.Namespace) -> int:
    samples = collect_samples(args)
    if not samples:
        print('VISION_TUNE_ERROR: no input images', file=sys.stderr, flush=True)
        return 2
    loaded = load_images(samples)
    image_failures = sum(1 for _, image in loaded if image is None)
    if image_failures == len(loaded):
        print('VISION_TUNE_ERROR: no readable images', file=sys.stderr, flush=True)
        return 2

    configs = list(make_grid(args))
    results = [evaluate_config(config, loaded) for config in configs]
    results.sort(
        key=lambda item: (
            item['score'],
            -item['obstacle']['positive_rate'],
            item['target']['positive_rate'],
        ),
        reverse=True,
    )
    truth = {
        'obstacle_known': sum(1 for sample in samples if sample.truth.obstacle_known),
        'target_known': sum(1 for sample in samples if sample.truth.target_known),
    }
    warning = ''
    if truth['obstacle_known'] == 0 and truth['target_known'] == 0:
        warning = 'no ground-truth labels found; scores are not supervised'
    report = {
        'image_count': len(samples),
        'image_failures': image_failures,
        'config_count': len(configs),
        'truth': truth,
        'warning': warning,
        'top_results': results[: max(1, args.top_n)],
    }

    output_dir = Path(args.output_dir)
    report_path = Path(args.report_json) if args.report_json else output_dir / 'vision_tune_report.json'
    write_report(report_path, report)
    print(f'WROTE_VISION_TUNE_REPORT: {report_path}', flush=True)

    if args.write_best_config:
        best_path = (
            Path(args.best_config_output)
            if args.best_config_output
            else output_dir / 'best_detector_config.json'
        )
        write_best_config(best_path, results[0])
        print(f'WROTE_BEST_DETECTOR_CONFIG: {best_path}', flush=True)

    print_results(report, args.top_n)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit < 0:
        raise SystemExit('limit must be >= 0')
    if args.top_n < 1:
        raise SystemExit('top-n must be >= 1')
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'VISION_TUNE_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
