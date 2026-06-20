#!/usr/bin/env python3
"""Offline replay of the OriginCar perception and decision pipeline."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2

try:
    from origin_competition_auto.auto_mission import (
        QrDecoder,
        load_config,
    )
    from origin_competition_auto.decision_parser import parse_qr_instruction
    from origin_competition_auto.llm_client import LlmClient
    from origin_competition_auto.vision_detector import VisionDetector
except ImportError:
    from auto_mission import QrDecoder, load_config  # type: ignore
    from decision_parser import parse_qr_instruction  # type: ignore
    from llm_client import LlmClient  # type: ignore
    from vision_detector import VisionDetector  # type: ignore


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def collect_images(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for item in args.input_file or []:
        paths.append(Path(item))
    for item in args.input_dir or []:
        paths.extend(iter_images(Path(item), recursive=args.recursive))
    if args.dataset_dir:
        root = Path(args.dataset_dir)
        for split in ('train', 'val', 'test', 'unlabeled'):
            paths.extend(iter_images(root / 'images' / split, recursive=True))

    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if args.limit > 0:
        unique = unique[:args.limit]
    return unique


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


def safe_stem(path: Path, index: int) -> str:
    name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in path.stem)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return f'{index:04d}_{name}_{stamp}'


def save_image(path: Path, image) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f'failed to save image: {path}')
    return str(path)


def replay_image(
    image_path: Path,
    index: int,
    args: argparse.Namespace,
    detector: VisionDetector,
    qr_decoder: QrDecoder,
    llm_client: LlmClient,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        'image': str(image_path),
        'ok': False,
        'error': '',
    }
    image = cv2.imread(str(image_path))
    if image is None:
        record['error'] = 'cannot_read_image'
        return record

    height, width = image.shape[:2]
    qr = qr_decoder.detect(image)
    instruction = parse_qr_instruction(qr.text)
    direction = instruction.direction
    decision = detector.analyze(image)
    stem = safe_stem(image_path, index)
    output_dir = Path(args.output_dir)

    overlay_path = ''
    if args.save_overlays:
        overlay = detector.draw_debug(image, decision)
        overlay_path = save_image(output_dir / 'overlays' / f'{stem}_overlay.jpg', overlay)

    target_image_path = str(image_path)
    crop_path = ''
    if decision.target_card is not None and not args.no_crops:
        crop = detector.crop_detection(image, decision.target_card)
        crop_path = save_image(output_dir / 'crops' / f'{stem}_target_crop.jpg', crop)
        target_image_path = crop_path

    llm_result = ''
    if not args.skip_llm:
        llm_result = llm_client.analyze(target_image_path)

    detections = [
        detection.to_dict() for detection in (decision.detections or [])
    ]
    record.update(
        {
            'ok': True,
            'width': width,
            'height': height,
            'qr_text': qr.text,
            'qr_method': qr.method,
            'qr_points': (
                qr.points.reshape(-1, 2).tolist()
                if qr.points is not None
                else []
            ),
            'direction': direction,
            'direction_source': instruction.source,
            'direction_confidence': instruction.confidence,
            'direction_ambiguous': instruction.ambiguous,
            'direction_reason': instruction.reason,
            'instruction_fields': instruction.fields,
            'obstacle_danger': decision.obstacle_danger,
            'obstacle_zone': decision.obstacle_zone,
            'obstacle_area_ratio': decision.obstacle_area_ratio,
            'obstacle': decision.obstacle.to_dict() if decision.obstacle else None,
            'target_card': (
                decision.target_card.to_dict() if decision.target_card else None
            ),
            'detections': detections,
            'target_image': target_image_path,
            'crop_image': crop_path,
            'overlay_image': overlay_path,
            'llm_result': llm_result,
            'offline_next_state': infer_next_state(qr.text, direction),
        }
    )
    return record


def infer_next_state(qr_text: str, direction: str) -> str:
    if not qr_text:
        return 'STOP_QR_NOT_FOUND'
    if direction == 'left':
        return 'LOOP_LEFT'
    if direction == 'right':
        return 'LOOP_RIGHT'
    return 'STOP_DIRECTION_UNKNOWN'


def summarize(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ok_records = [record for record in records if record.get('ok')]
    qr_count = sum(1 for record in ok_records if record.get('qr_text'))
    left_count = sum(1 for record in ok_records if record.get('direction') == 'left')
    right_count = sum(1 for record in ok_records if record.get('direction') == 'right')
    obstacle_count = sum(
        1 for record in ok_records if bool(record.get('obstacle_danger'))
    )
    target_count = sum(1 for record in ok_records if record.get('target_card'))
    error_count = len(records) - len(ok_records)
    return {
        'total': len(records),
        'ok': len(ok_records),
        'errors': error_count,
        'qr_found': qr_count,
        'left': left_count,
        'right': right_count,
        'obstacle_danger': obstacle_count,
        'target_card': target_count,
    }


def write_report(output_dir: Path, records: Sequence[Dict[str, object]]) -> str:
    path = output_dir / 'mission_replay_report.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'summary': summarize(records),
        'records': list(records),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)


def print_record(record: Dict[str, object]) -> None:
    if not record.get('ok'):
        print(f'REPLAY_ERROR image={record["image"]} error={record["error"]}', flush=True)
        return
    print(
        'REPLAY_IMAGE '
        f'image={record["image"]} '
        f'qr={record.get("qr_text") or ""!r} '
        f'direction={record.get("direction") or ""} '
        f'obstacle={record.get("obstacle_danger")}:{record.get("obstacle_zone")} '
        f'target={bool(record.get("target_card"))} '
        f'next={record.get("offline_next_state")}',
        flush=True,
    )
    llm_result = str(record.get('llm_result') or '')
    if llm_result:
        print(f'  LLM_RESULT: {llm_result}', flush=True)
    if record.get('overlay_image'):
        print(f'  OVERLAY: {record["overlay_image"]}', flush=True)
    if record.get('crop_image'):
        print(f'  CROP: {record["crop_image"]}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Offline replay for QR, vision, target crop, and LLM decisions.'
    )
    parser.add_argument('--input-file', action='append',
                        help='Image file to replay. May be repeated.')
    parser.add_argument('--input-dir', action='append',
                        help='Directory containing images. May be repeated.')
    parser.add_argument('--dataset-dir',
                        help='Dataset root with images/train,val,test,unlabeled.')
    parser.add_argument('--recursive', action='store_true',
                        help='Recurse through --input-dir directories.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Maximum number of images to replay; 0 means unlimited.')
    parser.add_argument('--config', help='Mission JSON config path.')
    parser.add_argument('--output-dir',
                        default='/root/dev_ws/debug/mission_replay')
    parser.add_argument('--save-overlays', action='store_true')
    parser.add_argument('--no-crops', action='store_true',
                        help='Do not save target-card crops.')
    parser.add_argument('--skip-llm', action='store_true')
    parser.add_argument('--llm-mode',
                        choices=['placeholder', 'disabled', 'openai-compatible'],
                        help='Override LLM mode from mission config.')
    parser.add_argument('--llm-api-url')
    parser.add_argument('--llm-model')
    parser.add_argument('--llm-api-key-env')
    parser.add_argument('--llm-timeout', type=float)
    parser.add_argument('--llm-prompt')
    parser.add_argument('--json', action='store_true',
                        help='Print full JSON report to stdout.')
    return parser


def apply_llm_overrides(config, args: argparse.Namespace) -> None:
    for attr, value in {
        'llm_mode': args.llm_mode,
        'llm_api_url': args.llm_api_url,
        'llm_model': args.llm_model,
        'llm_api_key_env': args.llm_api_key_env,
        'llm_timeout': args.llm_timeout,
        'llm_prompt': args.llm_prompt,
    }.items():
        if value is not None:
            setattr(config, attr, value)


def run(args: argparse.Namespace) -> int:
    image_paths = collect_images(args)
    if not image_paths:
        print('MISSION_REPLAY_ERROR: no input images', file=sys.stderr, flush=True)
        return 2

    config = load_config(args.config)
    apply_llm_overrides(config, args)
    config.validate()

    detector = VisionDetector(config.detector_config())
    qr_decoder = QrDecoder()
    llm_client = LlmClient(config.llm_config())

    records = []
    for index, image_path in enumerate(image_paths, start=1):
        record = replay_image(
            image_path=image_path,
            index=index,
            args=args,
            detector=detector,
            qr_decoder=qr_decoder,
            llm_client=llm_client,
        )
        records.append(record)
        if not args.json:
            print_record(record)

    report_path = write_report(Path(args.output_dir), records)
    summary = summarize(records)
    if args.json:
        print(
            json.dumps(
                {'summary': summary, 'records': records, 'report': report_path},
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    else:
        print(f'MISSION_REPLAY_SUMMARY: {summary}', flush=True)
        print(f'WROTE_MISSION_REPLAY_REPORT: {report_path}', flush=True)
    return 0 if summary['errors'] == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit < 0:
        raise SystemExit('limit must be >= 0')
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'MISSION_REPLAY_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
