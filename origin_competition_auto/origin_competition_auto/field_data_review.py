#!/usr/bin/env python3
"""Pre-field data review pipeline for OriginCar competition assets."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from origin_competition_auto import dataset_audit, mission_replay, vision_tune
except ImportError:
    import dataset_audit  # type: ignore
    import mission_replay  # type: ignore
    import vision_tune  # type: ignore


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, dict) else {}


def rate(part: int, total: int) -> float:
    return part / total if total else 0.0


def count_labeled_images(records: Sequence[Dict[str, object]]) -> int:
    return sum(1 for record in records if record.get('labels'))


def run_dataset_audit(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    if not args.dataset_dir or args.skip_audit:
        return {'enabled': False}

    dataset_dir = Path(args.dataset_dir)
    report = dataset_audit.audit_dataset(
        dataset_dir,
        require_labels=args.require_labels,
    )
    report_data = report.to_dict()
    report_path = output_dir / 'dataset_audit_report.json'
    write_json(report_path, report_data)

    artifacts: Dict[str, object] = {
        'report': str(report_path),
    }
    if args.write_dataset_yaml:
        yaml_path = output_dir / 'dataset.yaml'
        dataset_audit.write_yolo_yaml(report, yaml_path)
        artifacts['dataset_yaml'] = str(yaml_path)
    if args.preview_limit > 0:
        preview_dir = output_dir / 'label_previews'
        count = dataset_audit.write_previews(report, preview_dir, args.preview_limit)
        artifacts['label_preview_dir'] = str(preview_dir)
        artifacts['label_preview_count'] = count

    records = report_data.get('records', [])
    labeled_images = count_labeled_images(records if isinstance(records, list) else [])
    return {
        'enabled': True,
        'ok': bool(report_data.get('ok')),
        'artifacts': artifacts,
        'summary': {
            'root': report_data.get('root', ''),
            'classes': report_data.get('classes', []),
            'split_image_counts': report_data.get('split_image_counts', {}),
            'split_label_counts': report_data.get('split_label_counts', {}),
            'class_counts': report_data.get('class_counts', {}),
            'warnings': len(report_data.get('warnings', [])),
            'errors': len(report_data.get('errors', [])),
            'labeled_images': labeled_images,
        },
    }


def replay_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_file=args.input_file,
        input_dir=args.input_dir,
        dataset_dir=args.dataset_dir,
        recursive=args.recursive,
        limit=args.limit,
        config=args.config,
        output_dir=str(output_dir),
        save_overlays=not args.no_overlays,
        no_crops=args.no_crops,
        skip_llm=True,
        llm_mode='disabled',
        llm_api_url=None,
        llm_model=None,
        llm_api_key_env=None,
        llm_timeout=None,
        llm_prompt=None,
        json=False,
    )


def run_mission_replay(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    if args.skip_replay:
        return {'enabled': False}

    selected_args = replay_args(args, output_dir)
    image_paths = mission_replay.collect_images(selected_args)
    if not image_paths:
        return {
            'enabled': True,
            'ok': False,
            'error': 'no input images',
            'summary': {'total': 0},
            'artifacts': {},
        }

    returncode = mission_replay.run(selected_args)
    report_path = output_dir / 'mission_replay_report.json'
    report = load_json(report_path)
    return {
        'enabled': True,
        'ok': returncode == 0,
        'returncode': returncode,
        'artifacts': {
            'report': str(report_path),
            'output_dir': str(output_dir),
        },
        'summary': report.get('summary', {}),
    }


def tune_args(args: argparse.Namespace, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_dir=args.dataset_dir,
        input_file=args.input_file,
        input_dir=args.input_dir,
        recursive=args.recursive,
        limit=args.limit,
        output_dir=str(output_dir),
        report_json=str(output_dir / 'vision_tune_report.json'),
        write_best_config=True,
        best_config_output=str(output_dir / 'best_detector_config.json'),
        top_n=args.top_n,
        black_v_max=args.black_v_max,
        black_s_min=args.black_s_min,
        obstacle_roi_y_ratio=args.obstacle_roi_y_ratio,
        obstacle_min_area_ratio=args.obstacle_min_area_ratio,
        obstacle_danger_area_ratio=args.obstacle_danger_area_ratio,
        obstacle_center_band_ratio=args.obstacle_center_band_ratio,
        target_white_v_min=args.target_white_v_min,
        target_white_s_max=args.target_white_s_max,
        target_min_area_ratio=args.target_min_area_ratio,
    )


def run_vision_tune(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    if args.skip_vision_tune:
        return {'enabled': False}

    selected_args = tune_args(args, output_dir)
    returncode = vision_tune.run(selected_args)
    report_path = output_dir / 'vision_tune_report.json'
    best_config_path = output_dir / 'best_detector_config.json'
    report = load_json(report_path)
    top_results = report.get('top_results', [])
    best = top_results[0] if isinstance(top_results, list) and top_results else {}
    return {
        'enabled': True,
        'ok': returncode == 0,
        'returncode': returncode,
        'artifacts': {
            'report': str(report_path),
            'best_detector_config': str(best_config_path),
        },
        'summary': {
            'image_count': report.get('image_count', 0),
            'image_failures': report.get('image_failures', 0),
            'config_count': report.get('config_count', 0),
            'truth': report.get('truth', {}),
            'warning': report.get('warning', ''),
            'best_score': best.get('score', 0.0) if isinstance(best, dict) else 0.0,
            'best_config': best.get('config', {}) if isinstance(best, dict) else {},
        },
    }


def replay_rates(replay_summary: Dict[str, object]) -> Dict[str, float]:
    total = int(replay_summary.get('total') or 0)
    return {
        'qr_found_rate': rate(int(replay_summary.get('qr_found') or 0), total),
        'direction_rate': rate(
            int(replay_summary.get('left') or 0) + int(replay_summary.get('right') or 0),
            total,
        ),
        'target_card_rate': rate(int(replay_summary.get('target_card') or 0), total),
        'obstacle_danger_rate': rate(int(replay_summary.get('obstacle_danger') or 0), total),
    }


def make_recommendations(
    args: argparse.Namespace,
    audit: Dict[str, object],
    replay: Dict[str, object],
    tune: Dict[str, object],
    output_dir: Path,
) -> Dict[str, object]:
    warnings: List[str] = []
    next_actions: List[str] = []

    audit_summary = audit.get('summary', {}) if isinstance(audit.get('summary'), dict) else {}
    replay_summary = replay.get('summary', {}) if isinstance(replay.get('summary'), dict) else {}
    tune_summary = tune.get('summary', {}) if isinstance(tune.get('summary'), dict) else {}
    rates = replay_rates(replay_summary)

    dataset_ok = (not audit.get('enabled')) or bool(audit.get('ok'))
    replay_ok = bool(replay.get('enabled')) and bool(replay.get('ok'))
    tune_ok = (not tune.get('enabled')) or bool(tune.get('ok'))
    labeled_images = int(audit_summary.get('labeled_images') or 0)
    image_total = int(replay_summary.get('total') or 0)
    truth = tune_summary.get('truth', {}) if isinstance(tune_summary.get('truth'), dict) else {}
    supervised_tune = (
        int(truth.get('obstacle_known') or 0) > 0
        or int(truth.get('target_known') or 0) > 0
    )

    if image_total == 0:
        warnings.append('no images available for replay')
        next_actions.append('采集至少一轮二维码、障碍物、目标图和通道样本')
    if rates['qr_found_rate'] < args.min_qr_rate:
        warnings.append(
            f'qr_found_rate {rates["qr_found_rate"]:.2f} below {args.min_qr_rate:.2f}'
        )
        next_actions.append('补采不同距离、角度和光照下的二维码图片')
    if rates['direction_rate'] < args.min_qr_rate:
        warnings.append(
            f'direction_rate {rates["direction_rate"]:.2f} below {args.min_qr_rate:.2f}'
        )
        next_actions.append('确认比赛二维码内容包含 direction=left/right 或中文左/右')
    if rates['target_card_rate'] < args.min_target_rate:
        warnings.append(
            f'target_card_rate {rates["target_card_rate"]:.2f} below {args.min_target_rate:.2f}'
        )
        next_actions.append('补采图文标记牌和物品图像，必要时调 target 阈值')
    if audit.get('enabled') and labeled_images < args.min_labeled_images:
        warnings.append(
            f'labeled_images {labeled_images} below {args.min_labeled_images}'
        )
        next_actions.append('补充 YOLO 标注数据，至少覆盖障碍物、二维码牌、图文标记牌')
    if tune.get('enabled') and not supervised_tune:
        warnings.append('vision_tune has no supervised labels; best config is low-confidence')
        next_actions.append('给采集图片补 YOLO 标签后重新运行 field_data_review')

    best_config = tune_summary.get('best_config', {})
    mission_config_patch = dict(best_config) if isinstance(best_config, dict) else {}
    mission_config_patch_path = output_dir / 'recommended_mission_config_patch.json'
    write_json(
        mission_config_patch_path,
        {
            'confidence': 'supervised' if supervised_tune else 'low_unsupervised',
            'patch': mission_config_patch,
            'note': (
                'Apply these keys to mission_defaults.json only after image samples look correct.'
            ),
        },
    )

    yolo_training_ready = dataset_ok and labeled_images >= args.min_labeled_images
    yolo_runtime_recommended = bool(args.recommend_yolo_runtime and yolo_training_ready)
    run_profiles_patch = {
        'profiles': {
            'safe_static': {
                'require_yolo': False,
            },
            'field_low_speed': {
                'require_yolo': yolo_runtime_recommended,
            },
        },
        'note': (
            'Keep field_low_speed require_yolo=false until a trained/validated detector '
            'is available, unless --recommend-yolo-runtime was used and data is sufficient.'
        ),
    }
    run_profiles_patch_path = output_dir / 'recommended_run_profiles_patch.json'
    write_json(run_profiles_patch_path, run_profiles_patch)

    readiness = {
        'dataset_ok': dataset_ok,
        'replay_ok': replay_ok,
        'vision_tune_ok': tune_ok,
        'supervised_tune': supervised_tune,
        'yolo_training_ready': yolo_training_ready,
        'safe_static_review_ready': replay_ok and image_total > 0,
        'field_low_speed_ready': (
            dataset_ok
            and replay_ok
            and image_total > 0
            and rates['qr_found_rate'] >= args.min_qr_rate
            and rates['direction_rate'] >= args.min_qr_rate
        ),
    }

    return {
        'readiness': readiness,
        'rates': rates,
        'warnings': warnings,
        'next_actions': next_actions,
        'artifacts': {
            'recommended_mission_config_patch': str(mission_config_patch_path),
            'recommended_run_profiles_patch': str(run_profiles_patch_path),
        },
    }


def print_summary(report: Dict[str, object]) -> None:
    readiness = report.get('recommendations', {}).get('readiness', {})
    rates = report.get('recommendations', {}).get('rates', {})
    warnings = report.get('recommendations', {}).get('warnings', [])
    artifacts = report.get('artifacts', {})
    print('FIELD_DATA_REVIEW', flush=True)
    print(f'  output_dir: {report.get("output_dir")}', flush=True)
    print(f'  readiness: {readiness}', flush=True)
    print(f'  rates: {rates}', flush=True)
    print(f'  warnings: {len(warnings)}', flush=True)
    for warning in warnings[:20]:
        print(f'WARNING: {warning}', flush=True)
    print(f'  report: {artifacts.get("report")}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run dataset audit, mission replay, and vision tuning as one review.'
    )
    parser.add_argument('--dataset-dir')
    parser.add_argument('--input-file', action='append')
    parser.add_argument('--input-dir', action='append')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir', default='/root/dev_ws/debug/field_data_review')
    parser.add_argument('--config')
    parser.add_argument('--skip-audit', action='store_true')
    parser.add_argument('--skip-replay', action='store_true')
    parser.add_argument('--skip-vision-tune', action='store_true')
    parser.add_argument('--require-labels', action='store_true')
    parser.add_argument('--write-dataset-yaml', action='store_true')
    parser.add_argument('--preview-limit', type=int, default=20)
    parser.add_argument('--no-overlays', action='store_true')
    parser.add_argument('--no-crops', action='store_true')
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--strict', action='store_true',
                        help='Exit non-zero when readiness warnings remain.')
    parser.add_argument('--min-qr-rate', type=float, default=0.5)
    parser.add_argument('--min-target-rate', type=float, default=0.1)
    parser.add_argument('--min-labeled-images', type=int, default=20)
    parser.add_argument('--recommend-yolo-runtime', action='store_true')
    parser.add_argument('--black-v-max', default='65,85')
    parser.add_argument('--black-s-min', default='0,20')
    parser.add_argument('--obstacle-roi-y-ratio', default='0.45,0.55')
    parser.add_argument('--obstacle-min-area-ratio', default='0.015,0.03')
    parser.add_argument('--obstacle-danger-area-ratio', default='0.035,0.06')
    parser.add_argument('--obstacle-center-band-ratio', default='0.30,0.36')
    parser.add_argument('--target-white-v-min', default='140,160')
    parser.add_argument('--target-white-s-max', default='60,90')
    parser.add_argument('--target-min-area-ratio', default='0.015,0.025')
    parser.add_argument('--json', action='store_true')
    return parser


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_output_dir = output_dir / 'dataset_audit'
    replay_output_dir = output_dir / 'mission_replay'
    tune_output_dir = output_dir / 'vision_tune'

    audit = run_dataset_audit(args, audit_output_dir)
    replay = run_mission_replay(args, replay_output_dir)
    tune = run_vision_tune(args, tune_output_dir)
    recommendations = make_recommendations(args, audit, replay, tune, output_dir)

    report = {
        'output_dir': str(output_dir),
        'inputs': {
            'dataset_dir': args.dataset_dir or '',
            'input_file': args.input_file or [],
            'input_dir': args.input_dir or [],
            'recursive': args.recursive,
            'limit': args.limit,
        },
        'dataset_audit': audit,
        'mission_replay': replay,
        'vision_tune': tune,
        'recommendations': recommendations,
        'artifacts': {
            'report': str(output_dir / 'field_data_review_report.json'),
        },
    }
    write_json(output_dir / 'field_data_review_report.json', report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        print_summary(report)

    if not replay.get('enabled') or not replay.get('ok'):
        return 2
    if args.strict and recommendations.get('warnings'):
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.limit < 0:
        raise SystemExit('limit must be >= 0')
    if args.preview_limit < 0:
        raise SystemExit('preview-limit must be >= 0')
    if args.top_n < 1:
        raise SystemExit('top-n must be >= 1')
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'FIELD_DATA_REVIEW_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
