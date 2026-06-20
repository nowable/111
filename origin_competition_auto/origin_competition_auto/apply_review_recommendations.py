#!/usr/bin/env python3
"""Safely apply field_data_review recommendation patches to config files."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PACKAGE_NAME = 'origin_competition_auto'

MISSION_ALLOWED_KEYS = {
    'obstacle_roi_y_ratio',
    'obstacle_min_area_ratio',
    'obstacle_danger_area_ratio',
    'obstacle_center_band_ratio',
    'black_v_max',
    'black_s_min',
    'target_min_area_ratio',
    'target_white_s_max',
    'target_white_v_min',
    'yolo_min_confidence',
}

RUN_PROFILE_ALLOWED_KEYS = {
    'require_yolo',
    'llm_mode',
    'max_runtime',
    'drive_to_qr_timeout',
    'scan_qr_timeout',
    'loop_duration',
    'return_duration',
    'cruise_linear',
}


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_json(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object: {path}')
    return data


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def source_config_dir() -> Path:
    candidates = [
        Path.cwd() / 'src' / PACKAGE_NAME / 'config',
        Path.cwd() / 'config',
        Path(__file__).resolve().parents[1] / 'config',
    ]
    for candidate in candidates:
        if (candidate / 'mission_defaults.json').exists() and (
            candidate / 'run_profiles.json'
        ).exists():
            return candidate
    return Path(__file__).resolve().parents[1] / 'config'


def default_patch_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    review_dir = Path(args.review_dir) if args.review_dir else None
    mission_patch = (
        Path(args.mission_patch)
        if args.mission_patch
        else (review_dir / 'recommended_mission_config_patch.json' if review_dir else None)
    )
    run_profiles_patch = (
        Path(args.run_profiles_patch)
        if args.run_profiles_patch
        else (review_dir / 'recommended_run_profiles_patch.json' if review_dir else None)
    )
    if mission_patch is None or run_profiles_patch is None:
        raise ValueError('provide --review-dir or both explicit patch paths')
    return mission_patch, run_profiles_patch


def ensure_allowed_keys(
    patch: Dict[str, object],
    allowed: Iterable[str],
    label: str,
) -> List[str]:
    allowed_set = set(allowed)
    unknown = sorted(key for key in patch if key not in allowed_set)
    if unknown:
        raise ValueError(f'{label} patch contains unsupported keys: {", ".join(unknown)}')
    return sorted(patch)


def validate_mission_patch(patch_doc: Dict[str, object], require_supervised: bool) -> Dict[str, object]:
    patch = patch_doc.get('patch')
    if not isinstance(patch, dict):
        raise ValueError('mission patch must contain object field "patch"')
    confidence = str(patch_doc.get('confidence') or '')
    if confidence != 'supervised' and require_supervised:
        raise ValueError(
            'mission patch confidence is not supervised; pass --allow-low-confidence to override'
        )
    ensure_allowed_keys(patch, MISSION_ALLOWED_KEYS, 'mission config')
    for key, value in patch.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f'mission config {key} must be numeric')
        if value < 0:
            raise ValueError(f'mission config {key} must be >= 0')
    return dict(patch)


def validate_run_profiles_patch(patch_doc: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    profiles = patch_doc.get('profiles')
    if not isinstance(profiles, dict):
        raise ValueError('run profiles patch must contain object field "profiles"')
    validated: Dict[str, Dict[str, object]] = {}
    for profile_name, patch in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(patch, dict):
            raise ValueError('run profile patches must map names to objects')
        ensure_allowed_keys(patch, RUN_PROFILE_ALLOWED_KEYS, f'run profile {profile_name}')
        for key, value in patch.items():
            if key == 'require_yolo':
                if not isinstance(value, bool):
                    raise ValueError(f'{profile_name}.{key} must be boolean')
            elif key == 'llm_mode':
                if value not in ('placeholder', 'disabled', 'openai-compatible'):
                    raise ValueError(
                        f'{profile_name}.{key} must be placeholder, disabled, or openai-compatible'
                    )
            else:
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValueError(f'{profile_name}.{key} must be numeric and >= 0')
                if key == 'cruise_linear' and float(value) > 0.08:
                    raise ValueError(f'{profile_name}.{key} must be <= 0.08')
        validated[profile_name] = dict(patch)
    return validated


def diff_mapping(before: Dict[str, object], patch: Dict[str, object]) -> Dict[str, object]:
    changes = {}
    for key, after in patch.items():
        before_value = before.get(key)
        if before_value != after:
            changes[key] = {
                'before': before_value,
                'after': after,
            }
    return changes


def apply_mission_patch(config: Dict[str, object], patch: Dict[str, object]) -> Dict[str, object]:
    updated = dict(config)
    updated.update(patch)
    return updated


def apply_run_profiles_patch(
    config: Dict[str, object],
    patch: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    updated = json.loads(json.dumps(config))
    profiles = updated.setdefault('profiles', {})
    if not isinstance(profiles, dict):
        raise ValueError('run_profiles config has no profiles object')
    for profile_name, profile_patch in patch.items():
        if profile_name not in profiles:
            raise ValueError(f'unknown run profile in patch: {profile_name}')
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(f'run profile is not an object: {profile_name}')
        profile.update(profile_patch)
    return updated


def backup_file(path: Path) -> str:
    backup = path.with_name(f'{path.name}.bak_{timestamp()}')
    shutil.copy2(path, backup)
    return str(backup)


def run(args: argparse.Namespace) -> int:
    config_dir = source_config_dir()
    mission_config_path = (
        Path(args.mission_config)
        if args.mission_config
        else config_dir / 'mission_defaults.json'
    )
    run_profiles_path = (
        Path(args.run_profiles_config)
        if args.run_profiles_config
        else config_dir / 'run_profiles.json'
    )
    mission_patch_path, run_profiles_patch_path = default_patch_paths(args)

    mission_patch_doc = load_json(mission_patch_path)
    run_profiles_patch_doc = load_json(run_profiles_patch_path)
    mission_patch = validate_mission_patch(
        mission_patch_doc,
        require_supervised=bool(args.apply and not args.allow_low_confidence),
    )
    run_profiles_patch = validate_run_profiles_patch(run_profiles_patch_doc)

    mission_config = load_json(mission_config_path)
    run_profiles_config = load_json(run_profiles_path)
    mission_changes = diff_mapping(mission_config, mission_patch)

    run_profile_changes: Dict[str, object] = {}
    profiles = run_profiles_config.get('profiles')
    if not isinstance(profiles, dict):
        raise ValueError('run profiles config has no profiles object')
    for profile_name, profile_patch in run_profiles_patch.items():
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f'unknown run profile in patch: {profile_name}')
        changes = diff_mapping(profile, profile_patch)
        if changes:
            run_profile_changes[profile_name] = changes

    report: Dict[str, object] = {
        'apply_requested': bool(args.apply),
        'mission_config': str(mission_config_path),
        'run_profiles_config': str(run_profiles_path),
        'mission_patch': str(mission_patch_path),
        'run_profiles_patch': str(run_profiles_patch_path),
        'mission_patch_confidence': mission_patch_doc.get('confidence', ''),
        'mission_changes': mission_changes,
        'run_profile_changes': run_profile_changes,
        'changed': bool(mission_changes or run_profile_changes),
        'written': False,
        'backups': {},
    }

    if args.apply and report['changed']:
        if args.backup:
            report['backups'] = {
                'mission_config': backup_file(mission_config_path),
                'run_profiles_config': backup_file(run_profiles_path),
            }
        write_json(mission_config_path, apply_mission_patch(mission_config, mission_patch))
        write_json(
            run_profiles_path,
            apply_run_profiles_patch(run_profiles_config, run_profiles_patch),
        )
        report['written'] = True

    if args.report_json:
        write_json(Path(args.report_json), report)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        status = 'APPLIED' if report['written'] else 'DRY_RUN'
        if args.apply and not report['changed']:
            status = 'NO_CHANGES'
        print(f'APPLY_REVIEW_RECOMMENDATIONS: {status}', flush=True)
        print(f'  mission_config: {mission_config_path}', flush=True)
        print(f'  run_profiles_config: {run_profiles_path}', flush=True)
        print(f'  mission_changes: {len(mission_changes)}', flush=True)
        print(f'  run_profile_changes: {len(run_profile_changes)}', flush=True)
        if args.report_json:
            print(f'  report: {args.report_json}', flush=True)
        if not args.apply:
            print('  note: pass --apply to write changes', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Apply field_data_review recommended config patches safely.'
    )
    parser.add_argument('--review-dir')
    parser.add_argument('--mission-patch')
    parser.add_argument('--run-profiles-patch')
    parser.add_argument('--mission-config')
    parser.add_argument('--run-profiles-config')
    parser.add_argument('--allow-low-confidence', action='store_true')
    parser.add_argument('--apply', action='store_true',
                        help='Write config files. Default is dry-run only.')
    parser.add_argument('--backup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--report-json')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f'APPLY_REVIEW_RECOMMENDATIONS_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
