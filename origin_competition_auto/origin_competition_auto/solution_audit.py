#!/usr/bin/env python3
"""Audit the implemented OriginCar competition solution against readiness gates."""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


EXPECTED_TOOLS = [
    'motion_calibration',
    'qr_capture',
    'qr_parse_debug',
    'auto_mission',
    'competition_run',
    'vision_debug',
    'llm_debug',
    'mission_replay',
    'dataset_capture',
    'dataset_audit',
    'system_check',
    'vision_tune',
    'field_data_review',
    'apply_review_recommendations',
    'field_session',
    'yolo_pipeline',
    'handoff_bundle',
]

EXPECTED_DOCS = [
    'auto_mission_v1_详细说明.md',
    'vision_yolo_实施说明.md',
    '数据采集与赛前自检说明.md',
    'YOLO训练准备说明.md',
    '离线主流程回放说明.md',
    '视觉阈值调优说明.md',
    '二维码指令格式说明.md',
    '赛前运行手册.md',
    '场地数据复盘与配置固化说明.md',
    '场地采集会话说明.md',
    '总方案索引与验收矩阵.md',
    '赛前交付包说明.md',
]


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def status(ok: bool, partial: bool = False) -> str:
    if ok:
        return 'pass'
    if partial:
        return 'partial'
    return 'fail'


def source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_docs_dir() -> Path:
    candidates = [
        Path('/root/dev_ws/origin_competition_project/docs'),
        Path.cwd() / 'origin_competition_project' / 'docs',
        Path.cwd().parent / 'origin_competition_project' / 'docs',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def default_runs_dir() -> Path:
    return Path('/root/dev_ws/runs') if Path('/root/dev_ws').exists() else Path.cwd() / 'runs'


def default_sessions_dir() -> Path:
    return (
        Path('/root/dev_ws/field_sessions')
        if Path('/root/dev_ws').exists()
        else Path.cwd() / 'field_sessions'
    )


def default_models_dir() -> Path:
    return Path('/root/dev_ws/models') if Path('/root/dev_ws').exists() else Path.cwd() / 'models'


def default_handoff_dir() -> Path:
    candidates = [
        Path('/root/dev_ws/handoff_bundles'),
        Path('/root/dev_ws/handoff'),
        Path.cwd() / 'handoff_bundles',
        Path.cwd() / 'handoff',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def check_tools(src: Path, setup_text: str) -> Dict[str, object]:
    missing = []
    present = []
    for tool in EXPECTED_TOOLS:
        if f'{tool} =' in setup_text:
            present.append(tool)
        else:
            missing.append(tool)
    files = {
        path.stem: str(path)
        for path in (src / 'origin_competition_auto').glob('*.py')
    }
    return {
        'status': status(not missing),
        'present': present,
        'missing': missing,
        'python_file_count': len(files),
    }


def check_configs(src: Path) -> Dict[str, object]:
    config_dir = src / 'config'
    expected = [
        'mission_defaults.json',
        'run_profiles.json',
        'rdk_yolov8workconfig.json',
    ]
    files = {name: str(config_dir / name) for name in expected}
    missing = [name for name in expected if not (config_dir / name).exists()]
    run_profiles = read_json(config_dir / 'run_profiles.json') or {}
    profiles = run_profiles.get('profiles', {}) if isinstance(run_profiles, dict) else {}
    safe_static_ok = False
    speed_ok = False
    if isinstance(profiles, dict):
        safe = profiles.get('safe_static', {})
        safe_static_ok = isinstance(safe, dict) and safe.get('motion_capable') is False
        speeds = []
        for profile in profiles.values():
            if isinstance(profile, dict) and 'cruise_linear' in profile:
                speeds.append(float(profile['cruise_linear']))
        speed_ok = bool(speeds) and max(speeds) <= 0.08
    return {
        'status': status(not missing and safe_static_ok and speed_ok),
        'files': files,
        'missing': missing,
        'safe_static_motion_disabled': safe_static_ok,
        'profile_speed_limit_ok': speed_ok,
    }


def check_docs(docs_dir: Path) -> Dict[str, object]:
    missing = [name for name in EXPECTED_DOCS if not (docs_dir / name).exists()]
    return {
        'status': status(not missing),
        'docs_dir': str(docs_dir),
        'missing': missing,
        'present_count': len(EXPECTED_DOCS) - len(missing),
        'expected_count': len(EXPECTED_DOCS),
    }


def run_summaries(runs_dir: Path) -> List[Dict[str, object]]:
    records = []
    if not runs_dir.exists():
        return records
    for path in sorted(runs_dir.glob('*/run_summary.json')):
        data = read_json(path)
        if data is not None:
            data['_path'] = str(path)
            records.append(data)
    return records


def newest(records: Iterable[Dict[str, object]], predicate) -> Optional[Dict[str, object]]:
    selected = [record for record in records if predicate(record)]
    if not selected:
        return None
    return selected[-1]


def check_runs(runs_dir: Path) -> Dict[str, object]:
    records = run_summaries(runs_dir)
    safe = newest(
        records,
        lambda item: item.get('ok') is True
        and item.get('profile') == 'safe_static'
        and item.get('motion_enabled') is False,
    )
    yolo = newest(
        records,
        lambda item: item.get('ok') is True
        and item.get('profile') == 'safe_static'
        and item.get('start_yolo') is True
        and item.get('require_yolo') is True,
    )
    safe_obs = safe.get('mission_observations', {}) if isinstance(safe, dict) else {}
    return {
        'status': status(bool(safe and yolo)),
        'runs_dir': str(runs_dir),
        'run_count': len(records),
        'safe_static_ok': bool(safe),
        'safe_static_run': safe.get('_path') if safe else '',
        'safe_static_final_state': safe_obs.get('final_state', '') if isinstance(safe_obs, dict) else '',
        'yolo_static_ok': bool(yolo),
        'yolo_static_run': yolo.get('_path') if yolo else '',
    }


def find_field_review(session_dir: Path) -> Optional[Path]:
    candidates = [
        session_dir / 'field_data_review' / 'field_data_review_report.json',
    ]
    candidates.extend(sorted(session_dir.glob('*/field_data_review_report.json')))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def check_field_sessions(sessions_dir: Path) -> Dict[str, object]:
    sessions = []
    if sessions_dir.exists():
        for manifest_path in sorted(sessions_dir.glob('*/session_manifest.json')):
            session_dir = manifest_path.parent
            review_path = find_field_review(session_dir)
            review = read_json(review_path) if review_path else None
            rec = {
                'session_dir': str(session_dir),
                'manifest': str(manifest_path),
                'review_report': str(review_path) if review_path else '',
                'review_ok': bool(review),
            }
            if isinstance(review, dict):
                recommendations = review.get('recommendations', {})
                if isinstance(recommendations, dict):
                    rec['readiness'] = recommendations.get('readiness', {})
                    rec['rates'] = recommendations.get('rates', {})
            sessions.append(rec)
    ok = any(item.get('review_ok') for item in sessions)
    return {
        'status': status(ok, partial=bool(sessions)),
        'sessions_dir': str(sessions_dir),
        'session_count': len(sessions),
        'reviewed_session_count': sum(1 for item in sessions if item.get('review_ok')),
        'sessions': sessions[-5:],
    }


def check_models(models_dir: Path) -> Dict[str, object]:
    manifests = []
    if models_dir.exists():
        for path in sorted(models_dir.glob('*/model_manifest.json')):
            manifest = read_json(path) or {}
            expected = manifest.get('expected_artifacts', {})
            rdk_bin = ''
            if isinstance(expected, dict):
                rdk_bin = str(expected.get('rdk_bin') or '')
            manifests.append(
                {
                    'manifest': str(path),
                    'workconfig': str(manifest.get('workconfig', '')),
                    'rdk_bin': rdk_bin,
                    'workconfig_exists': Path(str(manifest.get('workconfig', ''))).exists(),
                    'rdk_bin_exists': bool(rdk_bin) and Path(rdk_bin).exists(),
                }
            )
    prepared = any(item.get('workconfig_exists') for item in manifests)
    deployed = any(item.get('rdk_bin_exists') for item in manifests)
    return {
        'status': status(deployed, partial=prepared),
        'models_dir': str(models_dir),
        'manifest_count': len(manifests),
        'prepared_workconfig': prepared,
        'custom_rdk_model_ready': deployed,
        'manifests': manifests[-5:],
    }


def check_handoff_bundle(handoff_dir: Path) -> Dict[str, object]:
    indexes = []
    if handoff_dir.exists():
        for path in sorted(handoff_dir.glob('**/bundle_index.json')):
            data = read_json(path) or {}
            counts = data.get('counts', {}) if isinstance(data, dict) else {}
            indexes.append(
                {
                    'bundle_index': str(path),
                    'bundle_dir': str(path.parent),
                    'generated_at': data.get('generated_at', '') if isinstance(data, dict) else '',
                    'counts': counts if isinstance(counts, dict) else {},
                }
            )
    return {
        'status': status(bool(indexes), partial=True),
        'handoff_dir': str(handoff_dir),
        'bundle_count': len(indexes),
        'latest_bundles': indexes[-3:],
    }


def check_system(args: argparse.Namespace) -> Dict[str, object]:
    if not args.run_system_check:
        return {
            'status': 'skipped',
            'reason': 'pass --run-system-check to execute read-only ROS graph checks',
        }
    cmd = ['ros2', 'run', 'origin_competition_auto', 'system_check', '--json']
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.system_check_timeout,
        )
    except Exception as exc:
        return {'status': 'fail', 'error': str(exc), 'command': cmd}
    parsed = None
    try:
        parsed = json.JSONDecoder().raw_decode(proc.stdout.strip())[0]
    except Exception:
        pass
    ok = isinstance(parsed, dict) and parsed.get('ok') is True and proc.returncode == 0
    return {
        'status': status(ok),
        'command': cmd,
        'returncode': proc.returncode,
        'parsed': parsed,
        'stderr': proc.stderr,
    }


def build_matrix(sections: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    labels = {
        'tools': 'ROS2 tool entrypoints',
        'configs': 'Config and safety profiles',
        'docs': 'Documentation set',
        'runs': 'Static mission and YOLO run evidence',
        'field_sessions': 'Field data collection/review evidence',
        'models': 'Custom YOLO model artifacts',
        'handoff_bundle': 'Handoff bundle evidence',
        'system': 'Live board system check',
    }
    matrix = []
    for key, section in sections.items():
        matrix.append(
            {
                'key': key,
                'label': labels.get(key, key),
                'status': section.get('status', 'unknown'),
            }
        )
    return matrix


def write_markdown(path: Path, report: Dict[str, object]) -> None:
    lines = [
        '# OriginCar Solution Readiness Audit',
        '',
        f'generated_at: {report["generated_at"]}',
        '',
        '## Matrix',
        '',
        '| Key | Status | Label |',
        '| --- | --- | --- |',
    ]
    for row in report['matrix']:
        lines.append(f'| {row["key"]} | {row["status"]} | {row["label"]} |')
    lines.extend(
        [
            '',
            '## Remaining High-Impact Gaps',
            '',
        ]
    )
    gaps = report.get('remaining_gaps', [])
    if gaps:
        for gap in gaps:
            lines.append(f'- {gap}')
    else:
        lines.append('- None recorded by this audit.')
    lines.extend(
        [
            '',
            '## Evidence',
            '',
            '```json',
            json.dumps(report['sections'], ensure_ascii=False, indent=2),
            '```',
            '',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def remaining_gaps(sections: Dict[str, Dict[str, object]]) -> List[str]:
    gaps = []
    if sections['models'].get('custom_rdk_model_ready') is not True:
        gaps.append('Custom YOLO .bin model is not ready; current YOLO support is prepared/validated as a pipeline, not trained custom deployment.')
    if sections['field_sessions'].get('reviewed_session_count', 0) == 0:
        gaps.append('No reviewed field capture session found.')
    else:
        ready = False
        for session in sections['field_sessions'].get('sessions', []):
            readiness = session.get('readiness', {}) if isinstance(session, dict) else {}
            if isinstance(readiness, dict) and readiness.get('field_low_speed_ready') is True:
                ready = True
        if not ready:
            gaps.append('Reviewed field sessions are not yet field_low_speed_ready; real field data and QR direction samples are still needed.')
    if sections['system'].get('status') == 'skipped':
        gaps.append('Live system_check was skipped in this audit.')
    return gaps


def run(args: argparse.Namespace) -> int:
    src = Path(args.source_dir) if args.source_dir else source_root()
    setup_path = src / 'setup.py'
    setup_text = setup_path.read_text(encoding='utf-8') if setup_path.exists() else ''
    docs_dir = Path(args.docs_dir) if args.docs_dir else default_docs_dir()
    runs_dir = Path(args.runs_dir) if args.runs_dir else default_runs_dir()
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else default_sessions_dir()
    models_dir = Path(args.models_dir) if args.models_dir else default_models_dir()
    handoff_dir = Path(args.handoff_dir) if args.handoff_dir else default_handoff_dir()

    sections = {
        'tools': check_tools(src, setup_text),
        'configs': check_configs(src),
        'docs': check_docs(docs_dir),
        'runs': check_runs(runs_dir),
        'field_sessions': check_field_sessions(sessions_dir),
        'models': check_models(models_dir),
        'handoff_bundle': check_handoff_bundle(handoff_dir),
        'system': check_system(args),
    }
    report = {
        'generated_at': now(),
        'source_dir': str(src),
        'matrix': build_matrix(sections),
        'sections': sections,
        'remaining_gaps': remaining_gaps(sections),
    }
    if args.output_json:
        write_json(Path(args.output_json), report)
    if args.output_md:
        write_markdown(Path(args.output_md), report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        print('SOLUTION_AUDIT', flush=True)
        for row in report['matrix']:
            print(f'  {row["key"]}: {row["status"]} - {row["label"]}', flush=True)
        for gap in report['remaining_gaps']:
            print(f'GAP: {gap}', flush=True)
        if args.output_json:
            print(f'  output_json: {args.output_json}', flush=True)
        if args.output_md:
            print(f'  output_md: {args.output_md}', flush=True)
    fail_statuses = {'fail'}
    return 1 if any(row['status'] in fail_statuses for row in report['matrix']) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Audit implemented OriginCar competition solution readiness.'
    )
    parser.add_argument('--source-dir')
    parser.add_argument('--docs-dir')
    parser.add_argument('--runs-dir')
    parser.add_argument('--sessions-dir')
    parser.add_argument('--models-dir')
    parser.add_argument('--handoff-dir')
    parser.add_argument('--run-system-check', action='store_true')
    parser.add_argument('--system-check-timeout', type=float, default=20.0)
    parser.add_argument('--output-json')
    parser.add_argument('--output-md')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'SOLUTION_AUDIT_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
