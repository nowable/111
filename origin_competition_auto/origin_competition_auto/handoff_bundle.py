#!/usr/bin/env python3
"""Build a compact handoff bundle for field testing and review."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TEXT_EXTENSIONS = {
    '.json',
    '.jsonl',
    '.md',
    '.txt',
    '.log',
    '.yaml',
    '.yml',
    '.sh',
    '.py',
}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def safe_rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_selected_files(
    source_dir: Path,
    target_dir: Path,
    extensions: Iterable[str],
    include_images: bool = False,
    max_files: int = 0,
) -> List[Dict[str, str]]:
    copied: List[Dict[str, str]] = []
    if not source_dir.exists():
        return copied
    allowed = set(extensions)
    if include_images:
        allowed |= IMAGE_EXTENSIONS
    for src in sorted(path for path in source_dir.rglob('*') if path.is_file()):
        if src.suffix.lower() not in allowed:
            continue
        if max_files > 0 and len(copied) >= max_files:
            break
        rel = safe_rel(src, source_dir)
        dst = target_dir / rel
        if copy_file(src, dst):
            copied.append({'source': str(src), 'target': str(dst)})
    return copied


def default_path(board_path: str, local_relative: str) -> Path:
    board = Path(board_path)
    if board.exists():
        return board
    cwd = Path.cwd()
    candidates = [
        cwd / local_relative,
        cwd.parent / local_relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return board


def latest_files(root: Path, pattern: str, limit: int) -> List[Path]:
    if not root.exists():
        return []
    files = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return files[-limit:] if limit > 0 else files


def copy_docs(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    docs_dir = Path(args.docs_dir)
    return copy_selected_files(docs_dir, bundle_dir / 'docs', {'.md'})


def copy_configs(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    source_dir = Path(args.source_dir)
    copied = []
    copied.extend(copy_selected_files(source_dir / 'config', bundle_dir / 'config', {'.json'}))
    launch_dir = source_dir / 'launch'
    copied.extend(copy_selected_files(launch_dir, bundle_dir / 'launch', {'.py'}))
    return copied


def copy_solution_audit(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    copied = []
    for src in (
        Path(args.solution_report_json),
        Path(args.solution_report_md),
    ):
        if src.exists():
            dst = bundle_dir / 'solution_audit' / src.name
            if copy_file(src, dst):
                copied.append({'source': str(src), 'target': str(dst)})
    return copied


def copy_runs(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    runs_dir = Path(args.runs_dir)
    copied = []
    summaries = latest_files(runs_dir, '*/run_summary.json', args.run_limit)
    for summary in summaries:
        run_dir = summary.parent
        target = bundle_dir / 'runs' / run_dir.name
        copied.extend(
            copy_selected_files(
                run_dir,
                target,
                TEXT_EXTENSIONS,
                include_images=args.include_images,
                max_files=args.max_files_per_run,
            )
        )
    return copied


def copy_sessions(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    sessions_dir = Path(args.sessions_dir)
    copied = []
    manifests = latest_files(sessions_dir, '*/session_manifest.json', args.session_limit)
    for manifest in manifests:
        session_dir = manifest.parent
        target = bundle_dir / 'field_sessions' / session_dir.name
        copied.extend(
            copy_selected_files(
                session_dir,
                target,
                TEXT_EXTENSIONS,
                include_images=args.include_images,
                max_files=args.max_files_per_session,
            )
        )
    return copied


def copy_models(args: argparse.Namespace, bundle_dir: Path) -> List[Dict[str, str]]:
    models_dir = Path(args.models_dir)
    copied = []
    manifests = latest_files(models_dir, '*/model_manifest.json', args.model_limit)
    for manifest in manifests:
        model_dir = manifest.parent
        target = bundle_dir / 'models' / model_dir.name
        copied.extend(copy_selected_files(model_dir, target, TEXT_EXTENSIONS))
    return copied


def write_readme(bundle_dir: Path, index: Dict[str, object]) -> None:
    lines = [
        '# OriginCar Competition Handoff Bundle',
        '',
        f'generated_at: {index["generated_at"]}',
        '',
        '## Contents',
        '',
        '- `docs/`: project documents and field instructions.',
        '- `config/`: mission defaults, run profiles, and YOLO workconfig templates.',
        '- `solution_audit/`: latest readiness audit report when available.',
        '- `runs/`: recent `competition_run` outputs.',
        '- `field_sessions/`: field capture session manifests and review reports.',
        '- `models/`: YOLO pipeline manifests, workconfigs, and scripts.',
        '',
        '## Read First',
        '',
        '1. Open `docs/总方案索引与验收矩阵.md`.',
        '2. Open `solution_audit/solution_readiness_report.md` if present.',
        '3. Before motion, run `competition_run --profile safe_static` on the board.',
        '4. Use `field_session create/status/review` for real field data.',
        '',
        '## Bundle Stats',
        '',
    ]
    for key, value in index.get('counts', {}).items():
        lines.append(f'- {key}: {value}')
    lines.append('')
    (bundle_dir / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def build_bundle(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f'handoff_{timestamp()}'
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = {
        'docs': copy_docs(args, output_dir),
        'configs': copy_configs(args, output_dir),
        'solution_audit': copy_solution_audit(args, output_dir),
        'runs': copy_runs(args, output_dir),
        'field_sessions': copy_sessions(args, output_dir),
        'models': copy_models(args, output_dir),
    }
    index = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'bundle_dir': str(output_dir),
        'inputs': {
            'source_dir': args.source_dir,
            'docs_dir': args.docs_dir,
            'runs_dir': args.runs_dir,
            'sessions_dir': args.sessions_dir,
            'models_dir': args.models_dir,
            'include_images': args.include_images,
        },
        'counts': {name: len(files) for name, files in sections.items()},
        'sections': sections,
    }
    write_json(output_dir / 'bundle_index.json', index)
    write_readme(output_dir, index)

    archive_path = ''
    if args.zip:
        archive_path = shutil.make_archive(str(output_dir), 'zip', output_dir)

    if args.json:
        payload = dict(index)
        payload['archive'] = archive_path
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    else:
        print('HANDOFF_BUNDLE_CREATED', flush=True)
        print(f'  bundle_dir: {output_dir}', flush=True)
        for name, count in index['counts'].items():
            print(f'  {name}: {count}', flush=True)
        if archive_path:
            print(f'  archive: {archive_path}', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Create a compact OriginCar competition handoff bundle.'
    )
    parser.add_argument('--source-dir',
                        default=str(default_path('/root/dev_ws/src/origin_competition_auto', 'origin_competition_auto')))
    parser.add_argument('--docs-dir',
                        default=str(default_path('/root/dev_ws/origin_competition_project/docs', 'origin_competition_project/docs')))
    parser.add_argument('--runs-dir',
                        default=str(default_path('/root/dev_ws/runs', 'origin_competition_project/runs')))
    parser.add_argument('--sessions-dir',
                        default=str(default_path('/root/dev_ws/field_sessions', 'origin_competition_project/field_sessions')))
    parser.add_argument('--models-dir',
                        default=str(default_path('/root/dev_ws/models', 'origin_competition_project/models')))
    parser.add_argument('--solution-report-json',
                        default='/root/dev_ws/solution_readiness_report.json')
    parser.add_argument('--solution-report-md',
                        default='/root/dev_ws/solution_readiness_report.md')
    parser.add_argument('--output-root',
                        default='/root/dev_ws/handoff_bundles')
    parser.add_argument('--output-dir')
    parser.add_argument('--run-limit', type=int, default=5)
    parser.add_argument('--session-limit', type=int, default=5)
    parser.add_argument('--model-limit', type=int, default=5)
    parser.add_argument('--max-files-per-run', type=int, default=80)
    parser.add_argument('--max-files-per-session', type=int, default=120)
    parser.add_argument('--include-images', action='store_true')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return build_bundle(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'HANDOFF_BUNDLE_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
