#!/usr/bin/env python3
"""Create and manage field data collection sessions."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from origin_competition_auto.dataset_capture import DEFAULT_CLASSES
    from origin_competition_auto import field_data_review
except ImportError:
    from dataset_capture import DEFAULT_CLASSES  # type: ignore
    import field_data_review  # type: ignore


DEFAULT_TASKS = [
    {
        'label': 'qr_left',
        'target_count': 20,
        'auto_label_mode': 'none',
        'note': '二维码左转指令，不同距离、角度、光照',
    },
    {
        'label': 'qr_right',
        'target_count': 20,
        'auto_label_mode': 'none',
        'note': '二维码右转指令，不同距离、角度、光照',
    },
    {
        'label': 'black_obstacle',
        'target_count': 30,
        'auto_label_mode': 'opencv-obstacle',
        'note': '黑色三角障碍物，左/中/右和不同距离',
    },
    {
        'label': 'image_card',
        'target_count': 20,
        'auto_label_mode': 'opencv-target',
        'note': '图文标记牌、物品图或目标图区域',
    },
    {
        'label': 'empty_floor',
        'target_count': 20,
        'auto_label_mode': 'none',
        'note': '无障碍、无二维码、无目标牌的负样本',
    },
    {
        'label': 'corridor_color',
        'target_count': 20,
        'auto_label_mode': 'none',
        'note': '黄色通道、蓝色大厅、浅绿色诊疗室等颜色样本',
    },
]


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_json(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'expected JSON object: {path}')
    return data


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


def default_session_name() -> str:
    return f'field_{timestamp()}'


def ensure_dataset_dirs(dataset_dir: Path, classes: Iterable[str]) -> None:
    for split in ('train', 'val', 'test', 'unlabeled'):
        (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'debug' / split).mkdir(parents=True, exist_ok=True)
    classes_path = dataset_dir / 'classes.txt'
    if not classes_path.exists():
        classes_path.write_text('\n'.join(classes) + '\n', encoding='utf-8')


def capture_command(
    dataset_dir: Path,
    label: str,
    count: int,
    interval: float,
    timeout: float,
    auto_label_mode: str,
    topic: str,
) -> List[str]:
    cmd = [
        'ros2',
        'run',
        'origin_competition_auto',
        'dataset_capture',
        '--topic',
        topic,
        '--output-dir',
        str(dataset_dir),
        '--split',
        'unlabeled',
        '--label',
        label,
        '--prefix',
        label,
        '--count',
        str(count),
        '--interval',
        str(interval),
        '--timeout',
        str(timeout),
        '--source-note',
        label,
        '--auto-label-mode',
        auto_label_mode,
    ]
    if auto_label_mode != 'none':
        cmd.append('--save-overlay')
    return cmd


def command_line(cmd: List[str]) -> str:
    return ' '.join(shell_quote(part) for part in cmd)


def build_manifest(args: argparse.Namespace, session_dir: Path, dataset_dir: Path) -> Dict[str, object]:
    tasks = []
    for task in DEFAULT_TASKS:
        tasks.append(
            {
                **task,
                'command': capture_command(
                    dataset_dir=dataset_dir,
                    label=str(task['label']),
                    count=int(task['target_count']),
                    interval=args.interval,
                    timeout=args.timeout,
                    auto_label_mode=str(task['auto_label_mode']),
                    topic=args.topic,
                ),
            }
        )
    return {
        'session_name': session_dir.name,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'session_dir': str(session_dir),
        'dataset_dir': str(dataset_dir),
        'topic': args.topic,
        'classes': list(DEFAULT_CLASSES),
        'tasks': tasks,
        'review_command': [
            'ros2',
            'run',
            'origin_competition_auto',
            'field_data_review',
            '--dataset-dir',
            str(dataset_dir),
            '--output-dir',
            str(session_dir / 'field_data_review'),
            '--recursive',
            '--write-dataset-yaml',
        ],
    }


def write_checklist(session_dir: Path, manifest: Dict[str, object]) -> None:
    lines = [
        '# OriginCar Field Data Collection Checklist',
        '',
        f'session: {manifest["session_name"]}',
        f'dataset: {manifest["dataset_dir"]}',
        '',
        '## Steps',
        '',
        '1. Keep the car stationary unless a separate motion test is being run.',
        '2. Source ROS environments and confirm /image is publishing.',
        '3. Run each capture command below while placing the matching object in view.',
        '4. Run field_data_review before any field_low_speed motion test.',
        '',
        '## Capture Tasks',
        '',
    ]
    for task in manifest['tasks']:  # type: ignore[index]
        lines.extend(
            [
                f'### {task["label"]}',
                '',
                f'- target_count: {task["target_count"]}',
                f'- note: {task["note"]}',
                '',
                '```bash',
                command_line(task['command']),
                '```',
                '',
            ]
        )
    lines.extend(
        [
            '## Review',
            '',
            '```bash',
            command_line(manifest['review_command']),  # type: ignore[arg-type]
            '```',
            '',
        ]
    )
    (session_dir / '采集清单.md').write_text('\n'.join(lines), encoding='utf-8')


def write_commands_script(session_dir: Path, manifest: Dict[str, object]) -> None:
    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        'source /opt/ros/humble/setup.bash',
        'source /opt/tros/humble/setup.bash 2>/dev/null || true',
        'source /root/dev_ws/install/setup.bash',
        '',
        'echo "Run the capture commands one by one. Do not run this whole script blindly."',
        'echo "Open 采集清单.md and copy the matching command for the current scene."',
        '',
        '# Capture commands:',
    ]
    for task in manifest['tasks']:  # type: ignore[index]
        lines.append(f'# {task["label"]}: {command_line(task["command"])}')
    lines.extend(
        [
            '',
            '# Review command:',
            '# ' + command_line(manifest['review_command']),  # type: ignore[arg-type]
            '',
        ]
    )
    path = session_dir / 'commands.sh'
    path.write_text('\n'.join(lines), encoding='utf-8')
    try:
        path.chmod(0o755)
    except OSError:
        pass


def create_session(args: argparse.Namespace) -> int:
    session_name = args.name or default_session_name()
    session_dir = Path(args.sessions_root) / session_name
    if session_dir.exists() and not args.force:
        raise ValueError(f'session already exists: {session_dir}')
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else session_dir / 'dataset'
    session_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_dirs(dataset_dir, DEFAULT_CLASSES)

    manifest = build_manifest(args, session_dir, dataset_dir)
    write_json(session_dir / 'session_manifest.json', manifest)
    write_checklist(session_dir, manifest)
    write_commands_script(session_dir, manifest)
    status = build_status(session_dir, dataset_dir)
    write_json(session_dir / 'session_status.json', status)

    print('FIELD_SESSION_CREATED', flush=True)
    print(f'  session_dir: {session_dir}', flush=True)
    print(f'  dataset_dir: {dataset_dir}', flush=True)
    print(f'  checklist: {session_dir / "采集清单.md"}', flush=True)
    print(f'  commands: {session_dir / "commands.sh"}', flush=True)
    return 0


def iter_metadata(dataset_dir: Path) -> Iterable[Dict[str, object]]:
    path = dataset_dir / 'metadata.jsonl'
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def count_images(dataset_dir: Path) -> Dict[str, int]:
    counts = {}
    for split in ('train', 'val', 'test', 'unlabeled'):
        image_dir = dataset_dir / 'images' / split
        counts[split] = sum(
            1
            for path in image_dir.rglob('*')
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ) if image_dir.exists() else 0
    return counts


def load_manifest(session_dir: Path) -> Dict[str, object]:
    path = session_dir / 'session_manifest.json'
    if not path.exists():
        raise ValueError(f'missing session_manifest.json: {path}')
    return read_json(path)


def build_status(session_dir: Path, dataset_dir: Optional[Path] = None) -> Dict[str, object]:
    manifest = load_manifest(session_dir)
    selected_dataset_dir = dataset_dir or Path(str(manifest['dataset_dir']))
    metadata = list(iter_metadata(selected_dataset_dir))
    by_label: Dict[str, int] = {}
    for record in metadata:
        label = str(record.get('scene_label') or 'unlabeled')
        by_label[label] = by_label.get(label, 0) + 1

    task_status = []
    complete_count = 0
    for task in manifest.get('tasks', []):
        if not isinstance(task, dict):
            continue
        label = str(task.get('label') or '')
        target = int(task.get('target_count') or 0)
        actual = by_label.get(label, 0)
        complete = actual >= target
        if complete:
            complete_count += 1
        task_status.append(
            {
                'label': label,
                'target_count': target,
                'actual_count': actual,
                'complete': complete,
                'missing': max(0, target - actual),
            }
        )

    image_counts = count_images(selected_dataset_dir)
    return {
        'session_dir': str(session_dir),
        'dataset_dir': str(selected_dataset_dir),
        'image_counts': image_counts,
        'metadata_count': len(metadata),
        'scene_counts': by_label,
        'tasks': task_status,
        'complete_tasks': complete_count,
        'total_tasks': len(task_status),
        'all_tasks_complete': complete_count == len(task_status) and bool(task_status),
        'ready_for_review': sum(image_counts.values()) > 0,
    }


def status_session(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir)
    status = build_status(session_dir)
    if args.report_json:
        write_json(Path(args.report_json), status)
    write_json(session_dir / 'session_status.json', status)
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    else:
        print('FIELD_SESSION_STATUS', flush=True)
        print(f'  session_dir: {status["session_dir"]}', flush=True)
        print(f'  dataset_dir: {status["dataset_dir"]}', flush=True)
        print(f'  image_counts: {status["image_counts"]}', flush=True)
        print(f'  complete_tasks: {status["complete_tasks"]}/{status["total_tasks"]}', flush=True)
        for task in status['tasks']:
            print(
                f'  task {task["label"]}: {task["actual_count"]}/{task["target_count"]} '
                f'missing={task["missing"]}',
                flush=True,
            )
    return 0


def review_session(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir)
    manifest = load_manifest(session_dir)
    dataset_dir = Path(str(manifest['dataset_dir']))
    output_dir = Path(args.output_dir) if args.output_dir else session_dir / 'field_data_review'
    review_args = argparse.Namespace(
        dataset_dir=str(dataset_dir),
        input_file=None,
        input_dir=None,
        recursive=True,
        limit=args.limit,
        output_dir=str(output_dir),
        config=args.config,
        skip_audit=False,
        skip_replay=False,
        skip_vision_tune=args.skip_vision_tune,
        require_labels=args.require_labels,
        write_dataset_yaml=True,
        preview_limit=args.preview_limit,
        no_overlays=args.no_overlays,
        no_crops=args.no_crops,
        top_n=args.top_n,
        strict=args.strict,
        min_qr_rate=args.min_qr_rate,
        min_target_rate=args.min_target_rate,
        min_labeled_images=args.min_labeled_images,
        recommend_yolo_runtime=args.recommend_yolo_runtime,
        black_v_max=args.black_v_max,
        black_s_min=args.black_s_min,
        obstacle_roi_y_ratio=args.obstacle_roi_y_ratio,
        obstacle_min_area_ratio=args.obstacle_min_area_ratio,
        obstacle_danger_area_ratio=args.obstacle_danger_area_ratio,
        obstacle_center_band_ratio=args.obstacle_center_band_ratio,
        target_white_v_min=args.target_white_v_min,
        target_white_s_max=args.target_white_s_max,
        target_min_area_ratio=args.target_min_area_ratio,
        json=args.json,
    )
    return field_data_review.run(review_args)


def add_review_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir')
    parser.add_argument('--config')
    parser.add_argument('--skip-vision-tune', action='store_true')
    parser.add_argument('--require-labels', action='store_true')
    parser.add_argument('--preview-limit', type=int, default=20)
    parser.add_argument('--no-overlays', action='store_true')
    parser.add_argument('--no-crops', action='store_true')
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--strict', action='store_true')
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Manage OriginCar field capture sessions.')
    sub = parser.add_subparsers(dest='command', required=True)

    create = sub.add_parser('create', help='Create a field data collection session.')
    create.add_argument('--name', help='Session name. Default is field_<timestamp>.')
    create.add_argument('--sessions-root', default='/root/dev_ws/field_sessions')
    create.add_argument('--dataset-dir')
    create.add_argument('--topic', default='/image')
    create.add_argument('--interval', type=float, default=0.5)
    create.add_argument('--timeout', type=float, default=45.0)
    create.add_argument('--force', action='store_true')

    status = sub.add_parser('status', help='Show capture progress for a session.')
    status.add_argument('--session-dir', required=True)
    status.add_argument('--report-json')
    status.add_argument('--json', action='store_true')

    review = sub.add_parser('review', help='Run field_data_review for a session.')
    review.add_argument('--session-dir', required=True)
    add_review_args(review)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == 'create':
        return create_session(args)
    if args.command == 'status':
        return status_session(args)
    if args.command == 'review':
        return review_session(args)
    raise ValueError(f'unknown command: {args.command}')


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'FIELD_SESSION_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
