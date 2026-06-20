#!/usr/bin/env python3
"""Prepare and validate YOLO training/deployment artifacts for OriginCar."""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from origin_competition_auto import dataset_audit
except ImportError:
    import dataset_audit  # type: ignore


DEFAULT_WORKCONFIG = {
    'task_num': 4,
    'dnn_Parser': 'yolov8',
    'model_output_count': 6,
    'reg_max': 16,
    'strides': [8, 16, 32],
    'score_threshold': 0.25,
    'nms_threshold': 0.7,
    'nms_top_k': 300,
    'output_order': [0, 1, 2, 3, 4, 5],
}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def command_line(parts: Sequence[str]) -> str:
    return ' '.join(shell_quote(str(part)) for part in parts)


def read_classes(dataset_dir: Path) -> List[str]:
    classes_path = dataset_dir / 'classes.txt'
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


def read_classes_file(path: Path) -> List[str]:
    if not path.exists():
        raise ValueError(f'missing classes file: {path}')
    classes = [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f'classes file is empty: {path}')
    return classes


def make_workconfig(
    model_file: str,
    classes_file: str,
    class_num: int,
    score_threshold: float,
    nms_threshold: float,
) -> Dict[str, object]:
    data = dict(DEFAULT_WORKCONFIG)
    data.update(
        {
            'model_file': model_file,
            'class_num': class_num,
            'cls_names_list': classes_file,
            'score_threshold': score_threshold,
            'nms_threshold': nms_threshold,
        }
    )
    return data


def write_classes_list(classes: Sequence[str], path: Path) -> None:
    write_text(path, '\n'.join(classes) + '\n')


def train_script(args: argparse.Namespace, dataset_yaml: Path, run_dir: Path) -> str:
    cmd = [
        'yolo',
        'detect',
        'train',
        f'model={args.base_model}',
        f'data={dataset_yaml}',
        f'imgsz={args.imgsz}',
        f'epochs={args.epochs}',
        f'batch={args.batch}',
        f'project={run_dir.parent}',
        f'name={run_dir.name}',
        'exist_ok=True',
    ]
    if args.device:
        cmd.append(f'device={args.device}')
    return (
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n\n'
        '# Run this on a training machine with ultralytics installed.\n'
        f'{command_line(cmd)}\n'
    )


def export_script(args: argparse.Namespace, pt_path: Path, export_dir: Path) -> str:
    cmd = [
        'yolo',
        'export',
        f'model={pt_path}',
        'format=onnx',
        f'imgsz={args.imgsz}',
        'opset=12',
        'simplify=True',
    ]
    return (
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n\n'
        '# Run this after training, on a machine with ultralytics installed.\n'
        f'mkdir -p {shell_quote(str(export_dir))}\n'
        f'{command_line(cmd)}\n'
        f'echo "Move/export ONNX artifacts into: {export_dir}"\n'
    )


def convert_notes(args: argparse.Namespace, onnx_path: Path, rdk_dir: Path) -> str:
    return (
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n\n'
        '# Placeholder for RDK X5 model conversion.\n'
        '# Use the Horizon/RDK model conversion toolchain that matches your board BSP.\n'
        '# Expected input ONNX:\n'
        f'#   {onnx_path}\n'
        '# Expected output BPU model:\n'
        f'#   {rdk_dir / (args.model_name + "_" + str(args.imgsz) + "x" + str(args.imgsz) + "_nv12.bin")}\n'
        '# After conversion, rerun yolo_pipeline validate without --allow-missing-model.\n'
        'echo "Install/use the official RDK conversion toolchain, then create the .bin above."\n'
    )


def deploy_script(workconfig_path: Path) -> str:
    return (
        '#!/usr/bin/env bash\n'
        'set -euo pipefail\n'
        'source /opt/ros/humble/setup.bash\n'
        'source /opt/tros/humble/setup.bash 2>/dev/null || true\n'
        'source /root/dev_ws/install/setup.bash\n\n'
        'ros2 launch origin_competition_auto rdk_yolo_detection.launch.py '
        f'config_file:={shell_quote(str(workconfig_path))}\n'
    )


def chmod_executable(path: Path) -> None:
    try:
        path.chmod(0o755)
    except OSError:
        pass


def prepare(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = read_classes(dataset_dir)
    audit_report = dataset_audit.audit_dataset(dataset_dir, require_labels=args.require_labels)
    dataset_yaml = output_dir / 'dataset.yaml'
    dataset_audit.write_yolo_yaml(audit_report, dataset_yaml)
    write_json(output_dir / 'dataset_audit_report.json', audit_report.to_dict())

    classes_file = Path(args.classes_file) if args.classes_file else output_dir / 'classes.list'
    write_classes_list(classes, classes_file)

    training_run_dir = output_dir / 'training_runs' / args.model_name
    pt_path = training_run_dir / 'weights' / 'best.pt'
    export_dir = output_dir / 'exports'
    onnx_path = export_dir / f'{args.model_name}.onnx'
    rdk_dir = output_dir / 'rdk'
    model_file = (
        args.model_file
        if args.model_file
        else str(rdk_dir / f'{args.model_name}_{args.imgsz}x{args.imgsz}_nv12.bin')
    )
    workconfig_path = (
        Path(args.workconfig_output)
        if args.workconfig_output
        else output_dir / f'{args.model_name}_workconfig.json'
    )
    workconfig = make_workconfig(
        model_file=model_file,
        classes_file=str(classes_file),
        class_num=len(classes),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    write_json(workconfig_path, workconfig)

    scripts_dir = output_dir / 'scripts'
    train_path = scripts_dir / 'train_ultralytics.sh'
    export_path = scripts_dir / 'export_onnx.sh'
    convert_path = scripts_dir / 'convert_rdk_placeholder.sh'
    deploy_path = scripts_dir / 'launch_custom_yolo.sh'
    write_text(train_path, train_script(args, dataset_yaml, training_run_dir))
    write_text(export_path, export_script(args, pt_path, export_dir))
    write_text(convert_path, convert_notes(args, onnx_path, rdk_dir))
    write_text(deploy_path, deploy_script(workconfig_path))
    for path in (train_path, export_path, convert_path, deploy_path):
        chmod_executable(path)

    manifest = {
        'model_name': args.model_name,
        'dataset_dir': str(dataset_dir),
        'output_dir': str(output_dir),
        'classes': classes,
        'class_num': len(classes),
        'dataset_yaml': str(dataset_yaml),
        'classes_file': str(classes_file),
        'workconfig': str(workconfig_path),
        'expected_artifacts': {
            'pt': str(pt_path),
            'onnx': str(onnx_path),
            'rdk_bin': model_file,
        },
        'scripts': {
            'train': str(train_path),
            'export_onnx': str(export_path),
            'convert_rdk_placeholder': str(convert_path),
            'launch_custom_yolo': str(deploy_path),
        },
        'audit_ok': audit_report.ok,
        'audit_warnings': len(audit_report.warnings),
        'audit_errors': len(audit_report.errors),
        'notes': [
            'Training and RDK conversion are not run by prepare.',
            'Use dataset_audit_report.json before training.',
            'Run validate after placing the converted .bin model.',
        ],
    }
    write_json(output_dir / 'model_manifest.json', manifest)
    write_readme(output_dir, manifest)

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    else:
        print('YOLO_PIPELINE_PREPARED', flush=True)
        print(f'  output_dir: {output_dir}', flush=True)
        print(f'  dataset_yaml: {dataset_yaml}', flush=True)
        print(f'  classes_file: {classes_file}', flush=True)
        print(f'  workconfig: {workconfig_path}', flush=True)
        print(f'  train_script: {train_path}', flush=True)
        print(f'  manifest: {output_dir / "model_manifest.json"}', flush=True)
    return 0 if audit_report.ok else 1


def write_readme(output_dir: Path, manifest: Dict[str, object]) -> None:
    lines = [
        '# OriginCar YOLO Model Artifacts',
        '',
        f'model: {manifest["model_name"]}',
        f'dataset: {manifest["dataset_dir"]}',
        '',
        '## Order',
        '',
        '1. Review `dataset_audit_report.json`.',
        '2. Run `scripts/train_ultralytics.sh` on a training machine.',
        '3. Run `scripts/export_onnx.sh` after training.',
        '4. Convert ONNX to an RDK X5 `.bin` with the official Horizon/RDK toolchain.',
        '5. Put the `.bin` at the path in `model_manifest.json`.',
        '6. Run `yolo_pipeline validate`.',
        '7. Launch with `scripts/launch_custom_yolo.sh`.',
        '',
        '## Expected Artifacts',
        '',
    ]
    expected = manifest.get('expected_artifacts', {})
    if isinstance(expected, dict):
        for name, path in expected.items():
            lines.append(f'- {name}: `{path}`')
    lines.extend(
        [
            '',
            '## Runtime',
            '',
            'The custom model must keep publishing detections on `/hobot_dnn_detection`.',
            '`auto_mission` does not need code changes when only the model/workconfig changes.',
            '',
        ]
    )
    write_text(output_dir / 'README.md', '\n'.join(lines))


def validate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError(f'expected object manifest: {manifest_path}')
    checks = []

    def add_check(name: str, path: str, required: bool = True) -> None:
        exists = Path(path).exists()
        checks.append(
            {
                'name': name,
                'path': path,
                'exists': exists,
                'required': required,
                'ok': exists or not required,
            }
        )

    add_check('dataset_yaml', str(manifest.get('dataset_yaml', '')))
    add_check('classes_file', str(manifest.get('classes_file', '')))
    add_check('workconfig', str(manifest.get('workconfig', '')))
    expected = manifest.get('expected_artifacts', {})
    if isinstance(expected, dict):
        add_check('trained_pt', str(expected.get('pt', '')), required=not args.allow_missing_training)
        add_check('onnx', str(expected.get('onnx', '')), required=not args.allow_missing_training)
        add_check('rdk_bin', str(expected.get('rdk_bin', '')), required=not args.allow_missing_model)

    report = {
        'manifest': str(manifest_path),
        'ok': all(check['ok'] for check in checks),
        'checks': checks,
    }
    if args.report_json:
        write_json(Path(args.report_json), report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        print('YOLO_PIPELINE_VALIDATE', flush=True)
        print(f'  ok: {report["ok"]}', flush=True)
        for check in checks:
            status = 'PASS' if check['ok'] else 'FAIL'
            print(f'  {status} {check["name"]}: {check["path"]}', flush=True)
    return 0 if report['ok'] else 1


def resolve_manifest_path(manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def relative_to_root(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def relative_to_raw(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def manifest_artifact_path(
    manifest_path: Path,
    current_output_dir: Path,
    old_output_raw: Path,
    old_output_dir: Path,
    value: object,
    fallback: Path,
) -> Path:
    if value:
        path = Path(str(value))
        if path.is_absolute():
            rel = relative_to_root(path, old_output_dir)
            if rel is not None:
                return current_output_dir / rel
            return path
        raw_rel = relative_to_raw(path, old_output_raw)
        if raw_rel is not None:
            return current_output_dir / raw_rel
        return manifest_path.parent / path
    return fallback


def copy_artifact(
    source: Optional[str],
    destination: Path,
    copy_enabled: bool,
) -> Dict[str, object]:
    if not source:
        return {
            'source': '',
            'target': str(destination),
            'provided': False,
            'copied': False,
            'exists': destination.exists(),
        }
    src = Path(source)
    if not src.exists() or not src.is_file():
        raise ValueError(f'artifact does not exist: {src}')
    target = destination if copy_enabled else src
    if copy_enabled:
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
    return {
        'source': str(src),
        'target': str(target),
        'provided': True,
        'copied': bool(copy_enabled),
        'exists': target.exists(),
    }


def promote(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError(f'expected object manifest: {manifest_path}')

    old_output_raw = Path(str(manifest.get('output_dir') or manifest_path.parent))
    old_output_dir = old_output_raw if old_output_raw.is_absolute() else manifest_path.parent / old_output_raw
    output_dir = manifest_path.parent
    expected = manifest.get('expected_artifacts', {})
    if not isinstance(expected, dict):
        expected = {}

    model_name = str(manifest.get('model_name', 'origin_yolo'))
    pt_dest = manifest_artifact_path(
        manifest_path,
        output_dir,
        old_output_raw,
        old_output_dir,
        expected.get('pt'),
        output_dir / 'training_runs' / model_name / 'weights' / 'best.pt',
    )
    onnx_dest = manifest_artifact_path(
        manifest_path,
        output_dir,
        old_output_raw,
        old_output_dir,
        expected.get('onnx'),
        output_dir / 'exports' / f'{model_name}.onnx',
    )
    rdk_dest = manifest_artifact_path(
        manifest_path,
        output_dir,
        old_output_raw,
        old_output_dir,
        expected.get('rdk_bin'),
        output_dir / 'rdk' / f'{model_name}_640x640_nv12.bin',
    )
    classes_dest = manifest_artifact_path(
        manifest_path,
        output_dir,
        old_output_raw,
        old_output_dir,
        manifest.get('classes_file'),
        output_dir / 'classes.list',
    )
    workconfig_path = manifest_artifact_path(
        manifest_path,
        output_dir,
        old_output_raw,
        old_output_dir,
        manifest.get('workconfig'),
        output_dir / f'{model_name}_workconfig.json',
    )

    artifact_report = {
        'pt': copy_artifact(args.pt, pt_dest, args.copy),
        'onnx': copy_artifact(args.onnx, onnx_dest, args.copy),
        'rdk_bin': copy_artifact(args.rdk_bin, rdk_dest, args.copy),
    }

    classes_source = args.classes_file
    if classes_source:
        classes_report = copy_artifact(classes_source, classes_dest, args.copy)
    else:
        classes_report = {
            'source': '',
            'target': str(classes_dest),
            'provided': False,
            'copied': False,
            'exists': classes_dest.exists(),
        }
    classes = read_classes_file(Path(str(classes_report['target'])))
    model_file = str(artifact_report['rdk_bin']['target'])
    if not Path(model_file).exists():
        raise ValueError(
            'RDK .bin is still missing after promote; pass --rdk-bin or place it at '
            f'{model_file}'
        )

    workconfig = make_workconfig(
        model_file=model_file,
        classes_file=str(classes_report['target']),
        class_num=len(classes),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    write_json(workconfig_path, workconfig)
    deploy_path = output_dir / 'scripts' / 'launch_custom_yolo.sh'
    write_text(deploy_path, deploy_script(workconfig_path))
    chmod_executable(deploy_path)

    expected['pt'] = str(artifact_report['pt']['target'])
    expected['onnx'] = str(artifact_report['onnx']['target'])
    expected['rdk_bin'] = str(artifact_report['rdk_bin']['target'])
    manifest['expected_artifacts'] = expected
    manifest['classes_file'] = str(classes_report['target'])
    manifest['classes'] = classes
    manifest['class_num'] = len(classes)
    manifest['output_dir'] = str(output_dir)
    manifest['workconfig'] = str(workconfig_path)
    if manifest.get('dataset_yaml'):
        manifest['dataset_yaml'] = str(
            manifest_artifact_path(
                manifest_path,
                output_dir,
                old_output_raw,
                old_output_dir,
                manifest.get('dataset_yaml'),
                output_dir / 'dataset.yaml',
            )
        )
    scripts = manifest.get('scripts', {})
    if not isinstance(scripts, dict):
        scripts = {}
    scripts['launch_custom_yolo'] = str(deploy_path)
    manifest['scripts'] = scripts
    manifest['promoted_at'] = now_iso()
    manifest['promotion'] = {
        'copy': bool(args.copy),
        'artifacts': artifact_report,
        'classes_file': classes_report,
        'score_threshold': args.score_threshold,
        'nms_threshold': args.nms_threshold,
        'launch_script': str(deploy_path),
    }
    write_json(manifest_path, manifest)
    write_readme(output_dir, manifest)

    report = {
        'ok': True,
        'manifest': str(manifest_path),
        'workconfig': str(workconfig_path),
        'model_file': model_file,
        'classes_file': str(classes_report['target']),
        'class_num': len(classes),
        'launch_script': str(deploy_path),
        'promotion': manifest['promotion'],
    }
    if args.report_json:
        write_json(Path(args.report_json), report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    else:
        print('YOLO_PIPELINE_PROMOTED', flush=True)
        print(f'  manifest: {manifest_path}', flush=True)
        print(f'  workconfig: {workconfig_path}', flush=True)
        print(f'  model_file: {model_file}', flush=True)
        print(f'  classes_file: {classes_report["target"]}', flush=True)
        print(f'  launch_script: {deploy_path}', flush=True)
    return 0


def workconfig(args: argparse.Namespace) -> int:
    classes = read_classes_file(Path(args.classes_file))
    data = make_workconfig(
        model_file=args.model_file,
        classes_file=args.classes_file,
        class_num=len(classes),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
    )
    write_json(Path(args.output), data)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
    else:
        print('YOLO_WORKCONFIG_WRITTEN', flush=True)
        print(f'  output: {args.output}', flush=True)
        print(f'  model_file: {args.model_file}', flush=True)
        print(f'  classes: {len(classes)}', flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prepare and validate OriginCar YOLO training/deployment artifacts.'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--dataset-dir', required=True)
    prepare_parser.add_argument('--output-dir', default='/root/dev_ws/models/origin_yolo')
    prepare_parser.add_argument('--model-name', default='origin_yolov8n')
    prepare_parser.add_argument('--base-model', default='yolov8n.pt')
    prepare_parser.add_argument('--imgsz', type=int, default=640)
    prepare_parser.add_argument('--epochs', type=int, default=100)
    prepare_parser.add_argument('--batch', type=int, default=16)
    prepare_parser.add_argument('--device', default='')
    prepare_parser.add_argument('--require-labels', action='store_true')
    prepare_parser.add_argument('--classes-file')
    prepare_parser.add_argument('--model-file')
    prepare_parser.add_argument('--workconfig-output')
    prepare_parser.add_argument('--score-threshold', type=float, default=0.25)
    prepare_parser.add_argument('--nms-threshold', type=float, default=0.7)
    prepare_parser.add_argument('--json', action='store_true')

    validate_parser = sub.add_parser('validate')
    validate_parser.add_argument('--manifest', required=True)
    validate_parser.add_argument('--allow-missing-training', action='store_true')
    validate_parser.add_argument('--allow-missing-model', action='store_true')
    validate_parser.add_argument('--report-json')
    validate_parser.add_argument('--json', action='store_true')

    promote_parser = sub.add_parser('promote')
    promote_parser.add_argument('--manifest', required=True)
    promote_parser.add_argument('--pt',
                                help='Trained best.pt artifact. Optional if already in the manifest path.')
    promote_parser.add_argument('--onnx',
                                help='Exported ONNX artifact. Optional if already in the manifest path.')
    promote_parser.add_argument('--rdk-bin', required=True,
                                help='Converted RDK X5 BPU .bin model.')
    promote_parser.add_argument('--classes-file',
                                help='Class names file. Optional if manifest classes_file already exists.')
    promote_parser.add_argument('--copy', action='store_true',
                                help='Copy supplied artifacts into the expected manifest locations.')
    promote_parser.add_argument('--score-threshold', type=float, default=0.25)
    promote_parser.add_argument('--nms-threshold', type=float, default=0.7)
    promote_parser.add_argument('--report-json')
    promote_parser.add_argument('--json', action='store_true')

    workconfig_parser = sub.add_parser('workconfig')
    workconfig_parser.add_argument('--model-file', required=True)
    workconfig_parser.add_argument('--classes-file', required=True)
    workconfig_parser.add_argument('--output', required=True)
    workconfig_parser.add_argument('--score-threshold', type=float, default=0.25)
    workconfig_parser.add_argument('--nms-threshold', type=float, default=0.7)
    workconfig_parser.add_argument('--json', action='store_true')
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command == 'prepare':
        return prepare(args)
    if args.command == 'validate':
        return validate(args)
    if args.command == 'promote':
        return promote(args)
    if args.command == 'workconfig':
        return workconfig(args)
    raise ValueError(f'unknown command: {args.command}')


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f'YOLO_PIPELINE_ERROR: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
