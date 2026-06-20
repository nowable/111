#!/usr/bin/env python3
"""Unified run wrapper for pre-field OriginCar competition tests."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PACKAGE_NAME = 'origin_competition_auto'

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


DEFAULT_PROFILES: Dict[str, Dict[str, object]] = {
    'safe_static': {
        'description': 'camera, QR, vision, and LLM only; never allow motion',
        'motion_capable': False,
        'require_yolo': False,
        'llm_mode': 'placeholder',
        'max_runtime': 45.0,
        'drive_to_qr_timeout': 0.8,
        'scan_qr_timeout': 8.0,
        'loop_duration': 0.0,
        'return_duration': 0.0,
        'cruise_linear': 0.03,
    },
    'bench_motion': {
        'description': 'rack-mounted low-speed smoke test',
        'motion_capable': True,
        'require_yolo': False,
        'llm_mode': 'placeholder',
        'max_runtime': 45.0,
        'drive_to_qr_timeout': 1.5,
        'scan_qr_timeout': 8.0,
        'loop_duration': 0.6,
        'return_duration': 0.5,
        'cruise_linear': 0.03,
    },
    'field_low_speed': {
        'description': 'conservative ground test after static and bench checks',
        'motion_capable': True,
        'require_yolo': False,
        'llm_mode': 'placeholder',
        'max_runtime': 90.0,
        'drive_to_qr_timeout': 2.0,
        'scan_qr_timeout': 10.0,
        'loop_duration': 1.2,
        'return_duration': 1.0,
        'cruise_linear': 0.03,
    },
}

PROFILE_REQUIRED_KEYS = {
    'description',
    'motion_capable',
    'require_yolo',
    'llm_mode',
    'max_runtime',
    'drive_to_qr_timeout',
    'scan_qr_timeout',
    'loop_duration',
    'return_duration',
    'cruise_linear',
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def command_text(cmd: Iterable[str]) -> str:
    return ' '.join(json.dumps(part, ensure_ascii=False) for part in cmd)


def default_profiles_config_path() -> Optional[Path]:
    candidates = []
    if get_package_share_directory is not None:
        try:
            candidates.append(
                Path(get_package_share_directory(PACKAGE_NAME))
                / 'config'
                / 'run_profiles.json'
            )
        except Exception:
            pass
    candidates.append(Path(__file__).resolve().parents[1] / 'config' / 'run_profiles.json')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_profiles(profiles: Dict[str, Dict[str, object]]) -> None:
    if not profiles:
        raise ValueError('run profile config contains no profiles')
    for name, profile in profiles.items():
        missing = sorted(PROFILE_REQUIRED_KEYS - set(profile))
        if missing:
            raise ValueError(f'profile {name} missing keys: {", ".join(missing)}')
        if not isinstance(profile['motion_capable'], bool):
            raise ValueError(f'profile {name} motion_capable must be boolean')
        if not isinstance(profile['require_yolo'], bool):
            raise ValueError(f'profile {name} require_yolo must be boolean')
        if str(profile['llm_mode']) not in ('placeholder', 'disabled', 'openai-compatible'):
            raise ValueError(
                f'profile {name} llm_mode must be placeholder, disabled, or openai-compatible'
            )
        for key in (
            'max_runtime',
            'drive_to_qr_timeout',
            'scan_qr_timeout',
            'loop_duration',
            'return_duration',
            'cruise_linear',
        ):
            try:
                value = float(profile[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'profile {name} {key} must be numeric') from exc
            if value < 0.0:
                raise ValueError(f'profile {name} {key} must be >= 0')
        if float(profile['cruise_linear']) > 0.08:
            raise ValueError(f'profile {name} cruise_linear must be <= 0.08')


def load_profiles(path: Optional[str]) -> Tuple[Dict[str, Dict[str, object]], str]:
    selected = Path(path) if path else default_profiles_config_path()
    if selected is None:
        profiles = json.loads(json.dumps(DEFAULT_PROFILES))
        validate_profiles(profiles)
        return profiles, 'builtin'
    with selected.open('r', encoding='utf-8') as f:
        data = json.load(f)
    raw_profiles = data.get('profiles') if isinstance(data, dict) else None
    if raw_profiles is None and isinstance(data, dict):
        raw_profiles = data
    if not isinstance(raw_profiles, dict):
        raise ValueError('run profile config must contain a profiles object')
    profiles: Dict[str, Dict[str, object]] = {}
    for name, profile in raw_profiles.items():
        if not isinstance(name, str) or not isinstance(profile, dict):
            raise ValueError('profile names must map to objects')
        profiles[name] = dict(profile)
    validate_profiles(profiles)
    return profiles, str(selected)


def run_capture(
    cmd: List[str],
    timeout: float,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    started = now_iso()
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        result = {
            'command': cmd,
            'started_at': started,
            'finished_at': now_iso(),
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'timed_out': False,
        }
    except subprocess.TimeoutExpired as exc:
        result = {
            'command': cmd,
            'started_at': started,
            'finished_at': now_iso(),
            'returncode': 124,
            'stdout': exc.stdout or '',
            'stderr': exc.stderr or '',
            'timed_out': True,
        }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            (
                f'$ {command_text(cmd)}\n'
                f'# started_at: {result["started_at"]}\n'
                f'# finished_at: {result["finished_at"]}\n'
                f'# returncode: {result["returncode"]}\n\n'
                f'{result["stdout"]}\n'
                f'{result["stderr"]}'
            ),
            encoding='utf-8',
        )
    return result


def parse_json_stdout(result: Dict[str, object]) -> Optional[object]:
    stdout = str(result.get('stdout') or '').strip()
    if not stdout:
        return None
    decoder = json.JSONDecoder()
    try:
        return decoder.raw_decode(stdout)[0]
    except json.JSONDecodeError:
        first_brace = stdout.find('{')
        if first_brace < 0:
            return None
        try:
            return decoder.raw_decode(stdout[first_brace:])[0]
        except json.JSONDecodeError:
            return None


def system_check_command(require_yolo: bool) -> List[str]:
    cmd = ['ros2', 'run', PACKAGE_NAME, 'system_check', '--json']
    if require_yolo:
        cmd.append('--require-yolo')
    return cmd


def run_system_check(
    run_dir: Path,
    name: str,
    require_yolo: bool,
    timeout: float,
) -> Dict[str, object]:
    cmd = system_check_command(require_yolo)
    result = run_capture(cmd, timeout, output_path=run_dir / f'{name}.log')
    parsed = parse_json_stdout(result)
    record = {
        'command': cmd,
        'require_yolo': require_yolo,
        'returncode': result['returncode'],
        'timed_out': result['timed_out'],
        'parsed': parsed,
        'stdout': result['stdout'] if parsed is None else '',
        'stderr': result['stderr'],
    }
    write_json(run_dir / f'{name}.json', record)
    return record


def system_check_ok(record: Dict[str, object]) -> bool:
    parsed = record.get('parsed')
    if isinstance(parsed, dict) and isinstance(parsed.get('ok'), bool):
        return bool(parsed['ok']) and int(record.get('returncode', 1)) == 0
    return int(record.get('returncode', 1)) == 0


def summarize_failed_checks(record: Dict[str, object]) -> str:
    parsed = record.get('parsed')
    if isinstance(parsed, dict):
        checks = parsed.get('checks')
        if isinstance(checks, list):
            failed = []
            for check in checks:
                if isinstance(check, dict) and not check.get('ok', False):
                    failed.append(f'{check.get("name")}: {check.get("detail")}')
            if failed:
                return '; '.join(failed)
    stderr = str(record.get('stderr') or '').strip()
    stdout = str(record.get('stdout') or '').strip()
    return stderr or stdout or 'system_check failed'


def popen_stream(cmd: List[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if os.name != 'nt':
        kwargs['start_new_session'] = True
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        **kwargs,
    )


def stop_process(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        handle = getattr(proc, '_competition_log_handle', None) if proc is not None else None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        return
    try:
        if os.name != 'nt':
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            if os.name != 'nt':
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
    handle = getattr(proc, '_competition_log_handle', None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def start_yolo(run_dir: Path) -> subprocess.Popen:
    cmd = ['ros2', 'launch', PACKAGE_NAME, 'rdk_yolo_detection.launch.py']
    log_path = run_dir / 'yolo_output.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('w', encoding='utf-8')
    log.write(f'$ {command_text(cmd)}\n# started_at: {now_iso()}\n\n')
    log.flush()
    kwargs = {}
    if os.name != 'nt':
        kwargs['start_new_session'] = True
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        **kwargs,
    )
    setattr(proc, '_competition_log_handle', log)
    return proc


def wait_for_yolo(
    run_dir: Path,
    proc: subprocess.Popen,
    timeout: float,
    system_check_timeout: float,
) -> Tuple[bool, str, Optional[Dict[str, object]]]:
    deadline = time.monotonic() + timeout
    last_record: Optional[Dict[str, object]] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False, f'YOLO launch exited early with code {proc.returncode}', last_record
        record = run_system_check(
            run_dir,
            'system_check_yolo',
            require_yolo=True,
            timeout=system_check_timeout,
        )
        last_record = record
        if system_check_ok(record):
            return True, '', record
        time.sleep(1.0)
    reason = 'YOLO topic did not become ready before timeout'
    if last_record is not None:
        reason = summarize_failed_checks(last_record)
    return False, reason, last_record


def stream_command(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    proc = popen_stream(cmd, log_path)
    with log_path.open('w', encoding='utf-8') as log:
        log.write(f'$ {command_text(cmd)}\n# started_at: {started}\n\n')
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        returncode = proc.wait()
        log.write(f'\n# finished_at: {now_iso()}\n# returncode: {returncode}\n')
    return int(returncode)


def parse_mission_output(log_path: Path) -> Dict[str, object]:
    if not log_path.exists():
        return {}
    states: List[str] = []
    saved_images: List[str] = []
    events: List[Dict[str, object]] = []
    event_counts: Dict[str, int] = {}
    mission_summary: Dict[str, str] = {}
    in_summary = False

    for raw_line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = raw_line.rstrip()
        if line.startswith('STATE: '):
            states.append(line.split(': ', 1)[1].strip())
            in_summary = False
            continue
        if line.startswith('SAVED_IMAGE: '):
            saved_images.append(line.split(': ', 1)[1].strip())
            continue
        if line == 'MISSION_SUMMARY':
            in_summary = True
            continue
        if in_summary and line.startswith('  ') and ':' in line:
            key, value = line.strip().split(':', 1)
            mission_summary[key.strip()] = value.strip()
            continue
        if line.startswith('EVENT '):
            payload = line[6:].strip()
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
                event_name = str(event.get('event') or 'unknown')
                event_counts[event_name] = event_counts.get(event_name, 0) + 1
            continue
        if line and not line.startswith('  '):
            in_summary = False

    summary_event = next(
        (
            event
            for event in reversed(events)
            if event.get('event') == 'mission_summary'
        ),
        {},
    )
    return {
        'states': states,
        'state_count': len(states),
        'final_state': states[-1] if states else '',
        'saved_images': saved_images,
        'event_counts': event_counts,
        'event_count': len(events),
        'last_event': events[-1] if events else {},
        'mission_summary': mission_summary,
        'mission_summary_event': summary_event,
        'qr_content': mission_summary.get('qr_content', ''),
        'qr_method': mission_summary.get('qr_method', ''),
        'direction': mission_summary.get('direction', ''),
        'direction_source': mission_summary.get('direction_source', ''),
        'target_image': mission_summary.get('target_image', ''),
        'llm_mode': mission_summary.get('llm_mode', ''),
        'llm_result': mission_summary.get('llm_result', ''),
        'mission_failure_reason': mission_summary.get('failure_reason', ''),
    }


def profile_value(args: argparse.Namespace, profile: Dict[str, object], key: str) -> object:
    value = getattr(args, key, None)
    if value is not None:
        return value
    return profile[key]


def build_auto_mission_command(
    args: argparse.Namespace,
    profile: Dict[str, object],
    run_dir: Path,
    motion_enabled: bool,
) -> List[str]:
    debug_dir = Path(args.debug_dir) if args.debug_dir else run_dir / 'debug_images'
    llm_mode = args.llm_mode or str(profile['llm_mode'])
    cmd = ['ros2', 'run', PACKAGE_NAME, 'auto_mission']
    if args.config:
        cmd += ['--config', args.config]
    if not motion_enabled:
        cmd.append('--no-motion')
    cmd += [
        '--start-state', args.start_state,
        '--debug-dir', str(debug_dir),
        '--max-runtime', str(profile_value(args, profile, 'max_runtime')),
        '--drive-to-qr-timeout', str(profile_value(args, profile, 'drive_to_qr_timeout')),
        '--scan-qr-timeout', str(profile_value(args, profile, 'scan_qr_timeout')),
        '--cruise-linear', str(profile_value(args, profile, 'cruise_linear')),
        '--loop-duration', str(profile_value(args, profile, 'loop_duration')),
        '--return-duration', str(profile_value(args, profile, 'return_duration')),
        '--llm-mode', llm_mode,
    ]
    if args.save_debug_images:
        cmd.append('--save-debug-images')
    if args.mock_qr_content is not None:
        cmd += ['--mock-qr-content', args.mock_qr_content]
    if args.force_direction is not None:
        cmd += ['--force-direction', args.force_direction]
    if args.llm_api_url:
        cmd += ['--llm-api-url', args.llm_api_url]
    if args.llm_model:
        cmd += ['--llm-model', args.llm_model]
    if args.llm_api_key_env:
        cmd += ['--llm-api-key-env', args.llm_api_key_env]
    if args.llm_timeout is not None:
        cmd += ['--llm-timeout', str(args.llm_timeout)]
    if args.llm_prompt:
        cmd += ['--llm-prompt', args.llm_prompt]
    return cmd


def build_replay_command(run_dir: Path, debug_dir: Path) -> Optional[List[str]]:
    if not debug_dir.exists():
        return None
    image_paths = []
    for pattern in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(debug_dir.rglob(pattern))
    if not image_paths:
        return None
    return [
        'ros2',
        'run',
        PACKAGE_NAME,
        'mission_replay',
        '--input-dir',
        str(debug_dir),
        '--recursive',
        '--limit',
        '30',
        '--output-dir',
        str(run_dir / 'mission_replay'),
        '--save-overlays',
        '--skip-llm',
    ]


def write_command_file(
    run_dir: Path,
    args: argparse.Namespace,
    commands: Dict[str, List[str]],
    motion_enabled: bool,
    profiles_config_path: str,
) -> None:
    lines = [
        f'argv: {command_text(sys.argv)}',
        f'profile: {args.profile}',
        f'profiles_config: {profiles_config_path}',
        f'allow_motion_arg: {args.allow_motion}',
        f'motion_enabled: {motion_enabled}',
        '',
    ]
    for name, cmd in commands.items():
        lines.append(f'{name}: {command_text(cmd)}')
    (run_dir / 'command.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def print_dry_run(
    args: argparse.Namespace,
    profile: Dict[str, object],
    motion_enabled: bool,
    profiles_config_path: str,
) -> int:
    fake_run_dir = Path(args.runs_dir) / f'{timestamp()}_{args.profile}'
    auto_cmd = build_auto_mission_command(args, profile, fake_run_dir, motion_enabled)
    print('competition_run dry-run')
    print(f'  profile: {args.profile}')
    print(f'  profiles_config: {profiles_config_path}')
    print(f'  profile_description: {profile["description"]}')
    print(f'  run_dir: {fake_run_dir}')
    print(f'  motion_enabled: {motion_enabled}')
    print(f'  system_check: {command_text(system_check_command(False))}')
    if args.start_yolo:
        print(f'  yolo_launch: {command_text(["ros2", "launch", PACKAGE_NAME, "rdk_yolo_detection.launch.py"])}')
    if args.require_yolo or args.start_yolo or bool(profile.get('require_yolo')):
        print(f'  yolo_check: {command_text(system_check_command(True))}')
    print(f'  auto_mission: {command_text(auto_cmd)}')
    print('dry-run: no directories, ROS subscriptions, launch processes, or /cmd_vel messages created')
    return 0


def run(args: argparse.Namespace) -> int:
    profiles, profiles_config_path = load_profiles(args.profiles_config)
    if args.profile not in profiles:
        available = ', '.join(sorted(profiles))
        raise ValueError(f'unknown profile {args.profile}; available: {available}')
    profile = profiles[args.profile]
    if args.allow_motion and not bool(profile['motion_capable']):
        raise ValueError(
            f'{args.profile} does not allow motion; choose a motion-capable profile'
        )
    motion_enabled = bool(args.allow_motion and profile['motion_capable'])
    require_yolo = bool(args.require_yolo or profile.get('require_yolo') or args.start_yolo)

    if args.dry_run:
        return print_dry_run(args, profile, motion_enabled, profiles_config_path)

    run_dir = Path(args.runs_dir) / f'{timestamp()}_{args.profile}'
    run_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = Path(args.debug_dir) if args.debug_dir else run_dir / 'debug_images'
    auto_cmd = build_auto_mission_command(args, profile, run_dir, motion_enabled)
    commands = {
        'system_check': system_check_command(False),
        'auto_mission': auto_cmd,
    }
    if args.start_yolo:
        commands['yolo_launch'] = ['ros2', 'launch', PACKAGE_NAME, 'rdk_yolo_detection.launch.py']
    if require_yolo:
        commands['system_check_yolo'] = system_check_command(True)
    write_command_file(run_dir, args, commands, motion_enabled, profiles_config_path)

    summary: Dict[str, object] = {
        'ok': False,
        'status': 'running',
        'failure_reason': '',
        'started_at': now_iso(),
        'finished_at': '',
        'profile': args.profile,
        'profiles_config': profiles_config_path,
        'profile_description': profile['description'],
        'effective_profile': profile,
        'run_dir': str(run_dir),
        'debug_dir': str(debug_dir),
        'allow_motion_arg': bool(args.allow_motion),
        'motion_enabled': motion_enabled,
        'require_yolo': require_yolo,
        'start_yolo': bool(args.start_yolo),
        'llm_mode': args.llm_mode or profile['llm_mode'],
        'commands': commands,
        'files': {
            'command': str(run_dir / 'command.txt'),
            'system_check': str(run_dir / 'system_check.json'),
            'mission_output': str(run_dir / 'mission_output.log'),
            'run_summary': str(run_dir / 'run_summary.json'),
        },
    }

    yolo_proc: Optional[subprocess.Popen] = None
    try:
        print(f'RUN_DIR: {run_dir}', flush=True)
        print('STEP: system_check', flush=True)
        check = run_system_check(
            run_dir,
            'system_check',
            require_yolo=False,
            timeout=args.system_check_timeout,
        )
        summary['system_check_returncode'] = check['returncode']
        if not system_check_ok(check):
            summary['status'] = 'failed'
            summary['failure_reason'] = summarize_failed_checks(check)
            return 1

        if args.start_yolo:
            print('STEP: start_yolo', flush=True)
            yolo_proc = start_yolo(run_dir)
            yolo_ok, reason, _record = wait_for_yolo(
                run_dir,
                yolo_proc,
                timeout=args.yolo_startup_timeout,
                system_check_timeout=args.system_check_timeout,
            )
            if not yolo_ok:
                summary['status'] = 'failed'
                summary['failure_reason'] = reason
                return 1
        elif require_yolo:
            print('STEP: require_yolo_check', flush=True)
            yolo_check = run_system_check(
                run_dir,
                'system_check_yolo',
                require_yolo=True,
                timeout=args.system_check_timeout,
            )
            summary['yolo_check_returncode'] = yolo_check['returncode']
            if not system_check_ok(yolo_check):
                summary['status'] = 'failed'
                summary['failure_reason'] = summarize_failed_checks(yolo_check)
                return 1

        print('STEP: auto_mission', flush=True)
        mission_log_path = run_dir / 'mission_output.log'
        mission_returncode = stream_command(auto_cmd, mission_log_path)
        summary['mission_returncode'] = mission_returncode
        summary['mission_observations'] = parse_mission_output(mission_log_path)

        replay_cmd = build_replay_command(run_dir, debug_dir)
        if replay_cmd is not None:
            print('STEP: mission_replay', flush=True)
            summary['commands']['mission_replay'] = replay_cmd  # type: ignore[index]
            replay = run_capture(
                replay_cmd,
                timeout=args.replay_timeout,
                output_path=run_dir / 'mission_replay.log',
            )
            summary['mission_replay_returncode'] = replay['returncode']
            summary['files']['mission_replay_log'] = str(run_dir / 'mission_replay.log')  # type: ignore[index]
            summary['files']['mission_replay_report'] = str(  # type: ignore[index]
                run_dir / 'mission_replay' / 'mission_replay_report.json'
            )

        print('STEP: post_system_check', flush=True)
        post_check = run_system_check(
            run_dir,
            'post_system_check',
            require_yolo=False,
            timeout=args.system_check_timeout,
        )
        summary['post_system_check_returncode'] = post_check['returncode']
        summary['files']['post_system_check'] = str(run_dir / 'post_system_check.json')  # type: ignore[index]

        if mission_returncode != 0:
            summary['status'] = 'failed'
            summary['failure_reason'] = f'auto_mission exited with code {mission_returncode}'
            return int(mission_returncode)

        summary['ok'] = True
        summary['status'] = 'completed'
        summary['failure_reason'] = ''
        return 0
    finally:
        if yolo_proc is not None:
            stop_process(yolo_proc)
            summary['yolo_returncode_after_stop'] = yolo_proc.poll()
        summary['finished_at'] = now_iso()
        if not summary.get('failure_reason') and not summary.get('ok'):
            summary['failure_reason'] = 'run interrupted before completion'
            summary['status'] = 'failed'
        write_json(run_dir / 'run_summary.json', summary)
        print(f'RUN_SUMMARY: {run_dir / "run_summary.json"}', flush=True)
        if summary.get('failure_reason'):
            print(f'RUN_FAILURE: {summary["failure_reason"]}', flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run system_check, optional YOLO, and auto_mission with records.'
    )
    parser.add_argument('--profile', default='safe_static')
    parser.add_argument('--profiles-config',
                        help='Run profile JSON path; default uses package config/run_profiles.json.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--runs-dir', default='/root/dev_ws/runs')
    parser.add_argument('--allow-motion', action='store_true',
                        help='Allow non-zero /cmd_vel only for motion-capable profiles.')
    parser.add_argument('--require-yolo', action='store_true')
    parser.add_argument('--start-yolo', action='store_true')
    parser.add_argument('--save-debug-images', action='store_true')
    parser.add_argument('--mock-qr-content')
    parser.add_argument('--force-direction', choices=['left', 'right'])
    parser.add_argument('--start-state', default='IDLE')
    parser.add_argument('--config')
    parser.add_argument('--debug-dir')
    parser.add_argument('--max-runtime', type=float)
    parser.add_argument('--drive-to-qr-timeout', type=float)
    parser.add_argument('--scan-qr-timeout', type=float)
    parser.add_argument('--cruise-linear', type=float)
    parser.add_argument('--loop-duration', type=float)
    parser.add_argument('--return-duration', type=float)
    parser.add_argument('--llm-mode',
                        choices=['placeholder', 'disabled', 'openai-compatible'])
    parser.add_argument('--llm-api-url')
    parser.add_argument('--llm-model')
    parser.add_argument('--llm-api-key-env')
    parser.add_argument('--llm-timeout', type=float)
    parser.add_argument('--llm-prompt')
    parser.add_argument('--system-check-timeout', type=float, default=12.0)
    parser.add_argument('--yolo-startup-timeout', type=float, default=30.0)
    parser.add_argument('--replay-timeout', type=float, default=30.0)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
