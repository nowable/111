#!/usr/bin/env python3
"""First competition mission state machine for OriginCar."""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from pyzbar import pyzbar as _pyzbar
except ImportError:
    _pyzbar = None

try:
    import rclpy
    try:
        from ai_msgs.msg import PerceptionTargets
    except ImportError:
        PerceptionTargets = None
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Float32, String
except ImportError:
    rclpy = None
    PerceptionTargets = None
    Twist = None
    CompressedImage = None
    Float32 = None
    String = None
    qos_profile_sensor_data = None
    SignalHandlerOptions = None

    class Node:  # type: ignore[no-redef]
        pass

    def remove_ros_args(args: List[str]) -> List[str]:
        return args

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None

try:
    from origin_competition_auto.decision_parser import (
        parse_direction as parse_qr_direction,
        parse_qr_instruction,
    )
    from origin_competition_auto.llm_client import LlmClient, LlmConfig
    from origin_competition_auto.ackermann_turn import execute_shunt_turn
    from origin_competition_auto.motion_state import MotionState, reverse_plan, angle_diff
    from origin_competition_auto.lane_follow import LaneFollower, LaneFollowConfig
    from origin_competition_auto.route_map import (
        DEFAULT_ROUTE_MAP,
        Pose2D,
        Waypoint,
        map_pose_from_odom,
        pose_from_mapping,
    )
    from origin_competition_auto.vision_detector import (
        Detection,
        DetectorConfig,
        VisionDetector,
        parse_ai_targets_msg,
    )
except ImportError:
    from decision_parser import (  # type: ignore[no-redef]
        parse_direction as parse_qr_direction,
        parse_qr_instruction,
    )
    from llm_client import LlmClient, LlmConfig  # type: ignore[no-redef]
    from ackermann_turn import execute_shunt_turn  # type: ignore[no-redef]
    from motion_state import MotionState, reverse_plan, angle_diff  # type: ignore[no-redef]
    from lane_follow import LaneFollower, LaneFollowConfig  # type: ignore[no-redef]
    from route_map import (  # type: ignore[no-redef]
        DEFAULT_ROUTE_MAP,
        Pose2D,
        Waypoint,
        map_pose_from_odom,
        pose_from_mapping,
    )
    from vision_detector import (  # type: ignore[no-redef]
        Detection,
        DetectorConfig,
        VisionDetector,
        parse_ai_targets_msg,
    )


PACKAGE_NAME = 'origin_competition_auto'
STATE_SEQUENCE = [
    'IDLE',
    'PREFLIGHT',
    'DRIVE_TO_QR',
    'SCAN_QR',
    'DECIDE_DIRECTION',
    'CORRIDOR_FOLLOW',
    'CORRIDOR_TO_LOOP_TURN',
    'LOOP_LEFT',
    'LOOP_RIGHT',
    'CAPTURE_TARGET_IMAGE',
    'CALL_LLM_API',
    'RETURN_HOME',
    'STOP',
    'DONE',
]

# Map each state to the physical competition zone it operates in. Emitted on
# every EVENT so log analysis / parse_mission_output can segment by zone.
STATE_ZONES = {
    'IDLE': 'A_HALL',
    'PREFLIGHT': 'A_HALL',
    'DRIVE_TO_QR': 'A_HALL',
    'SCAN_QR': 'A_HALL',
    'DECIDE_DIRECTION': 'A_HALL',
    'CORRIDOR_FOLLOW': 'B_CORRIDOR',
    'CORRIDOR_TO_LOOP_TURN': 'B_CORRIDOR',
    'LOOP_LEFT': 'C_RING',
    'LOOP_RIGHT': 'C_RING',
    'CAPTURE_TARGET_IMAGE': 'C_RING',
    'CALL_LLM_API': 'C_RING',
    'RETURN_HOME': 'RETURN',
    'STOP': 'A_HALL',
    'DONE': 'A_HALL',
}


def default_fixed_routes() -> Dict[str, List[Dict[str, object]]]:
    return {
        'A_TO_QR': [
            {'type': 'drive', 'distance': 0.37},
            {'type': 'drive', 'distance': 0.597},
            {'type': 'turn', 'angle_deg': 5.8},
            {'type': 'drive', 'distance': 0.546},
            {'type': 'drive', 'distance': 0.551},
            {'type': 'turn', 'angle_deg': 4.7},
            {'type': 'drive', 'distance': 0.563},
            {'type': 'turn', 'angle_deg': 4.1},
            {'type': 'drive', 'distance': 0.529},
            {'type': 'turn', 'angle_deg': 6.1},
            {'type': 'drive', 'distance': 0.496},
            {'type': 'drive', 'distance': 0.424},
            {'type': 'turn', 'angle_deg': 12.5},
            {'type': 'drive', 'distance': 0.344},
            {'type': 'drive', 'distance': 0.273},
            {'type': 'turn', 'angle_deg': -15.7},
            {'type': 'seek_qr', 'max_distance': 0.225},
        ],
        'QR_TO_CORRIDOR': [
            {'type': 'turn', 'angle_deg': 140.5},
            {'type': 'drive', 'distance': 0.888},
            {'type': 'turn', 'angle_deg': 7.3},
            {'type': 'drive', 'distance': 1.053},
            {'type': 'turn', 'angle_deg': -34.3},
            {'type': 'drive', 'distance': 0.256},
        ],
        'B_CORRIDOR': [
            {'type': 'turn', 'angle_deg': -51.3},
            {'type': 'drive', 'distance': 0.58},
        ],
        'C_LOOP_RIGHT': [
            {'type': 'arc', 'angle_deg': -330, 'radius_hint': 0.7},
        ],
        'C_LOOP_LEFT': [
            {'type': 'arc', 'angle_deg': 330, 'radius_hint': 0.7},
        ],
    }


@dataclass
class MissionConfig:
    cmd_vel_topic: str = '/cmd_vel'
    image_topic: str = '/image'
    voltage_topic: str = '/PowerVoltage'
    state_topic: str = '/competition/state'
    odom_topic: str = '/odom'
    imu_topic: str = '/imu/data'
    motion_state_enabled: bool = True
    use_imu_yaw: bool = True
    waypoint_min_dist: float = 0.05
    waypoint_min_yaw: float = 0.10
    odom_stale_timeout: float = 1.0
    debug_dir: str = '/root/dev_ws/debug/competition'
    rate_hz: float = 10.0
    stop_repeat: int = 5
    max_linear: float = 0.35
    max_angular: float = 0.58
    cruise_linear: float = 0.03
    turn_angular: float = 0.2
    # DEPRECATED 2026-06-11: 'shunt' 原地蹭转实测 45s 仅 128.7°，比赛 180s 烧不起。
    # 默认 'arc'（前进弧线 + integrated_yaw 闭环，150° 约 6.5s @ angular=0.4）。
    # 'shunt' 仅保留作极端窄场兜底，禁止再设为默认。
    ackermann_turn_mode: str = 'arc'  # 'arc' | 'shunt'(DEPRECATED)
    shunt_forward_linear: float = 0.06
    shunt_reverse_linear: float = 0.06
    shunt_pulse_s: float = 1.0
    shunt_stop_s: float = 0.20
    shunt_turn_timeout: float = 45.0
    steering_turn_linear: float = 0.03
    loop_linear: float = 0.03
    loop_angular: float = 0.2
    # Route modes: map2d=waypoint map, fixed2d=distance/yaw segments, legacy=old flow.
    route_mode: str = 'map2d'       # 'map2d' | 'fixed2d' | 'legacy'
    map_route_enabled: bool = True
    map_start_pose: Dict[str, float] = field(
        default_factory=lambda: {'x': 0.284, 'y': 0.246, 'heading_deg': 3.1}
    )
    map_waypoint_tol: float = 0.08
    map_heading_kp: float = 1.0
    map_max_linear: float = 0.03
    map_loop_linear: float = 0.05
    map_loop_turn_linear: float = 0.04
    map_max_angular: float = 0.25
    map_route_timeout: float = 120.0
    map_start_heading_guard_enabled: bool = True
    map_start_heading_guard_rad: float = 0.35
    map_detour_enabled: bool = True
    map_detour_turn_s: float = 0.8
    map_detour_drive_s: float = 0.9
    map_detour_recover_s: float = 0.8
    map_detour_cooldown_s: float = 1.0
    fixed_route_enabled: bool = True
    fixed_route_linear: float = 0.3
    fixed_route_turn_linear: float = 0.03
    fixed_route_turn_angular: float = 0.5
    fixed_route_heading_tol: float = 0.10
    fixed_route_dist_tol: float = 0.05
    fixed_route_segment_timeout: float = 30.0
    fixed_routes: Dict[str, List[Dict[str, object]]] = field(default_factory=default_fixed_routes)
    # Phase C: yellow lane following (corridor + ring)
    lane_follow_enabled: bool = True
    lane_roi_y_ratio: float = 0.6
    lane_roi_height_ratio: float = 0.35
    lane_kp: float = 0.75
    lane_kd: float = 0.13
    lane_base_linear: float = 0.03
    lane_min_mask_area_ratio: float = 0.01
    # Erosion kernel (px) for yellow corridor/ring follow: removes the thin yellow
    # grid-WALL lines (黄白相间网格) so the solid lane drives the centroid. 0=off.
    # Tune on-site if the car tracks the grid wall instead of the solid lane.
    lane_mask_erode_px: int = 0
    lane_lost_behavior: str = 'creep'   # 'creep' | 'stop'
    hall_line_follow_enabled: bool = True
    ring_bias: float = 0.25
    ring_avoid_linear_scale: float = 0.5
    corridor_length_m: float = 2.0
    green_exit_ratio: float = 0.15
    # Debounce the green-floor corridor exit: require this many CONSECUTIVE frames
    # over green_exit_ratio before declaring the C-zone reached. Without it a single
    # transient spike (reflection / motion blur / a stray frame) ends the corridor
    # early and dumps the car into the ring before it arrives (field 6-14: 报
    # green_floor 但 CORRIDOR_DIAG green_ratio 仅 0.000-0.006).
    green_exit_frames: int = 3
    loop_complete_fraction: float = 0.92
    loop_timeout: float = 40.0
    loop_min_time_s: float = 8.0
    corridor_timeout: float = 30.0
    corridor_require_green_exit: bool = False
    transition_settle_s: float = 0.3
    marker_capture_min_area: float = 0.02
    target_marker_policy: str = 'directional_one'
    marker_capture_window_m: float = 0.35
    return_linear: float = -0.03
    return_angular: float = 0.0
    # Phase F: trajectory-replay return home
    return_mode: str = 'map2d_fallback'   # 'map2d_fallback' | 'trajectory' | 'timed'
    map_return_route_enabled: bool = True
    map_return_timeout: float = 90.0
    return_waypoint_stride: int = 3
    return_min_segment_dist: float = 0.05
    return_drive_linear: float = 0.04
    return_turn_linear: float = 0.03
    return_turn_angular: float = 0.4
    return_heading_tol: float = 0.10     # rad; stop rotating within this
    return_dist_tol: float = 0.05        # m; stop driving within this
    return_segment_timeout: float = 30.0
    return_total_timeout: float = 60.0
    return_reacquire_enabled: bool = True
    return_reacquire_turn_deg: float = 90.0
    return_reacquire_forward_s: float = 0.8
    return_reacquire_turn2_deg: float = 42.0
    return_reacquire_forward2_s: float = 1.0
    return_reacquire_search_timeout: float = 6.0
    return_reacquire_linear: float = 0.05
    return_reacquire_angular: float = 0.22
    return_line_lost_grace_frames: int = 5
    preflight_timeout: float = 5.0
    drive_to_qr_timeout: float = 20.0
    drive_qr_area_threshold: float = 0.026
    drive_to_qr_max_distance: float = 2.5
    # Inchworm approach to the QR board: scan a clean STATIONARY frame, and only
    # if no QR decodes, creep forward one short chunk before scanning again.
    # Detection while moving is unreliable (motion blur garbles even the QR
    # boundary); stationary decode works down to QR-area ~0.008.
    drive_chunk_s: float = 2.0          # smooth forward drive between stationary scans
    drive_scan_settle_s: float = 0.5    # pause after stopping so blur clears
    # When the black hall line is LOST mid-drive — typically right after an
    # obstacle-avoidance turn swings the car off it — recover by arcing back
    # toward the side the line was last seen, at a USABLE speed, instead of
    # crawling straight at cruise_linear (0.04 m/s) which looks like the car has
    # stalled and never re-finds the line/QR (field 6-17: 避障扭头后看不见线/码
    # 就极慢直行). 0 angular falls back to a straight nudge at drive_lost_linear.
    drive_lost_linear: float = 0.3
    drive_lost_angular: float = 0.30
    recover_after_avoid_s: float = 3.0
    recover_sweep_max_s: float = 1.5
    drive_recover_search_roi: float = 0.4  # search ROI y-start; captures line that drifted above bottom follow ROI
    qr_detect_time_budget_s: float = 0.25
    qr_min_area_ratio: float = 0.01
    # 扫码重锚定：码牌解码停车点物理位置确定（=航点 qr_reanchor_waypoint），
    # 在此把地图系平移漂移清零，斩断 A 区开环误差向 B/C/返航传播。只锚 x/y 不动 yaw。
    qr_reanchor_enabled: bool = True
    qr_reanchor_waypoint: str = 'QR_BOARD'
    qr_reanchor_max_jump: float = 0.8  # 修正量超此值视为异常，跳过并告警
    qr_center_band: float = 0.4
    qr_seek_after_s: float = 4.0
    qr_seek_angular: float = 0.15
    qr_align_after_scan_enabled: bool = True
    qr_exit_heading_deg: float = 180.0
    qr_align_heading_tol: float = 0.10
    scan_qr_timeout: float = 8.0
    # Anti-jump: require this many consecutive same-direction decodes before
    # locking the direction. Once locked it never changes (rule: 不可跳变).
    qr_confirm_count: int = 3
    # Explicit decoded-text -> 'left'/'right' map, consulted BEFORE keyword
    # parsing. The competition QR content is unknown ahead of time (practice
    # field decodes to e.g. "1334"); scan it on-site and add the mapping here,
    # e.g. {"1334": "right"}. Keys are matched on the stripped decoded text.
    qr_direction_map: Dict[str, str] = field(default_factory=dict)
    loop_duration: float = 2.0
    capture_timeout: float = 3.0
    return_duration: float = 1.0
    max_runtime: float = 175.0
    image_stale_timeout: float = 3.0
    use_yolo_detections: bool = True
    yolo_topic: str = '/hobot_dnn_detection'
    yolo_stale_timeout: float = 1.0
    yolo_min_confidence: float = 0.45
    # In-process YOLO11n (yolo_detector.py): '' disables; .bin on board / .onnx local.
    yolo_local_model: str = ''
    yolo_local_conf: float = 0.50
    # When the local model is healthy, skip HSV obstacle masks (YOLO primary).
    yolo_replaces_hsv_obstacle: bool = True
    yolo_local_min_interval: float = 0.12
    obstacle_avoid_enabled: bool = True
    obstacle_turn_linear: float = 0.03
    obstacle_turn_angular: float = 0.18
    # Forward speed used to DRIVE PAST an obstacle once it is off to the side
    # (zone left/right). At the low obstacle_turn_linear the Ackermann car only
    # pivots beside a side cone and stays stuck on it (field 6-17); a faster
    # forward push translates the car past it so it leaves the frame.
    obstacle_pass_linear: float = 0.3
    obstacle_roi_y_ratio: float = 0.45
    obstacle_min_area_ratio: float = 0.015
    obstacle_danger_area_ratio: float = 0.035
    obstacle_center_band_ratio: float = 0.36
    black_v_max: int = 85
    black_s_min: int = 20
    # Named HSV color profiles and which color each role uses (Phase A).
    color_profiles: Dict[str, Dict[str, object]] = field(default_factory=dict)
    obstacle_color: str = 'dark_blue'
    lane_color: str = 'yellow'
    marker_color: str = 'orange'
    green_color: str = 'green'
    require_triangle: bool = False
    target_crop_enabled: bool = True
    target_min_area_ratio: float = 0.025
    target_white_s_max: int = 90
    target_white_v_min: int = 150
    llm_mode: str = 'placeholder'
    llm_api_url: str = 'https://api.openai.com/v1/chat/completions'
    llm_model: str = 'gpt-4.1-mini'
    llm_api_key_env: str = 'OPENAI_API_KEY'
    llm_timeout: float = 20.0
    llm_prompt: str = (
        'Identify the main object or symbol in this competition image. '
        'Answer briefly in Chinese, and include uncertainty if the image is unclear.'
    )
    llm_max_image_dim: int = 1024
    llm_jpeg_quality: int = 85
    llm_downscale_enabled: bool = True
    llm_result_topic: str = '/competition/llm_result'
    # Vision bridge: QR board -> corridor mouth (legacy route_mode only)
    bridge_timeout: float = 15.0
    bridge_yellow_area_min: float = 0.06
    bridge_center_band: float = 0.24
    bridge_turn_angular: float = 0.50
    bridge_turn_linear: float = 0.03
    bridge_search_linear: float = 0.08
    bridge_search_angular: float = 0.40
    bridge_align_linear: float = 0.06
    bridge_phase1_forward_s: float = 2.0
    bridge_phase1_forward_angular: float = 0.12
    bridge_yellow_detect_min_ratio: float = 0.03
    bridge_yellow_search_linear: float = 0.08
    bridge_yellow_search_angular: float = 0.35
    bridge_yellow_follow_linear: float = 0.08
    bridge_yellow_follow_angular: float = 0.30
    bridge_yellow_parallel_target_offset: float = 0.55
    bridge_yellow_parallel_tol: float = 0.10
    bridge_black_search_linear: float = 0.06
    bridge_black_reentry_enabled: bool = True
    bridge_black_reentry_roi_y_ratio: float = 0.45
    bridge_black_reentry_confirm_frames: int = 2
    bridge_black_reentry_turn_angle_deg: float = 120.0
    bridge_black_reentry_turn_angular: float = 0.5
    bridge_black_reentry_turn_linear: float = 0.08
    bridge_black_reentry_forward_s: float = 1.2
    bridge_entry_reference_enabled: bool = True
    bridge_entry_reference_image: str = '/root/dev_ws/debug/competition/return_gate_snapshot.jpg'
    bridge_entry_reference_min_score: float = 0.55
    bridge_entry_reference_timeout: float = 4.0
    bridge_entry_reference_roi_x_ratio: float = 0.25
    bridge_entry_reference_roi_y_ratio: float = 0.12
    bridge_entry_reference_roi_w_ratio: float = 0.50
    bridge_entry_reference_roi_h_ratio: float = 0.45
    bridge_search_dir: str = 'left'
    bridge_target_roi_y_ratio: float = 0.18
    bridge_target_min_area_ratio: float = 0.015
    bridge_target_bottom_ratio: float = 0.52
    bridge_target_center_tol: float = 0.12
    bridge_target_center_gate_ratio: float = 0.22
    bridge_yellow_erode_px: int = 21
    bridge_target_turn_bias: float = 0.08
    # Legacy black-line bridge knob kept for backward compatibility with older
    # configs/logs. The current bridge seeks the solid yellow corridor mouth.
    bridge_line_lost_frames: int = 3
    # 扫码后倒车解卡 (field 6-15): SCAN_QR 停车点车头顶死立牌 (~0.3m)，相机只剩
    # 立牌 + 橙色发布台垫，黑线被遮 → 旧 bridge 把橙垫当通道口在 1.1s 假触发。
    # 进 bridge 先倒车脱离码牌。停车点黑线本就在 ROI 里(车沿线开到牌前)，靠"看见线"
    # 停会几乎不倒就退出→forward bridge 朝牌前进撞码(field 6-17)。改为按 odom 里程
    # 后退固定距离 bridge_backup_distance_m，退到任务发布区后方、能看见通道口为止。
    bridge_backup_enabled: bool = True
    bridge_backup_linear: float = -0.3    # 倒车速度(负=后退；幅值受 max_linear 钳)
    bridge_backup_distance_m: float = 0.35  # 后退固定距离(odom 里程判停；现场调)
    # 倒车时同时给角速度→阿克曼边退边转头,把车头从"对着码牌/右侧网格"摆向中间通道。
    # 符号:>0=车头左转(CCW)。码牌在右、通道在中,默认左转朝中间;转反了翻符号即可。
    bridge_backup_angular: float = 0.30
    bridge_backup_max_s: float = 6.0       # 倒车时间上限(安全帽；无 odom 时按此纯计时后退)
    bridge_post_backup_forward_s: float = 4.0
    bridge_post_backup_linear: float = 0.30
    bridge_post_backup_angular: float = 0.0

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> 'MissionConfig':
        known = {field.name for field in fields(cls)}
        values = {key: value for key, value in data.items() if key in known}
        return cls(**values)

    def apply_args(self, args: argparse.Namespace) -> None:
        override_pairs = {
            'max_runtime': args.max_runtime,
            'drive_to_qr_timeout': args.drive_to_qr_timeout,
            'scan_qr_timeout': args.scan_qr_timeout,
            'capture_timeout': args.capture_timeout,
            'cruise_linear': args.cruise_linear,
            'turn_angular': args.turn_angular,
            'loop_duration': args.loop_duration,
            'return_duration': args.return_duration,
            'debug_dir': args.debug_dir,
            'llm_mode': args.llm_mode,
            'llm_api_url': args.llm_api_url,
            'llm_model': args.llm_model,
            'llm_api_key_env': args.llm_api_key_env,
            'llm_timeout': args.llm_timeout,
            'llm_prompt': args.llm_prompt,
        }
        for name, value in override_pairs.items():
            if value is not None:
                setattr(self, name, value)

    def validate(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError('rate_hz must be > 0')
        if self.stop_repeat < 1:
            raise ValueError('stop_repeat must be >= 1')
        if self.max_linear <= 0.0 or self.max_linear > 3.0:
            raise ValueError('max_linear must be in (0, 3.0]')
        if self.max_angular <= 0.0 or self.max_angular > 0.8:
            raise ValueError('max_angular must be in (0, 0.8]')
        if self.ackermann_turn_mode not in ('shunt', 'arc'):
            raise ValueError("ackermann_turn_mode must be 'shunt' or 'arc'")
        if not 0.0 < self.shunt_forward_linear <= self.max_linear:
            raise ValueError('shunt_forward_linear must be in (0, max_linear]')
        if not 0.0 < self.shunt_reverse_linear <= self.max_linear:
            raise ValueError('shunt_reverse_linear must be in (0, max_linear]')
        if not 0.0 <= self.steering_turn_linear <= self.max_linear:
            raise ValueError('steering_turn_linear must be in [0, max_linear]')
        duration_names = [
            'preflight_timeout',
            'drive_to_qr_timeout',
            'scan_qr_timeout',
            'loop_duration',
            'capture_timeout',
            'return_duration',
            'max_runtime',
            'image_stale_timeout',
            'yolo_stale_timeout',
            'llm_timeout',
            'map_route_timeout',
            'map_detour_turn_s',
            'map_detour_drive_s',
            'map_detour_recover_s',
            'map_detour_cooldown_s',
            'shunt_pulse_s',
            'shunt_stop_s',
            'shunt_turn_timeout',
            'map_return_timeout',
            'fixed_route_segment_timeout',
            'qr_detect_time_budget_s',
            'recover_after_avoid_s',
            'recover_sweep_max_s',
        ]
        for name in duration_names:
            if getattr(self, name) < 0.0:
                raise ValueError(f'{name} must be >= 0')
        if self.obstacle_turn_angular < 0.0:
            raise ValueError('obstacle_turn_angular must be >= 0')
        if not 0.0 <= self.obstacle_turn_linear <= self.max_linear:
            raise ValueError('obstacle_turn_linear must be in [0, max_linear]')
        if self.route_mode not in ('map2d', 'fixed2d', 'legacy'):
            raise ValueError("route_mode must be 'map2d', 'fixed2d', or 'legacy'")
        if self.map_max_linear <= 0.0 or self.map_max_linear > self.max_linear:
            raise ValueError('map_max_linear must be in (0, max_linear]')
        if self.map_loop_linear <= 0.0 or self.map_loop_linear > self.max_linear:
            raise ValueError('map_loop_linear must be in (0, max_linear]')
        if not 0.0 <= self.map_loop_turn_linear <= self.map_loop_linear:
            raise ValueError('map_loop_turn_linear must be in [0, map_loop_linear]')
        if not 0.0 < self.map_max_angular <= self.max_angular:
            raise ValueError('map_max_angular must be in (0, max_angular]')
        if self.map_waypoint_tol <= 0.0:
            raise ValueError('map_waypoint_tol must be > 0')
        if self.map_heading_kp < 0.0:
            raise ValueError('map_heading_kp must be >= 0')
        if self.map_start_heading_guard_rad < 0.0:
            raise ValueError('map_start_heading_guard_rad must be >= 0')
        if not isinstance(self.map_start_pose, dict):
            raise ValueError('map_start_pose must be an object')
        if self.fixed_route_linear <= 0.0 or self.fixed_route_linear > self.max_linear:
            raise ValueError('fixed_route_linear must be in (0, max_linear]')
        if not 0.0 <= self.fixed_route_turn_linear <= self.max_linear:
            raise ValueError('fixed_route_turn_linear must be in [0, max_linear]')
        if not 0.0 < self.fixed_route_turn_angular <= self.max_angular:
            raise ValueError('fixed_route_turn_angular must be in (0, max_angular]')
        if self.fixed_route_heading_tol <= 0.0 or self.fixed_route_dist_tol <= 0.0:
            raise ValueError('fixed route tolerances must be > 0')
        if not isinstance(self.fixed_routes, dict):
            raise ValueError('fixed_routes must be an object')
        for color_name in (
            self.obstacle_color,
            self.lane_color,
            self.marker_color,
            self.green_color,
        ):
            if not isinstance(color_name, str) or not color_name:
                raise ValueError('color role names must be non-empty strings')
        if self.drive_to_qr_max_distance <= 0.0:
            raise ValueError('drive_to_qr_max_distance must be > 0')
        if not 0.0 <= self.qr_min_area_ratio <= 1.0:
            raise ValueError('qr_min_area_ratio must be in [0, 1]')
        if not 0.0 < self.qr_center_band <= 2.0:
            raise ValueError('qr_center_band must be in (0, 2]')
        if self.qr_seek_after_s < 0.0:
            raise ValueError('qr_seek_after_s must be >= 0')
        if not 0.0 <= self.qr_seek_angular <= self.max_angular:
            raise ValueError('qr_seek_angular must be in [0, max_angular]')
        if self.qr_align_heading_tol <= 0.0:
            raise ValueError('qr_align_heading_tol must be > 0')
        if self.qr_confirm_count < 1:
            raise ValueError('qr_confirm_count must be >= 1')
        if self.lane_base_linear <= 0.0 or self.lane_base_linear > self.max_linear:
            raise ValueError('lane_base_linear must be in (0, max_linear]')
        if not 0.0 <= self.lane_roi_y_ratio < 1.0:
            raise ValueError('lane_roi_y_ratio must be in [0, 1)')
        if not 0.0 < self.lane_roi_height_ratio <= 1.0:
            raise ValueError('lane_roi_height_ratio must be in (0, 1]')
        if self.lane_kp < 0.0 or self.lane_kd < 0.0:
            raise ValueError('lane_kp and lane_kd must be >= 0')
        if not 0.0 < self.loop_complete_fraction <= 1.5:
            raise ValueError('loop_complete_fraction must be in (0, 1.5]')
        if self.lane_lost_behavior not in ('creep', 'stop'):
            raise ValueError("lane_lost_behavior must be 'creep' or 'stop'")
        if self.corridor_length_m <= 0.0:
            raise ValueError('corridor_length_m must be > 0')
        if self.transition_settle_s < 0.0:
            raise ValueError('transition_settle_s must be >= 0')
        if self.target_marker_policy not in ('directional_one', 'best_any'):
            raise ValueError("target_marker_policy must be 'directional_one' or 'best_any'")
        if self.marker_capture_window_m <= 0.0:
            raise ValueError('marker_capture_window_m must be > 0')
        if self.return_mode not in ('map2d_fallback', 'trajectory', 'timed'):
            raise ValueError("return_mode must be 'map2d_fallback', 'trajectory', or 'timed'")
        if self.return_waypoint_stride < 1:
            raise ValueError('return_waypoint_stride must be >= 1')
        if self.return_drive_linear <= 0.0 or self.return_drive_linear > self.max_linear:
            raise ValueError('return_drive_linear must be in (0, max_linear]')
        if not 0.0 <= self.return_turn_linear <= self.max_linear:
            raise ValueError('return_turn_linear must be in [0, max_linear]')
        if not 0.0 < self.return_turn_angular <= self.max_angular:
            raise ValueError('return_turn_angular must be in (0, max_angular]')
        if self.return_heading_tol <= 0.0 or self.return_dist_tol <= 0.0:
            raise ValueError('return tolerances must be > 0')
        if self.llm_mode not in (
            'placeholder',
            'disabled',
            'openai-compatible',
            'openai',
            'http',
        ):
            raise ValueError('llm_mode must be placeholder, disabled, or openai-compatible')

    def fixed_route_active(self) -> bool:
        return self.fixed_route_enabled and self.route_mode == 'fixed2d'

    def map_route_active(self) -> bool:
        return self.map_route_enabled and self.route_mode == 'map2d'

    def detector_config(self) -> DetectorConfig:
        mapping: Dict[str, object] = {
            'obstacle_roi_y_ratio': self.obstacle_roi_y_ratio,
            'obstacle_min_area_ratio': self.obstacle_min_area_ratio,
            'obstacle_danger_area_ratio': self.obstacle_danger_area_ratio,
            'obstacle_center_band_ratio': self.obstacle_center_band_ratio,
            'black_v_max': self.black_v_max,
            'black_s_min': self.black_s_min,
            'target_min_area_ratio': self.target_min_area_ratio,
            'target_white_s_max': self.target_white_s_max,
            'target_white_v_min': self.target_white_v_min,
            'yolo_min_confidence': self.yolo_min_confidence,
            'obstacle_color': self.obstacle_color,
            'lane_color': self.lane_color,
            'marker_color': self.marker_color,
            'green_color': self.green_color,
            'require_triangle': self.require_triangle,
        }
        # Only override profiles the user actually supplied; defaults fill the rest.
        if self.color_profiles:
            mapping['color_profiles'] = self.color_profiles
        return DetectorConfig.from_mapping(mapping)

    def llm_config(self) -> LlmConfig:
        return LlmConfig(
            mode=self.llm_mode,
            api_url=self.llm_api_url,
            model=self.llm_model,
            api_key_env=self.llm_api_key_env,
            timeout=self.llm_timeout,
            prompt=self.llm_prompt,
            max_image_dim=self.llm_max_image_dim,
            jpeg_quality=self.llm_jpeg_quality,
            downscale_enabled=self.llm_downscale_enabled,
        )

    def lane_follow_config(
        self, bias: float = 0.0, side_mode: str = 'center', erode_px: int = 0,
        roi_y_ratio: Optional[float] = None, roi_height_ratio: Optional[float] = None,
    ) -> LaneFollowConfig:
        roi_y_ratio = self.lane_roi_y_ratio if roi_y_ratio is None else roi_y_ratio
        roi_height_ratio = self.lane_roi_height_ratio if roi_height_ratio is None else roi_height_ratio
        return LaneFollowConfig(
            roi_y_ratio=roi_y_ratio,
            roi_height_ratio=roi_height_ratio,
            kp=self.lane_kp,
            kd=self.lane_kd,
            base_linear=self.lane_base_linear,
            max_angular=self.max_angular,
            min_mask_area_ratio=self.lane_min_mask_area_ratio,
            bias=bias,
            side_mode=side_mode,
            mask_erode_px=erode_px,
        )


class QrResult:
    def __init__(
        self,
        text: str = '',
        points: Optional[np.ndarray] = None,
        method: str = 'not_found',
    ) -> None:
        self.text = text
        self.points = points
        self.method = method


class QrDecoder:
    def __init__(self) -> None:
        cv2.setNumThreads(1)
        self.detector = cv2.QRCodeDetector()
        self.pyzbar_available = _pyzbar is not None

    def _detect_pyzbar(
        self, gray: np.ndarray, scales: tuple = (1.0, 2.0)
    ) -> Optional[QrResult]:
        """pyzbar decode on grayscale; returns result only on real decode."""
        if _pyzbar is None:
            return None
        for scale in scales:
            if scale == 1.0:
                probe = gray
            else:
                probe = cv2.resize(
                    gray, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            try:
                symbols = _pyzbar.decode(probe)
            except Exception:
                return None
            for sym in symbols:
                if not sym.data:
                    continue
                text = sym.data.decode('utf-8', errors='replace')
                points = np.array(
                    [[p.x / scale, p.y / scale] for p in sym.polygon],
                    dtype=np.float32,
                )
                method = 'pyzbar' if scale == 1.0 else f'pyzbar_x{int(scale)}'
                return QrResult(text, points, method)
        return None

    def detect(
        self, image: np.ndarray, time_budget_s: Optional[float] = None
    ) -> QrResult:
        detect_start_t = time.monotonic()

        def budget_expired() -> bool:
            return (
                time_budget_s is not None
                and time.monotonic() - detect_start_t >= time_budget_s
            )

        gray_first = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 板上实测：无码帧若全跑（pyzbar 2x + OpenCV x2/x3 全图）单帧可达 1.8s，
        # 拖垮 DRIVE_TO_QR 控制环。故 miss 路径只留便宜步骤，贵的全部
        # 移到"OpenCV 已发现候选框"（说明视野里很可能有码）之后。
        zbar = self._detect_pyzbar(gray_first, scales=(1.0,))
        if zbar is not None:
            return zbar
        gray = gray_first
        candidates = [('bgr', image), ('gray', gray)]

        best_points = None
        for name, candidate in candidates:
            decoded, points, _ = self.detector.detectAndDecode(candidate)
            if decoded:
                return QrResult(decoded, self._scale_points_back(points, name), name)
            if points is not None and best_points is None:
                best_points = self._scale_points_back(points, name)

        result: Optional[QrResult] = None
        heavy_budget_expired = False
        if best_points is not None:
            if budget_expired():
                heavy_budget_expired = True
            else:
                zbar = self._detect_pyzbar(gray, scales=(2.0,))
                if zbar is not None:
                    result = zbar
            if result is None and not heavy_budget_expired:
                if budget_expired():
                    heavy_budget_expired = True
                else:
                    for warp_name, warped in self._qr_warps_from_points(gray, best_points):
                        if budget_expired():
                            heavy_budget_expired = True
                            break
                        decoded, _, _ = self.detector.detectAndDecode(warped)
                        if decoded:
                            result = QrResult(decoded, best_points, warp_name)
                            break

            if result is None and not heavy_budget_expired:
                if budget_expired():
                    heavy_budget_expired = True
                else:
                    for crop_name, crop in self._qr_crops_from_points(gray, best_points):
                        if budget_expired():
                            heavy_budget_expired = True
                            break
                        enlarged = cv2.resize(
                            crop, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC
                        )
                        decoded, _, _ = self.detector.detectAndDecode(enlarged)
                        if decoded:
                            result = QrResult(decoded, best_points, f'{crop_name}_gray_x6')
                            break

                        _, binary = cv2.threshold(
                            enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                        )
                        decoded, _, _ = self.detector.detectAndDecode(binary)
                        if decoded:
                            result = QrResult(decoded, best_points, f'{crop_name}_otsu_x6')
                            break

        if result is None:
            result = QrResult('', best_points, 'not_found')
        elapsed_s = time.monotonic() - detect_start_t
        if elapsed_s >= 0.25:
            print(
                f'QR_DETECT_SLOW elapsed={elapsed_s:.3f}s path={result.method or "not_found"}',
                flush=True,
            )
        return result

    def _scale_points_back(
        self, points: Optional[np.ndarray], method: str
    ) -> Optional[np.ndarray]:
        if points is None or '_x' not in method:
            return points
        try:
            scale = float(method.rsplit('_x', 1)[1])
        except ValueError:
            return points
        return points / scale

    def _qr_crops_from_points(
        self, gray: np.ndarray, points: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        pts = points.reshape(-1, 2)
        min_x, min_y = np.floor(pts.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(pts.max(axis=0)).astype(int)
        width = max_x - min_x
        height = max_y - min_y
        crops = []
        for ratio in (0.15, 0.3):
            pad = int(max(width, height) * ratio)
            x1 = max(0, min_x - pad)
            y1 = max(0, min_y - pad)
            x2 = min(gray.shape[1], max_x + pad)
            y2 = min(gray.shape[0], max_y + pad)
            if x2 > x1 and y2 > y1:
                crops.append((f'crop_pad_{ratio:.2f}', gray[y1:y2, x1:x2]))
        return crops

    def _qr_warps_from_points(
        self, gray: np.ndarray, points: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        pts = self._order_points(points.reshape(4, 2).astype(np.float32))
        warps = []
        for size in (600, 800):
            dst = np.array(
                [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(gray, matrix, (size, size))
            border = int(size * 0.15)
            bordered = cv2.copyMakeBorder(
                warped,
                border,
                border,
                border,
                border,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            warps.append((f'perspective_border_{size}', bordered))
        return warps

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        ordered = np.zeros((4, 2), dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1).reshape(-1)
        ordered[0] = pts[np.argmin(sums)]
        ordered[2] = pts[np.argmax(sums)]
        ordered[1] = pts[np.argmin(diffs)]
        ordered[3] = pts[np.argmax(diffs)]
        return ordered


class VisionBuffer:
    def __init__(self, node: Node, topic: str) -> None:
        self.node = node
        self.latest_image: Optional[np.ndarray] = None
        self.latest_time: Optional[float] = None
        self.frame_count = 0
        self.subscription = node.create_subscription(
            CompressedImage,
            topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

    def _image_callback(self, msg: CompressedImage) -> None:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            self.node.get_logger().warn('failed to decode camera frame')
            return
        self.latest_image = image
        self.latest_time = time.monotonic()
        self.frame_count += 1

    def has_fresh_image(self, max_age: float) -> bool:
        return self.get_latest(max_age=max_age) is not None

    def get_latest(self, max_age: Optional[float] = None) -> Optional[np.ndarray]:
        if self.latest_image is None or self.latest_time is None:
            return None
        if max_age is not None and time.monotonic() - self.latest_time > max_age:
            return None
        return self.latest_image.copy()

    def save_image(self, path: Path, image: Optional[np.ndarray] = None) -> bool:
        selected = image if image is not None else self.get_latest()
        if selected is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return cv2.imwrite(str(path), selected)


class MotionCommander:
    def __init__(self, node: Node, config: MissionConfig, no_motion: bool) -> None:
        self.node = node
        self.config = config
        self.no_motion = no_motion
        self.publisher = node.create_publisher(Twist, config.cmd_vel_topic, 10)
        self._suppression_logged = False

    def subscriber_count(self) -> int:
        return self.publisher.get_subscription_count()

    def publish(self, linear_x: float, angular_z: float) -> None:
        linear = self._clamp(linear_x, self.config.max_linear, 'linear.x')
        angular = self._clamp(angular_z, self.config.max_angular, 'angular.z')
        if self.no_motion and (abs(linear) > 1e-6 or abs(angular) > 1e-6):
            if not self._suppression_logged:
                self.node.get_logger().warn(
                    '--no-motion active; suppressing non-zero /cmd_vel'
                )
                self._suppression_logged = True
            linear = 0.0
            angular = 0.0
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher.publish(msg)

    def stop_once(self) -> None:
        self.publish(0.0, 0.0)

    def stop(self, repeat: Optional[int] = None) -> None:
        count = repeat if repeat is not None else self.config.stop_repeat
        for _ in range(max(1, count)):
            self.stop_once()

    def _clamp(self, value: float, limit: float, label: str) -> float:
        if abs(value) <= limit:
            return value
        clipped = math.copysign(limit, value)
        self.node.get_logger().warn(
            f'{label}={value:.3f} exceeds limit {limit:.3f}; using {clipped:.3f}'
        )
        return clipped


class SafetyGuard:
    def __init__(self, node: Node, config: MissionConfig) -> None:
        self.node = node
        self.config = config
        self.started_at = time.monotonic()
        self.latest_voltage: Optional[float] = None
        self.latest_voltage_time: Optional[float] = None
        self.subscription = node.create_subscription(
            Float32,
            config.voltage_topic,
            self._voltage_callback,
            10,
        )

    def _voltage_callback(self, msg: Float32) -> None:
        self.latest_voltage = float(msg.data)
        self.latest_voltage_time = time.monotonic()

    def runtime_ok(self) -> bool:
        return time.monotonic() - self.started_at <= self.config.max_runtime

    def runtime_remaining(self) -> float:
        return self.config.max_runtime - (time.monotonic() - self.started_at)

    def has_voltage(self) -> bool:
        return self.latest_voltage is not None


class MissionAbort(RuntimeError):
    pass


class AutoMissionNode(Node):
    def __init__(
        self,
        config: MissionConfig,
        args: argparse.Namespace,
    ) -> None:
        super().__init__('auto_mission')
        self.config = config
        self.args = args
        self.debug_dir = Path(config.debug_dir)
        self.state_pub = self.create_publisher(String, config.state_topic, 10)
        self.llm_result_pub = self.create_publisher(String, config.llm_result_topic, 10)
        self.vision = VisionBuffer(self, config.image_topic)
        self.motion = MotionCommander(self, config, args.no_motion)
        self.guard = SafetyGuard(self, config)
        self.qr_decoder = QrDecoder()
        self.vision_detector = VisionDetector(config.detector_config())
        self.llm_client = LlmClient(config.llm_config())
        self.motion_state: Optional[MotionState] = None
        if config.motion_state_enabled:
            self.motion_state = MotionState(
                self,
                odom_topic=config.odom_topic,
                imu_topic=config.imu_topic,
                use_imu_yaw=config.use_imu_yaw,
                waypoint_min_dist=config.waypoint_min_dist,
                waypoint_min_yaw=config.waypoint_min_yaw,
                stale_timeout=config.odom_stale_timeout,
            )
            self.get_logger().info(
                f'motion_state on odom={config.odom_topic} imu={config.imu_topic}')
        self.latest_yolo_detections: List[Detection] = []
        self.latest_yolo_time: Optional[float] = None
        self.yolo_subscription = None
        if config.use_yolo_detections and PerceptionTargets is not None:
            self.yolo_subscription = self.create_subscription(
                PerceptionTargets,
                config.yolo_topic,
                self._yolo_callback,
                10,
            )
            self.get_logger().info(f'subscribed to YOLO detections {config.yolo_topic}')
        elif config.use_yolo_detections:
            self.get_logger().warn('ai_msgs is unavailable; YOLO detections disabled')
        self.local_yolo = None
        self._local_yolo_cache: List[Detection] = []
        self._local_yolo_cache_time = 0.0
        self._bridge_entry_reference_mask: Optional[np.ndarray] = None
        self._bridge_entry_reference_ready = False
        self._load_bridge_entry_reference()
        if config.yolo_local_model:
            try:
                from .yolo_detector import YoloDetector
            except ImportError:
                from yolo_detector import YoloDetector
            try:
                self.local_yolo = YoloDetector(
                    config.yolo_local_model, conf_thres=config.yolo_local_conf)
                self.get_logger().info(
                    f'local YOLO backend={self.local_yolo.backend} '
                    f'model={config.yolo_local_model}')
            except Exception as exc:
                self.get_logger().warn(
                    f'local YOLO load failed, falling back to HSV: {exc}')
        self.started_at = time.monotonic()
        self.current_state = 'IDLE'
        self.state_entry_counts: Dict[str, int] = {}
        self.obstacle_event_count = 0
        self.target_card_seen_count = 0
        self.saved_images: List[str] = []
        self.captured_markers: List[Dict[str, object]] = []
        self.failure_reason = ''
        self.qr_content = args.mock_qr_content or ''
        self.qr_method = ''
        self.direction = args.force_direction or ''
        self.direction_source = ''
        # Anti-jump lock: once a direction is confirmed it is frozen.
        self.direction_locked = bool(args.force_direction)
        self._dir_vote = ''        # candidate direction being accumulated
        self._dir_vote_count = 0   # consecutive same-direction decode count
        self.target_image_path: Optional[str] = None
        self.target_marker_name = ''
        self.target_marker_route = ''
        self.llm_result = ''
        self.route_map = DEFAULT_ROUTE_MAP
        self._map_odom_start: Optional[Tuple[float, float, float]] = None
        self._map_start_pose: Pose2D = pose_from_mapping(
            config.map_start_pose,
            self.route_map.start_pose,
        )
        self._last_map_obstacle_time = 0.0

    def _yolo_callback(self, msg: object) -> None:
        self.latest_yolo_detections = parse_ai_targets_msg(
            msg,
            min_confidence=self.config.yolo_min_confidence,
        )
        self.latest_yolo_time = time.monotonic()

    def _fresh_yolo_detections(self) -> List[Detection]:
        if self.latest_yolo_time is None:
            return []
        if time.monotonic() - self.latest_yolo_time > self.config.yolo_stale_timeout:
            return []
        return list(self.latest_yolo_detections)

    def _local_yolo_detections(self, image) -> List[Detection]:
        if self.local_yolo is None or image is None:
            return []
        now = time.monotonic()
        if now - self._local_yolo_cache_time < self.config.yolo_local_min_interval:
            return list(self._local_yolo_cache)
        try:
            raw = self.local_yolo.detect(image)
        except Exception as exc:
            self.get_logger().warn(f'local YOLO inference failed, disabling: {exc}')
            self.local_yolo = None
            return []
        self._local_yolo_cache = [
            Detection(
                label=det.class_name,
                confidence=det.score,
                bbox=(int(det.x1), int(det.y1),
                      int(det.x2 - det.x1), int(det.y2 - det.y1)),
                source='yolo_local',
            )
            for det in raw
        ]
        self._local_yolo_cache_time = now
        return list(self._local_yolo_cache)

    def _analyze_frame(self, image) -> 'VisionDecision':
        """Single entry for perception: topic YOLO + in-process YOLO + HSV."""
        local = self._local_yolo_detections(image)
        merged = self._fresh_yolo_detections() + local
        keep_hsv = not (
            self.local_yolo is not None and self.config.yolo_replaces_hsv_obstacle)
        return self.vision_detector.analyze(
            image, yolo_detections=merged, color_obstacles=keep_hsv)

    def run_mission(self) -> int:
        state = self.args.start_state
        self.get_logger().info(
            f'auto_mission start_state={state}, no_motion={self.args.no_motion}'
        )
        try:
            while rclpy.ok() and state != 'DONE':
                if not self.guard.runtime_ok():
                    self.failure_reason = 'max runtime exceeded'
                    state = 'STOP'
                self.transition(state)
                state = self._run_state(state)
            self.transition('DONE')
            self._print_summary()
            return 0 if not self.failure_reason else 1
        except KeyboardInterrupt:
            self.failure_reason = 'keyboard interrupt'
            self.get_logger().warn('interrupted; stopping')
            self._stop_with_spin()
            self._print_summary()
            return 130
        except MissionAbort as exc:
            self.failure_reason = str(exc)
            self.get_logger().error(self.failure_reason)
            self._stop_with_spin()
            self._print_summary()
            return 2
        finally:
            self._stop_with_spin()

    def transition(self, state: str) -> None:
        self.current_state = state
        self.state_entry_counts[state] = self.state_entry_counts.get(state, 0) + 1
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)
        print(f'STATE: {state}', flush=True)
        self.get_logger().info(f'STATE: {state}')
        self._event('state_enter', state=state, count=self.state_entry_counts[state])

    def _event(self, event: str, **fields: object) -> None:
        payload: Dict[str, object] = {
            'event': event,
            'state': self.current_state,
            'zone': STATE_ZONES.get(self.current_state, 'UNKNOWN'),
            'runtime_sec': round(time.monotonic() - self.started_at, 3),
        }
        payload.update(fields)
        print(
            'EVENT ' + json.dumps(payload, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    def _load_bridge_entry_reference(self) -> None:
        if not self.config.bridge_entry_reference_enabled:
            return
        path = str(self.config.bridge_entry_reference_image or '').strip()
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            print(f'BRIDGE_ENTRY_REFERENCE_LOAD_FAIL path={path}', flush=True)
            return
        mask = self._bridge_entry_reference_mask_for_image(image)
        if mask is None or float(mask.mean()) <= 1e-6:
            print(f'BRIDGE_ENTRY_REFERENCE_EMPTY path={path}', flush=True)
            return
        self._bridge_entry_reference_mask = mask
        self._bridge_entry_reference_ready = True
        print(
            'BRIDGE_ENTRY_REFERENCE_LOADED '
            f'path={path} green_ratio={float(mask.mean()):.3f}',
            flush=True,
        )

    def _run_state(self, state: str) -> str:
        handlers: Dict[str, Callable[[], str]] = {
            'IDLE': self._state_idle,
            'PREFLIGHT': self._state_preflight,
            'DRIVE_TO_QR': self._state_drive_to_qr,
            'SCAN_QR': self._state_scan_qr,
            'DECIDE_DIRECTION': self._state_decide_direction,
            'CORRIDOR_FOLLOW': self._state_corridor_follow,
            'CORRIDOR_TO_LOOP_TURN': self._state_corridor_to_loop_turn,
            'LOOP_LEFT': lambda: self._state_loop('left'),
            'LOOP_RIGHT': lambda: self._state_loop('right'),
            'CAPTURE_TARGET_IMAGE': self._state_capture_target_image,
            'CALL_LLM_API': self._state_call_llm_api,
            'RETURN_HOME': self._state_return_home,
            'STOP': self._state_stop,
        }
        handler = handlers.get(state)
        if handler is None:
            self.failure_reason = f'unsupported state: {state}'
            return 'STOP'
        return handler()

    def _state_idle(self) -> str:
        return 'PREFLIGHT'

    def _state_preflight(self) -> str:
        timeout = self.config.preflight_timeout
        if not self.args.no_motion:
            if not self._wait_until(
                lambda: self.motion.subscriber_count() > 0,
                timeout,
                spin_step=0.05,
            ):
                self.failure_reason = (
                    f'no subscriber on {self.config.cmd_vel_topic}'
                )
                return 'STOP'
        else:
            self._wait_until(lambda: True, 0.2)

        if not self._wait_until(
            lambda: self.vision.has_fresh_image(self.config.image_stale_timeout),
            timeout,
            spin_step=0.05,
        ):
            self.failure_reason = f'no fresh image on {self.config.image_topic}'
            return 'STOP'

        if not self._wait_until(self.guard.has_voltage, timeout, spin_step=0.05):
            message = f'no voltage sample on {self.config.voltage_topic}'
            if self.args.no_motion:
                self.get_logger().warn(message)
            else:
                self.failure_reason = message
                return 'STOP'

        if self.guard.latest_voltage is not None:
            print(f'PREFLIGHT_VOLTAGE: {self.guard.latest_voltage:.2f}V', flush=True)

        if self.motion_state is not None:
            if not self._wait_until(
                self.motion_state.has_odom, timeout, spin_step=0.05
            ):
                message = f'no odom sample on {self.config.odom_topic}'
                if self.args.no_motion:
                    self.get_logger().warn(message)
                else:
                    self.failure_reason = message
                    return 'STOP'
            else:
                print(
                    'PREFLIGHT_MOTION_STATE: '
                    f'odom={int(self.motion_state.has_odom())} '
                    f'imu={int(self.motion_state.has_imu())}',
                    flush=True,
                )
        return 'DRIVE_TO_QR'

    def _state_drive_to_qr(self) -> str:
        if self.motion_state is not None:
            self.motion_state.reset()
            self.motion_state.start_recording()
        if self._map_route_ready():
            self._reset_map_frame()
            result = self._execute_map_route('A_TO_QR', allow_qr=True)
            return result or 'SCAN_QR'
        if self._fixed_route_ready():
            result = self._execute_fixed_route('A_TO_QR', allow_qr=True)
            return result or 'SCAN_QR'
        follower = LaneFollower(
            self.vision_detector, 'black',
            self.config.lane_follow_config(bias=0.0, side_mode='center'))
        follower.reset()
        search_follower = LaneFollower(
            self.vision_detector, 'black',
            self.config.lane_follow_config(
                bias=0.0, side_mode='center',
                roi_y_ratio=self.config.drive_recover_search_roi,
                roi_height_ratio=1.0 - self.config.drive_recover_search_roi))
        search_follower.reset()
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.drive_to_qr_timeout
        self._last_avoid_turn = 0.0
        self._last_avoid_time = 0.0
        lane_lost_frames = 0
        _recover_log_t: float = 0.0
        _sweep_start: Optional[float] = None
        _qr_check_skip = 0
        print(
            'DRIVE_TO_QR: black-line follow, obstacle avoid, '
            f'max_distance={self.config.drive_to_qr_max_distance:.2f}m '
            f'timeout={self.config.drive_to_qr_timeout:.1f}s',
            flush=True,
        )
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during DRIVE_TO_QR'
                return 'STOP'
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            _qr_check_skip += 1
            if _qr_check_skip % 15 == 0 and _pyzbar is not None:
                result = self._drive_check_qr_area(image)
                if result is not None:
                    return result
            decision = self._analyze_frame(image)
            if decision.obstacle_danger and self.config.obstacle_avoid_enabled:
                self._handle_drive_obstacle(image, decision, interval)
                lane_lost_frames = 0
                _sweep_start = None
                continue
            cmd = follower.compute(image)
            if cmd.lane_found:
                lane_lost_frames = 0
                _sweep_start = None
                self.motion.publish(cmd.linear, cmd.angular)
            else:
                lane_lost_frames += 1
                if lane_lost_frames < max(1, self.config.return_line_lost_grace_frames):
                    self._lane_lost_step(follower, interval)
                    continue
                recently_avoided = (
                    time.monotonic() - self._last_avoid_time
                    < self.config.recover_after_avoid_s
                )
                if recently_avoided:
                    scmd = search_follower.compute(image)
                    if scmd.lane_found:
                        _sweep_start = None
                        self.motion.publish(self.config.drive_lost_linear, scmd.angular)
                        _now = time.time()
                        if _now - _recover_log_t >= 0.5:
                            print(f'DRIVE_TO_QR_RELINE offset={scmd.offset_norm:.2f}', flush=True)
                            _recover_log_t = _now
                        self._spin_sleep(interval)
                        continue
                    if _sweep_start is None:
                        _sweep_start = time.monotonic()
                    swept_long = (
                        time.monotonic() - _sweep_start
                        >= self.config.recover_sweep_max_s
                    )
                    if swept_long or abs(self._last_avoid_turn) <= 1e-3:
                        sweep = 0.0
                    else:
                        sweep = -math.copysign(
                            self.config.drive_lost_angular, self._last_avoid_turn)
                    self.motion.publish(self.config.drive_lost_linear, sweep)
                    _now = time.time()
                    if _now - _recover_log_t >= 0.5:
                        _dir = 'R' if sweep < 0 else 'L' if sweep > 0 else 'straight'
                        print(f'DRIVE_TO_QR_SWEEP dir={_dir}', flush=True)
                        _recover_log_t = _now
                    self._spin_sleep(interval)
                    continue
                print('DRIVE_TO_QR_LINE_LOST: switching to SCAN_QR', flush=True)
                self._stop_with_spin()
                return 'SCAN_QR'
            if (self.motion_state is not None
                    and self.motion_state.traveled_distance()
                    >= self.config.drive_to_qr_max_distance):
                print(
                    'DRIVE_TO_QR_MAX_DISTANCE '
                    f'dist={self.motion_state.traveled_distance():.2f}m',
                    flush=True,
                )
                self._event(
                    'drive_to_qr_max_distance',
                    distance=round(self.motion_state.traveled_distance(), 3),
                )
                self._stop_with_spin()
                return 'SCAN_QR'
            self._spin_sleep(interval)
        print('DRIVE_TO_QR_TIMEOUT: switching to SCAN_QR', flush=True)
        self._stop_with_spin()
        return 'SCAN_QR'

    def _handle_drive_obstacle(self, image, decision, interval: float) -> None:
        """React to a dark-blue obstacle during approach: log, optionally save, turn."""
        self.obstacle_event_count += 1
        print(
            'VISION_OBSTACLE '
            f'zone={decision.obstacle_zone} '
            f'area={decision.obstacle_area_ratio:.4f}',
            flush=True,
        )
        self._event(
            'vision_obstacle',
            zone=decision.obstacle_zone,
            area_ratio=round(decision.obstacle_area_ratio, 5),
            count=self.obstacle_event_count,
            source='dark_blue_opencv_or_yolo',
        )
        if self.args.save_debug_images:
            overlay = self.vision_detector.draw_debug(image, decision)
            self._save_debug_image('obstacle_debug', overlay)
        zone = decision.obstacle_zone
        if zone in ('left', 'right'):
            # Obstacle is already off to the side: it no longer blocks the path
            # ahead. DRIVE FORWARD to translate past it (only a gentle residual
            # steer away for clearance) instead of pivoting beside it — at the low
            # turn speed the car otherwise stays stuck rotating next to the cone
            # while it remains in view (field 6-17).
            linear = self.config.obstacle_pass_linear
            turn = self._avoidance_turn_for_zone(zone) * 0.4
        else:
            # Obstacle dead ahead (center): steer hard away while creeping so it
            # moves off the path.
            linear = self.config.obstacle_turn_linear
            turn = self._avoidance_turn_for_zone(zone)
        self._last_avoid_turn = turn
        self._last_avoid_time = time.monotonic()
        self.motion.publish(linear, turn)
        self._spin_sleep(interval)

    def _drive_check_qr_area(self, image: np.ndarray) -> Optional[str]:
        """Fast pyzbar-only QR check during drive; returns SCAN_QR if area >= threshold."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = _pyzbar.decode(gray)
        for b in barcodes:
            x, y, bw, bh = b.rect
            area_ratio = (bw * bh) / (w * h)
            if area_ratio >= self.config.drive_qr_area_threshold:
                self.qr_content = b.data.decode()
                self.qr_method = 'pyzbar_drive'
                print(f'QR_DRIVE_AREA_OK text={self.qr_content} area={area_ratio:.4f}', flush=True)
                self._event('qr_drive_area_ok', content=self.qr_content, area=round(area_ratio, 4))
                self._stop_with_spin()
                return 'SCAN_QR'
        return None

    def _drive_probe_qr(self, image, elapsed: float) -> Optional[str]:
        """Probe for the QR board; steer toward it and stop when readable.

        Returns a next-state string when the approach should end, else None to
        keep cruising. Uses QrDecoder points (board localizable even before the
        code decodes) plus area/centering gates so we stop AT the board, not on
        a dead timer.
        """
        qr = self.qr_decoder.detect(
            image, time_budget_s=self.config.qr_detect_time_budget_s
        )
        center, area_ratio = self._qr_points_center_area(qr.points, image)
        if qr.text:
            self.qr_content = qr.text
            self.qr_method = qr.method
            print(
                f'QR_CANDIDATE_DURING_DRIVE: {qr.text} '
                f'area={area_ratio:.3f}',
                flush=True,
            )
            self._event(
                'qr_candidate',
                method=qr.method,
                content=qr.text,
                area_ratio=round(area_ratio, 4),
            )
            self._confirm_direction(qr.text)
        if center is not None and area_ratio >= self.config.qr_min_area_ratio:
            h, w = image.shape[:2]
            offset = (center[0] - w / 2.0) / (w / 2.0)  # -1 left .. +1 right
            half_band = self.config.qr_center_band / 2.0
            if abs(offset) <= half_band:
                if not qr.text:
                    # Board localized + centred but not decoded. This is almost
                    # always MOTION BLUR: a stationary frame decodes the QR even
                    # at code-area 0.008, but a moving frame fails even point-blank.
                    # Creeping forward drove the car straight INTO the board
                    # (field 6-14). So STOP here and hand to SCAN_QR, which decodes
                    # from a sharp stationary frame with the heavy multi-attempt
                    # pipeline.
                    print(
                        f'QR_BOARD_STOP_FOR_SCAN area={area_ratio:.3f} offset={offset:.2f}',
                        flush=True,
                    )
                    self._event(
                        'qr_board_stop_for_scan',
                        area_ratio=round(area_ratio, 4),
                        offset=round(offset, 3),
                    )
                    self._stop_with_spin()
                    return 'SCAN_QR'
                print(
                    f'QR_BOARD_REACHED area={area_ratio:.3f} offset={offset:.2f}',
                    flush=True,
                )
                self._event(
                    'qr_board_reached',
                    area_ratio=round(area_ratio, 4),
                    offset=round(offset, 3),
                )
                self._stop_with_spin()
                self._reanchor_map_frame(self.config.qr_reanchor_waypoint)
                return 'DECIDE_DIRECTION' if self.direction_locked else 'SCAN_QR'
            # Visible but off-center: steer toward it while creeping.
            turn = -self.config.qr_seek_angular if offset > 0 else self.config.qr_seek_angular
            self.motion.publish(self.config.cruise_linear * 0.6, turn)
            self._spin_sleep(1.0 / self.config.rate_hz)
            return None
        # QR not visible: hand back to the caller (black-line follow / cruise).
        # The line leads to the corridor where the QR board sits, so the timed
        # "seek right" crutch is dropped — it would fight line-following.
        return None

    def _qr_points_center_area(self, points, image):
        """Return ((cx, cy), area_ratio) from QR corner points, or (None, 0.0)."""
        if points is None:
            return None, 0.0
        try:
            pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        except Exception:
            return None, 0.0
        if pts.shape[0] < 3:
            return None, 0.0
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
        area = float(cv2.contourArea(pts.astype(np.float32)))
        h, w = image.shape[:2]
        return (cx, cy), area / float(max(1, w * h))

    def _resolve_qr_direction(self, text: str) -> Tuple[str, str]:
        """Map decoded QR text -> ('left'/'right'/'', source).

        Competition QR content is unknown ahead of time (practice field decodes
        to e.g. "1334"), so an explicit config map `qr_direction_map` is checked
        FIRST; on a miss we fall back to keyword parsing (clockwise/左转/...).
        """
        key = (text or '').strip()
        mapped = self.config.qr_direction_map.get(key)
        if mapped in ('left', 'right'):
            return mapped, 'config_map'
        instruction = parse_qr_instruction(text)
        return instruction.direction, instruction.source

    def _confirm_direction(self, text: str) -> bool:
        """Accumulate consecutive same-direction decodes; lock once stable.

        Anti-jump (rule 不可跳变): a single frame is never trusted. We require
        `qr_confirm_count` consecutive decodes that all parse to the SAME
        direction. On reaching it we freeze direction/qr_content for good; any
        later decode (even a conflicting one) is ignored.
        """
        if self.direction_locked:
            return True
        direction, source = self._resolve_qr_direction(text)
        if direction not in ('left', 'right'):
            return False  # undecodable frame: ignore, do not reset the streak
        if direction == self._dir_vote:
            self._dir_vote_count += 1
        else:
            self._dir_vote = direction
            self._dir_vote_count = 1
        if self._dir_vote_count >= self.config.qr_confirm_count:
            self.direction = direction
            self.direction_source = source
            self.qr_content = text
            self.direction_locked = True
            print(
                f'DIRECTION_LOCKED: {direction} '
                f'after {self._dir_vote_count} consistent reads',
                flush=True,
            )
            self._event(
                'direction_locked',
                direction=direction,
                votes=self._dir_vote_count,
                content=text,
            )
            return True
        return False

    def _state_scan_qr(self) -> str:
        self._stop_with_spin()
        if self.args.mock_qr_content:
            self.qr_content = self.args.mock_qr_content
            self.qr_method = 'mock'
            print(f'QR_DETECTED: {self.qr_content}', flush=True)
            print('QR_METHOD: mock', flush=True)
            self._event('qr_detected', method='mock', content=self.qr_content)
            return 'DECIDE_DIRECTION'

        deadline = time.monotonic() + self.config.scan_qr_timeout
        last_failure_save = 0.0
        last_qr_text = ''
        qr_text_count = 0
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during SCAN_QR'
                return 'STOP'
            rclpy.spin_once(self, timeout_sec=0.05)
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                continue
            qr = self.qr_decoder.detect(image)
            if qr.text:
                self.qr_method = qr.method
                print(f'QR_DETECTED: {qr.text}', flush=True)
                print(f'QR_METHOD: {qr.method}', flush=True)
                self._event('qr_detected', method=qr.method, content=qr.text)
                if qr.points is not None:
                    print(f'QR_POINTS: {qr.points.reshape(-1, 2).tolist()}', flush=True)
                if self.args.save_debug_images:
                    self._save_debug_image('qr_success', image)
                # Text consistency counter: escape infinite loop even when
                # direction is unparsable (e.g. QR content is just "3").
                if qr.text == last_qr_text:
                    qr_text_count += 1
                else:
                    last_qr_text = qr.text
                    qr_text_count = 1
                if qr_text_count >= self.config.qr_confirm_count:
                    self.qr_content = qr.text
                    print(f'QR_FINALIZED: {qr.text} after {qr_text_count} consistent reads', flush=True)
                    self._event('qr_finalized', content=qr.text, reads=qr_text_count)
                    if self.direction_locked:
                        pass
                    elif self._dir_vote and self._dir_vote_count >= 1:
                        self.direction = self._dir_vote
                        self.direction_source = 'locked'
                        self.direction_locked = True
                        print(f'QR_DIRECTION_FALLBACK: {self._dir_vote}', flush=True)
                    return 'DECIDE_DIRECTION'
                if self._confirm_direction(qr.text):
                    return 'DECIDE_DIRECTION'
                continue
            if self.args.save_debug_images and time.monotonic() - last_failure_save > 1.0:
                last_failure_save = time.monotonic()
                self._save_debug_image('qr_last_failure', image)

        if self.direction_locked:
            return 'DECIDE_DIRECTION'
        if self._dir_vote and self._dir_vote_count >= 1:
            # Timed out before reaching qr_confirm_count: fall back to the
            # majority candidate we did see, lock it, and proceed (degraded).
            self.direction = self._dir_vote
            self.direction_locked = True
            print(
                f'QR_TIMEOUT_LOCK: {self._dir_vote} '
                f'with only {self._dir_vote_count} reads (< qr_confirm_count)',
                flush=True,
            )
            return 'DECIDE_DIRECTION'
        self.failure_reason = 'QR scan timeout'
        return 'STOP'

    def _state_decide_direction(self) -> str:
        if self.args.force_direction:
            self.direction = self.args.force_direction
            self.direction_source = 'force_direction'
        elif self.direction_locked and self.direction in ('left', 'right'):
            # Direction already frozen in SCAN_QR; never re-parse (不可跳变).
            if not self.direction_source:
                self.direction_source = 'locked'
        else:
            self.direction, self.direction_source = self._resolve_qr_direction(
                self.qr_content)
        if self.direction not in ('left', 'right'):
            self.failure_reason = (
                f'cannot parse direction from QR: {self.qr_content!r} '
                f'(add it to qr_direction_map, e.g. {{"{self.qr_content.strip()}": "right"}})'
            )
            return 'STOP'
        print(f'QR_CONTENT: {self.qr_content}', flush=True)
        print(f'DIRECTION: {self.direction}', flush=True)
        print(f'DIRECTION_SOURCE: {self.direction_source}', flush=True)
        self._event(
            'direction_decided',
            direction=self.direction,
            source=self.direction_source,
        )
        # Always go through the vision bridge (backup + corridor search).
        # The map/fixed-route path for QR_TO_CORRIDOR was removed because the
        # actual waypoints don't match the field layout required for backup.
        # CORRIDOR_FOLLOW (below) still uses the map/fixed route for the corridor.
        if self.config.qr_align_after_scan_enabled:
            if not self._align_after_qr_scan():
                return 'STOP'
        if self._bridge_qr_to_corridor():
            self._settle_transition('A_HALL->B_CORRIDOR')
            return 'CORRIDOR_FOLLOW'
        else:
            self.failure_reason = 'bridge_qr_to_corridor timeout'
            return 'STOP'


    def _bridge_qr_to_corridor(self) -> bool:
        """After QR scan: left-turn to parallel forbidden zone, then straight to find black line."""
        if self.args.no_motion:
            self._event('bridge_qr_corridor_dryrun')
            return True

        self._event('bridge_qr_corridor_start', timeout=self.config.bridge_timeout)
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.bridge_timeout

        yellow_lower = np.array([20, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 80, 80])

        ref_edge_x = None
        ref_saved = False
        prev_edge_x = None
        stability_count = 0

        print('BRIDGE_QR: forward+left turn, align to forbidden zone', flush=True)

        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return False

            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue

            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

            right_roi = yellow_mask[int(h*0.5):, int(w*0.6):]
            edge_x = None
            for col in range(right_roi.shape[1]):
                if np.any(right_roi[:, col] > 0):
                    edge_x = col + int(w * 0.6)
                    break

            if not ref_saved:
                if edge_x is not None:
                    if prev_edge_x is not None and abs(edge_x - prev_edge_x) <= 3:
                        stability_count += 1
                    else:
                        stability_count = 0
                    prev_edge_x = edge_x
                    if stability_count >= 5:
                        ref_edge_x = edge_x
                        ref_saved = True
                        if self.args.save_debug_images:
                            self._save_debug_image('bridge_qr_reference', image)
                        print(f'BRIDGE_QR_REFERENCE edge_x={edge_x}/{w} ratio={edge_x/w:.2f}', flush=True)
                        self._event('bridge_qr_reference', edge_x=edge_x, ratio=round(edge_x/w, 3))
                        continue
                self.motion.publish(0.1, 0.15)
            else:
                error = (edge_x - ref_edge_x) if edge_x is not None else 0.0
                if abs(error) > 15:
                    correction = error * 0.008
                    correction = max(-0.12, min(0.12, correction))
                else:
                    correction = 0.0

                black_mask = cv2.inRange(hsv, black_lower, black_upper)
                center_roi = black_mask[int(h*0.75):, int(w*0.25):int(w*0.75)]
                black_ratio = np.mean(center_roi > 0) if center_roi.size > 0 else 0.0

                if black_ratio > 0.04:
                    print(f'BRIDGE_QR_BLACK_LINE ratio={black_ratio:.3f}', flush=True)
                    self._event('bridge_qr_black_line', ratio=round(black_ratio, 3))
                    self._stop_with_spin()
                    print('BRIDGE_QR: turn right 90deg into corridor', flush=True)
                    self._event('bridge_qr_turn_right')
                    self._rotate_relative_arc(
                        -math.radians(60),
                        math.radians(5),
                        self.config.shunt_turn_timeout,
                    )
                    self._stop_with_spin()
                    self._settle_transition('A_HALL->B_CORRIDOR')
                    return True

                self.motion.publish(0.1, correction)

            self._spin_sleep(interval)

        self.motion.publish(0.0, 0.0)
        self._event('bridge_qr_corridor_timeout')
        return False

    def _find_bridge_corridor_target(
        self,
        image: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """Return (offset_norm, area_ratio, bottom_ratio) for the solid corridor blob.

        The two yellow-white side walls are thin striped structures. A strong
        erosion suppresses those stripes while preserving the thicker solid yellow
        corridor entrance.
        """
        h, w = image.shape[:2]
        roi_y = int(h * self.config.bridge_target_roi_y_ratio)
        roi = image[roi_y:, :]
        mask, _ = self.vision_detector.color_mask(roi, self.config.lane_color, roi_y_ratio=None)
        if self.config.bridge_yellow_erode_px > 0:
            k = int(self.config.bridge_yellow_erode_px)
            mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[Tuple[float, float, float, float]] = None
        min_area = self.config.bridge_target_min_area_ratio * w * h
        center_gate = max(0.05, min(1.0, self.config.bridge_target_center_gate_ratio))
        roi_h = max(1, roi.shape[0])
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            cx = x + bw / 2.0
            bottom = y + bh
            offset_norm = (cx - (w / 2.0)) / max(1.0, w / 2.0)
            if abs(offset_norm) > center_gate:
                continue
            area_ratio = area / float(max(1, w * h))
            bottom_ratio = bottom / float(roi_h)
            score = area_ratio * 2.2 + bottom_ratio - abs(offset_norm) * 1.6
            if best is None or score > best[0]:
                best = (score, offset_norm, area_ratio, bottom_ratio)

        if best is None:
            return None
        return best[1], best[2], best[3]

    def _bridge_yellow_seen(self, image: np.ndarray) -> Tuple[bool, Optional[float]]:
        h = image.shape[0]
        roi_y = int(h * self.config.bridge_target_roi_y_ratio)
        roi = image[roi_y:, :]
        mask, _ = self.vision_detector.color_mask(roi, self.config.lane_color, roi_y_ratio=None)
        area_ratio = float((mask > 0).sum()) / float(max(1, mask.size))
        if area_ratio < self.config.bridge_yellow_detect_min_ratio:
            return False, None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return True, None
        contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(contour)
        w = roi.shape[1]
        cx = x + bw / 2.0
        offset_norm = (cx - (w / 2.0)) / max(1.0, w / 2.0)
        return True, float(offset_norm)

    def _bridge_entry_reference_mask_for_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        x0 = int(w * self.config.bridge_entry_reference_roi_x_ratio)
        y0 = int(h * self.config.bridge_entry_reference_roi_y_ratio)
        rw = int(w * self.config.bridge_entry_reference_roi_w_ratio)
        rh = int(h * self.config.bridge_entry_reference_roi_h_ratio)
        x1 = min(w, x0 + max(1, rw))
        y1 = min(h, y0 + max(1, rh))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = image[y0:y1, x0:x1]
        mask, _ = self.vision_detector.color_mask(roi, self.config.green_color, roi_y_ratio=None)
        small = cv2.resize(mask, (64, 48), interpolation=cv2.INTER_AREA)
        return (small > 0).astype(np.uint8)

    def _bridge_entry_reference_score(self, image: np.ndarray) -> Optional[float]:
        if not self._bridge_entry_reference_ready or self._bridge_entry_reference_mask is None:
            return None
        current = self._bridge_entry_reference_mask_for_image(image)
        if current is None:
            return None
        ref = self._bridge_entry_reference_mask
        inter = np.logical_and(ref > 0, current > 0).sum()
        union = np.logical_or(ref > 0, current > 0).sum()
        if union <= 0:
            return 0.0
        return float(inter) / float(union)

    def _bridge_entry_reference_gate(self, interval: float) -> bool:
        if not self.config.bridge_entry_reference_enabled or not self._bridge_entry_reference_ready:
            return True
        deadline = time.monotonic() + max(0.1, self.config.bridge_entry_reference_timeout)
        best_score = 0.0
        while time.monotonic() < deadline and self.guard.runtime_ok():
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            score = self._bridge_entry_reference_score(image)
            if score is None:
                self._spin_sleep(interval)
                continue
            best_score = max(best_score, score)
            if score >= self.config.bridge_entry_reference_min_score:
                print(
                    'BRIDGE_ENTRY_REFERENCE_MATCH '
                    f'score={score:.3f}',
                    flush=True,
                )
                self._event(
                    'bridge_entry_reference_match',
                    score=round(score, 4),
                )
                return True
            self.motion.publish(
                max(
                    self.config.bridge_turn_linear,
                    self.config.bridge_black_search_linear,
                    self.config.cruise_linear,
                ),
                0.0,
            )
            self._spin_sleep(interval)
        print(
            'BRIDGE_ENTRY_REFERENCE_TIMEOUT '
            f'best_score={best_score:.3f}',
            flush=True,
        )
        self._event(
            'bridge_entry_reference_timeout',
            best_score=round(best_score, 4),
        )
        return True

    def _bridge_black_reentry_turn(self, interval: float) -> None:
        angular = -abs(self.config.bridge_black_reentry_turn_angular)
        linear = max(
            self.config.bridge_turn_linear,
            self.config.bridge_black_reentry_turn_linear,
        )
        target_rad = math.radians(self.config.bridge_black_reentry_turn_angle_deg)
        timeout_s = max(0.8, target_rad / max(0.05, abs(angular)) + 0.4)
        print(
            'BRIDGE_BLACK_REENTRY_TURN '
            f'angle_deg={self.config.bridge_black_reentry_turn_angle_deg:.1f} '
            f'linear={linear:.2f} angular={angular:+.2f}',
            flush=True,
        )
        self._event(
            'bridge_black_reentry_turn_start',
            angle_deg=round(self.config.bridge_black_reentry_turn_angle_deg, 1),
            linear=round(linear, 3),
            angular=round(angular, 3),
        )
        if self.motion_state is not None:
            self.motion_state.reset_yaw()
        deadline = time.monotonic() + timeout_s
        turned = 0.0
        while time.monotonic() < deadline and self.guard.runtime_ok():
            if self.motion_state is not None:
                turned = abs(self.motion_state.integrated_yaw())
                if target_rad - turned <= math.radians(4.0):
                    break
            self.motion.publish(linear, angular)
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        self._event(
            'bridge_black_reentry_turn_done',
            turned_deg=round(math.degrees(turned), 1),
        )
        deadline = time.monotonic() + max(0.0, self.config.bridge_black_reentry_forward_s)
        while time.monotonic() < deadline and self.guard.runtime_ok():
            self.motion.publish(self.config.bridge_align_linear, 0.0)
            self._spin_sleep(interval)

    def _bridge_post_backup_pulse(self, interval: float, search_dir: float) -> None:
        """Nudge forward-left/right after reversing so the corridor mouth re-enters view."""
        pulse_s = max(0.0, self.config.bridge_post_backup_forward_s)
        if pulse_s <= 0.0:
            return
        linear = max(self.config.bridge_turn_linear, self.config.bridge_post_backup_linear)
        angular = search_dir * abs(self.config.bridge_post_backup_angular)
        print(
            'BRIDGE_POST_BACKUP_PULSE '
            f'linear={linear:.2f} angular={angular:+.2f} '
            f'duration={pulse_s:.2f}s',
            flush=True,
        )
        self._event(
            'bridge_post_backup_pulse',
            linear=linear,
            angular=angular,
            duration_s=round(pulse_s, 2),
        )
        deadline = time.monotonic() + pulse_s
        while time.monotonic() < deadline and self.guard.runtime_ok():
            self.motion.publish(linear, angular)
            self._spin_sleep(interval)
        # Hand control directly into the next bridge-search step without an
        # inserted stop; the user wants one continuous straight push here.

    def _bridge_backup(self, line_follower: 'LaneFollower', interval: float) -> None:
        """Reverse a FIXED DISTANCE away from the QR board (odom-metered).

        At the SCAN_QR stop the car is jammed ~0.3m from the board, very close to
        the orange task-release pad, and must back up to the central yellow
        corridor entrance before threading it. We can't stop the reverse on
        "saw the black line": the line runs right under the car (it drove in on
        it) so it is visible from frame 1 — stopping on it barely reverses at all
        and the forward bridge then drives into the board (field 6-17). So we
        reverse a fixed distance by odom; if odom is unavailable, fall back to a
        purely timed reverse. line_follower param kept for signature stability.
        """
        dist = self.config.bridge_backup_distance_m
        print(
            'BRIDGE_BACKUP: reversing '
            f'(v={self.config.bridge_backup_linear:.2f} '
            f'dist={dist:.2f}m max={self.config.bridge_backup_max_s:.1f}s)',
            flush=True,
        )
        self._event(
            'bridge_backup_start',
            linear=self.config.bridge_backup_linear,
            distance_m=dist,
            max_s=self.config.bridge_backup_max_s,
        )
        deadline = time.monotonic() + self.config.bridge_backup_max_s
        start = (
            self.motion_state.traveled_distance()
            if self.motion_state is not None else None
        )
        # Record start heading so we can REPORT how far/which way the reverse arc
        # actually swung the car's yaw. On Ackermann the heading-change direction
        # while reversing is not obvious to predict (it inverts vs forward), so we
        # measure it from odom rather than guess the angular sign (field 6-17).
        yaw_start = (
            self.motion_state.absolute_yaw()
            if self.motion_state is not None else None
        )
        reached = False
        while time.monotonic() < deadline and self.guard.runtime_ok():
            if start is not None:
                traveled = abs(self.motion_state.traveled_distance() - start)
                if traveled >= dist:
                    reached = True
                    break
            # Reverse on an ARC: the angular term swings the car's heading off the
            # board / right-side grid toward the central corridor while it backs up
            # (Ackermann: heading changes only while moving). field 6-17.
            self.motion.publish(
                self.config.bridge_backup_linear,
                self.config.bridge_backup_angular,
            )
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        # Report the actual yaw swing so we can confirm (next field run) whether
        # the reverse arc turned the heading toward the central corridor or the
        # wrong way — turning bridge_backup_angular's sign into a data decision,
        # not a guess. angle_diff>0 = turned LEFT (CCW).
        yaw_turn_deg = None
        if yaw_start is not None and self.motion_state is not None:
            yaw_end = self.motion_state.absolute_yaw()
            if yaw_end is not None:
                yaw_turn = angle_diff(yaw_end, yaw_start)
                yaw_turn_deg = math.degrees(yaw_turn)
                print(
                    'BRIDGE_BACKUP: heading turned '
                    f'{yaw_turn_deg:+.0f}deg ({"LEFT" if yaw_turn > 0 else "RIGHT"})',
                    flush=True,
                )
        if start is not None:
            traveled = abs(self.motion_state.traveled_distance() - start)
            print(
                f'BRIDGE_BACKUP: done traveled={traveled:.2f}m '
                f'reached={reached}',
                flush=True,
            )
            self._event(
                'bridge_backup_done',
                traveled_m=round(traveled, 3),
                reached=reached,
                yaw_turn_deg=(round(yaw_turn_deg, 1)
                              if yaw_turn_deg is not None else None),
            )
        else:
            print('BRIDGE_BACKUP: done (timed, no odom)', flush=True)
            self._event('bridge_backup_done', traveled_m=None, reached=False)
        # Settle so motion blur from the reverse clears before the bridge reads a
        # frame (same rationale as the inchworm scan settle).
        self._spin_sleep(self.config.drive_scan_settle_s)

    def _green_floor_ratio(self, image) -> float:
        """Fraction of the lower ROI that matches the green (clinic) profile."""
        try:
            mask, _ = self.vision_detector.color_mask(
                image, self.config.green_color, roi_y_ratio=self.config.lane_roi_y_ratio)
        except Exception:
            return 0.0
        return float((mask > 0).sum()) / float(max(1, mask.size))

    def _align_after_qr_scan(self) -> bool:
        if self.motion_state is None:
            self.failure_reason = 'QR align requested but motion_state is unavailable'
            return False
        pose = self._current_map_pose()
        if pose is None:
            self.failure_reason = 'QR align requested but map pose is unavailable'
            return False
        target_heading = math.radians(self.config.qr_exit_heading_deg)
        target_delta = angle_diff(target_heading, pose.yaw)
        print(
            'QR_ALIGN_HEADING_START '
            f'target_deg={self.config.qr_exit_heading_deg:.1f} '
            f'current_deg={math.degrees(pose.yaw):.1f} '
            f'delta_deg={math.degrees(target_delta):.1f} '
            f'mode={self.config.ackermann_turn_mode}',
            flush=True,
        )
        self._event(
            'qr_align_heading_start',
            target_deg=round(self.config.qr_exit_heading_deg, 1),
            current_deg=round(math.degrees(pose.yaw), 1),
            delta_deg=round(math.degrees(target_delta), 1),
            mode=self.config.ackermann_turn_mode,
        )
        if abs(target_delta) <= self.config.qr_align_heading_tol:
            print('QR_ALIGN_HEADING_DONE reason=already_aligned', flush=True)
            self._event('qr_align_heading_done', reason='already_aligned')
            return True
        if self.config.ackermann_turn_mode == 'shunt':
            result = execute_shunt_turn(
                target_delta,
                motion_state=self.motion_state,
                publish_velocity=self.motion.publish,
                spin_sleep=lambda: self._spin_sleep(1.0 / self.config.rate_hz),
                forward_linear=self.config.shunt_forward_linear,
                reverse_linear=self.config.shunt_reverse_linear,
                angular=self.config.fixed_route_turn_angular,
                heading_tol=self.config.qr_align_heading_tol,
                timeout=self.config.shunt_turn_timeout,
                pulse_s=self.config.shunt_pulse_s,
                stop_s=self.config.shunt_stop_s,
                ok=rclpy.ok,
                runtime_ok=self.guard.runtime_ok,
            )
            self.motion.publish(0.0, 0.0)
            print(
                'QR_ALIGN_HEADING_DONE '
                f'reason={result.reason} '
                f'turned_deg={math.degrees(result.turned_rad):.1f} '
                f'elapsed={result.elapsed_s:.1f} cycles={result.cycles}',
                flush=True,
            )
            self._event(
                'qr_align_heading_done',
                reason=result.reason,
                turned_deg=round(math.degrees(result.turned_rad), 1),
                elapsed_s=round(result.elapsed_s, 1),
                cycles=result.cycles,
            )
            if not result.ok:
                self.failure_reason = 'QR align heading timeout'
                return False
            return True
        self._rotate_relative_arc(
            target_delta,
            self.config.qr_align_heading_tol,
            self.config.shunt_turn_timeout,
        )
        print('QR_ALIGN_HEADING_DONE reason=arc_complete', flush=True)
        self._event('qr_align_heading_done', reason='arc_complete')
        return True

    def _rotate_relative_arc(self, target_delta: float, heading_tol: float, timeout: float) -> None:
        direction = 1.0 if target_delta >= 0.0 else -1.0
        interval = 1.0 / self.config.rate_hz
        self.motion_state.reset_yaw()
        deadline = time.monotonic() + max(0.1, timeout)
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return
            turned = self.motion_state.integrated_yaw()
            if abs(target_delta) - abs(turned) <= heading_tol:
                break
            self.motion.publish(
                self.config.fixed_route_turn_linear,
                direction * self.config.fixed_route_turn_angular,
            )
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)

    def _lane_lost_step(self, follower, interval: float) -> None:
        """Behavior when the yellow lane is not visible."""
        if self.config.lane_lost_behavior == 'stop':
            self.motion.publish(0.0, 0.0)
        else:  # creep slowly straight to re-acquire
            self.motion.publish(self.config.lane_base_linear * 0.5, 0.0)
        self._spin_sleep(interval)

    def _state_corridor_follow(self) -> str:
        """Zone B: follow the straight yellow corridor until distance/green/timeout."""
        if self._map_route_ready() and self.route_map.route('B_CORRIDOR'):
            result = self._execute_map_route('B_CORRIDOR', lane_color=self.config.lane_color)
            if result == 'STOP':
                return 'STOP'
            self._event('corridor_follow_done', exit_reason='map_route')
            self._stop_with_spin()
            self._settle_transition('B_CORRIDOR->C_RING')
            return 'CORRIDOR_TO_LOOP_TURN'
        if self._fixed_route_ready() and 'B_CORRIDOR' in self.config.fixed_routes:
            result = self._execute_fixed_route('B_CORRIDOR', lane_color=self.config.lane_color)
            if result == 'STOP':
                return 'STOP'
            self._event('corridor_follow_done', exit_reason='fixed_route')
            self._stop_with_spin()
            self._settle_transition('B_CORRIDOR->C_RING')
            return 'CORRIDOR_TO_LOOP_TURN'
        if not self.config.lane_follow_enabled:
            return 'CORRIDOR_TO_LOOP_TURN'
        follower = LaneFollower(
            self.vision_detector, self.config.lane_color,
            self.config.lane_follow_config(
                bias=0.0, side_mode='center',
                erode_px=self.config.lane_mask_erode_px))
        follower.reset()
        corridor_start_dist = 0.0
        if self.motion_state is not None:
            self.motion_state.reset_yaw()
            corridor_start_dist = self.motion_state.distance_marker()
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.corridor_timeout
        _last_obstacle_probe_corr: float = 0.0
        _last_corr_save: float = 0.0
        green_frames = 0
        exit_reason = 'timeout'
        print(
            f'CORRIDOR_FOLLOW: yellow lane, length={self.config.corridor_length_m:.1f}m '
            f'timeout={self.config.corridor_timeout:.0f}s',
            flush=True,
        )
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during CORRIDOR_FOLLOW'
                return 'STOP'
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            # Field diagnostics: periodically save the corridor view + green ratio
            # so the solid yellow lane vs the yellow grid walls can be told apart.
            if self.args.save_debug_images and (time.monotonic() - _last_corr_save) > 0.7:
                _last_corr_save = time.monotonic()
                print(
                    f'CORRIDOR_DIAG green_ratio={self._green_floor_ratio(image):.3f}',
                    flush=True,
                )
                self._save_debug_image('corridor', image)
            if (
                self.motion_state is not None
                and self.motion_state.traveled_distance() - corridor_start_dist
                >= self.config.corridor_length_m
            ):
                exit_reason = 'distance'
                print('CORRIDOR_DONE reason=distance', flush=True)
                break
            green_ratio = self._green_floor_ratio(image)
            if green_ratio >= self.config.green_exit_ratio:
                green_frames += 1
            else:
                green_frames = 0
            if green_frames >= self.config.green_exit_frames:
                exit_reason = 'green_floor'
                print(
                    f'CORRIDOR_DONE reason=green_floor green_ratio={green_ratio:.3f} '
                    f'frames={green_frames}',
                    flush=True,
                )
                break
            _now_corr = time.monotonic()
            if self.config.obstacle_avoid_enabled and _now_corr - _last_obstacle_probe_corr >= 0.3:
                _last_obstacle_probe_corr = _now_corr
                _obs_corr = self._analyze_frame(image)
                if _obs_corr.obstacle_danger:
                    self.motion.publish(
                        self.config.obstacle_turn_linear,
                        self._avoidance_turn_for_zone(_obs_corr.obstacle_zone),
                    )
                    self._spin_sleep(interval)
                    continue
            cmd = follower.compute(image)
            if cmd.lane_found:
                self.motion.publish(cmd.linear, cmd.angular)
                self._spin_sleep(interval)
            else:
                self._lane_lost_step(follower, interval)
        else:
            exit_reason = 'timeout'
            print('CORRIDOR_DONE reason=timeout', flush=True)
        # Fail-soft transition guard: if we are supposed to require a green-floor
        # confirmation before entering the ring but never saw it, warn and
        # proceed anyway rather than stalling at the corridor end.
        if exit_reason != 'green_floor' and self.config.corridor_require_green_exit:
            self.get_logger().warn(
                f'CORRIDOR_FOLLOW exited via {exit_reason} without green-floor '
                'confirmation; entering ring fail-soft')
            self._event('corridor_green_missing', exit_reason=exit_reason)
        self._event('corridor_follow_done', exit_reason=exit_reason)
        self._stop_with_spin()
        self._settle_transition('B_CORRIDOR->C_RING')
        return 'CORRIDOR_TO_LOOP_TURN'

    def _state_corridor_to_loop_turn(self) -> str:
        target_delta = math.pi / 2.0 if self.direction == 'left' else -math.pi / 2.0
        self._event(
            'corridor_to_loop_turn_start',
            direction=self.direction,
            target_deg=round(math.degrees(target_delta), 1),
        )
        self._rotate_relative_arc(
            target_delta,
            self.config.qr_align_heading_tol,
            self.config.shunt_turn_timeout,
        )
        self._stop_with_spin()
        return 'LOOP_LEFT' if self.direction == 'left' else 'LOOP_RIGHT'

    def _state_loop(self, direction: str) -> str:
        """Zone C: follow the yellow ring ~360deg, capturing orange markers inline."""
        route_name = 'C_LOOP_LEFT' if direction == 'left' else 'C_LOOP_RIGHT'
        target_marker = 'MARKER_LEFT' if direction == 'left' else 'MARKER_RIGHT'
        self.target_marker_name = target_marker
        self.target_marker_route = route_name
        print(f'TARGET_MARKER_LOCKED name={target_marker} route={route_name}', flush=True)
        self._event('target_marker_locked', marker=target_marker, route=route_name)
        if self._map_route_ready() and self.route_map.route(route_name):
            result = self._execute_map_route(
                route_name,
                capture_markers=True,
                target_marker=target_marker,
            )
            if result == 'STOP':
                return 'STOP'
            self._stop_with_spin()
            return 'RETURN_HOME'
        if self._fixed_route_ready() and route_name in self.config.fixed_routes:
            result = self._execute_fixed_route(route_name, capture_markers=True)
            if result == 'STOP':
                return 'STOP'
            self._stop_with_spin()
            return 'RETURN_HOME'
        if not self.config.lane_follow_enabled:
            return self._state_loop_timed(direction)
        # Bias hugs the inner edge: ccw (left) ring keeps lane on the right.
        bias = self.config.ring_bias if direction == 'left' else -self.config.ring_bias
        side_mode = 'right_edge' if direction == 'left' else 'left_edge'
        follower = LaneFollower(
            self.vision_detector, self.config.lane_color,
            self.config.lane_follow_config(
                bias=bias, side_mode=side_mode,
                erode_px=self.config.lane_mask_erode_px))
        follower.reset()
        if self.motion_state is not None:
            self.motion_state.reset_yaw()
        return self._run_ring_follow(direction, follower)

    def _state_loop_timed(self, direction: str) -> str:
        """Fallback timed turn (original behavior) when lane following disabled."""
        angular = self.config.loop_angular if direction == 'left' else -self.config.loop_angular
        self._timed_motion(
            label=f'LOOP_{direction.upper()}',
            linear=self.config.loop_linear,
            angular=angular,
            duration=self.config.loop_duration,
        )
        self._stop_with_spin()
        return 'RETURN_HOME'

    def _run_ring_follow(self, direction: str, follower) -> str:
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.loop_timeout
        target_turn = 2.0 * math.pi * self.config.loop_complete_fraction
        last_capture_probe = 0.0
        _last_obstacle_probe_ring: float = 0.0
        _last_ring_obs_log: float = 0.0
        _loop_start = time.monotonic()
        print(
            f'LOOP_{direction.upper()}: yellow ring follow, '
            f'complete at {math.degrees(target_turn):.0f}deg '
            f'timeout={self.config.loop_timeout:.0f}s',
            flush=True,
        )
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = f'max runtime exceeded during LOOP_{direction.upper()}'
                return 'STOP'
            # Loop completion via integrated yaw (relative, drift-tolerant).
            if self.motion_state is not None:
                turned = abs(self.motion_state.integrated_yaw())
                if (turned >= target_turn
                        and time.monotonic() - _loop_start >= self.config.loop_min_time_s):
                    reason = 'yaw_complete'
                    break
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            now = time.monotonic()
            if now - last_capture_probe >= 0.3:
                last_capture_probe = now
                self._maybe_capture_marker(image)
            _ring_obs_lin_scale = 1.0
            if self.config.obstacle_avoid_enabled and now - _last_obstacle_probe_ring >= 0.3:
                _last_obstacle_probe_ring = now
                _obs_ring = self._analyze_frame(image)
                if _obs_ring.obstacle_danger:
                    _ring_obs_lin_scale = self.config.ring_avoid_linear_scale
                    if now - _last_ring_obs_log >= 2.0:
                        _last_ring_obs_log = now
                        self._event('ring_obstacle_slow')
                        self.get_logger().info(
                            f'RING: obstacle detected, slowing to {self.config.ring_avoid_linear_scale}x speed')
            cmd = follower.compute(image)
            if cmd.lane_found:
                self.motion.publish(cmd.linear * _ring_obs_lin_scale, cmd.angular)
                self._spin_sleep(interval)
            else:
                self._lane_lost_step(follower, interval)
        turned_deg = (
            math.degrees(abs(self.motion_state.integrated_yaw()))
            if self.motion_state is not None else 0.0
        )
        print(f'LOOP_DONE reason={reason} turned={turned_deg:.0f}deg '
              f'markers={len(self.captured_markers)}', flush=True)
        self._event(
            'loop_done', zone='C_RING', reason=reason,
            turned_deg=round(turned_deg, 1), markers=len(self.captured_markers),
        )
        self._stop_with_spin()
        return 'RETURN_HOME'

    def _maybe_capture_marker(self, image, marker_name: Optional[str] = None) -> None:
        """Save an orange-marker crop when one is large enough during the ring."""
        decision = self._analyze_frame(image)
        card = decision.target_card
        if card is None:
            return
        h, w = image.shape[:2]
        area_ratio = card.area / float(max(1, w * h))
        if area_ratio < self.config.marker_capture_min_area:
            return
        crop = self.vision_detector.crop_detection(image, card)
        path = self._save_debug_image('ring_marker', crop)
        self.captured_markers.append({
            'path': path, 'area_ratio': round(area_ratio, 4),
            'label': card.label,
            'marker': marker_name or '',
        })
        self.target_card_seen_count += 1
        print(
            f'RING_MARKER_CAPTURED marker={marker_name or ""} '
            f'area={area_ratio:.3f} path={path}',
            flush=True,
        )
        self._event(
            'ring_marker_captured',
            marker=marker_name or '',
            area_ratio=round(area_ratio, 4),
            path=path,
        )

    def _state_capture_target_image(self) -> str:
        # Prefer the best orange marker captured inline during the ring.
        if self.captured_markers:
            candidates = self.captured_markers
            if (
                self.config.target_marker_policy == 'directional_one'
                and self.target_marker_name
            ):
                directional = [
                    marker for marker in self.captured_markers
                    if marker.get('marker') == self.target_marker_name
                ]
                if directional:
                    candidates = directional
            best = max(candidates, key=lambda m: m.get('area_ratio', 0.0))
            self.target_image_path = str(best.get('path') or '') or None
            if self.target_image_path:
                print(
                    f'TARGET_IMAGE: {self.target_image_path} '
                    f'(marker={best.get("marker", "")} '
                    f'best of {len(candidates)} selected markers)',
                    flush=True,
                )
                self._event(
                    'target_capture', source='ring_marker',
                    target_marker=best.get('marker', ''),
                    target_card_seen_count=self.target_card_seen_count,
                    target_image=self.target_image_path,
                )
                return 'CALL_LLM_API'
        deadline = time.monotonic() + self.config.capture_timeout
        image = None
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is not None:
                break
        if image is None:
            self.failure_reason = 'no image available for target capture'
            return 'STOP'
        decision = self._analyze_frame(image)
        if decision.target_card is not None:
            self.target_card_seen_count += 1
        if self.args.save_debug_images:
            overlay = self.vision_detector.draw_debug(image, decision)
            self._save_debug_image('target_debug', overlay)
        if self.config.target_crop_enabled and decision.target_card is not None:
            crop = self.vision_detector.crop_detection(image, decision.target_card)
            path = self._save_debug_image('target_crop', crop)
        else:
            path = self._save_debug_image('target_image', image)
        self.target_image_path = str(path) if path is not None else None
        if self.target_image_path:
            print(f'TARGET_IMAGE: {self.target_image_path}', flush=True)
        self._event(
            'target_capture',
            target_card_seen=decision.target_card is not None,
            target_card_seen_count=self.target_card_seen_count,
            target_image=self.target_image_path or '',
        )
        return 'CALL_LLM_API'

    def _avoidance_turn_for_zone(self, zone: str) -> float:
        turn = self.config.obstacle_turn_angular
        if zone == 'left':
            return -turn
        if zone == 'right':
            return turn
        return turn

    def _state_call_llm_api(self) -> str:
        self._stop_with_spin()
        if self.config.llm_mode == 'disabled':
            self.llm_result = ''
            print('LLM_SKIPPED: llm_mode=disabled', flush=True)
            self._event(
                'llm_result',
                llm_mode=self.config.llm_mode,
                result_preview='',
                published_topic=self.config.llm_result_topic,
                skipped=True,
            )
            return 'RETURN_HOME'
        self.llm_result = self.llm_client.analyze(self.target_image_path)
        print(f'LLM_RESULT: {self.llm_result}', flush=True)
        result_msg = String()
        result_msg.data = self.llm_result
        self.llm_result_pub.publish(result_msg)
        self._event(
            'llm_result',
            llm_mode=self.config.llm_mode,
            result_preview=self.llm_result[:200],
            published_topic=self.config.llm_result_topic,
        )
        return 'RETURN_HOME'

    def _state_return_home(self) -> str:
        follower = LaneFollower(
            self.vision_detector,
            'black',
            self.config.lane_follow_config(
                bias=0.0,
                side_mode='center',
            ),
        )
        follower.reset()
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.return_total_timeout
        print('RETURN_HOME: reacquire black line, then follow home', flush=True)
        self._event('return_home_plan', mode='black_line_follow_reacquire')
        if not self._prepare_return_line_follow(follower, interval):
            self.failure_reason = 'return_home line reacquire timeout'
            self._stop_with_spin()
            return 'STOP'
        lane_lost_frames = 0
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during RETURN_HOME'
                return 'STOP'
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            cmd = follower.compute(image)
            if not cmd.lane_found:
                lane_lost_frames += 1
                if lane_lost_frames < max(1, self.config.return_line_lost_grace_frames):
                    self._lane_lost_step(follower, interval)
                    continue
                print('RETURN_HOME_LINE_LOST: attempting reacquire', flush=True)
                self._event(
                    'return_home_line_lost',
                    lost_frames=lane_lost_frames,
                    retry=True,
                )
                if self._prepare_return_line_follow(follower, interval):
                    lane_lost_frames = 0
                    continue
                print('RETURN_HOME_DONE reason=line_lost', flush=True)
                self._event('return_home_done', reason='line_lost')
                self._stop_with_spin()
                return 'STOP'
            lane_lost_frames = 0
            self.motion.publish(cmd.linear, cmd.angular)
            self._spin_sleep(interval)
        print('RETURN_HOME_DONE reason=timeout', flush=True)
        self._event('return_home_done', reason='timeout')
        self._stop_with_spin()
        return 'STOP'

    def _prepare_return_line_follow(self, follower, interval: float) -> bool:
        if not self.config.return_reacquire_enabled:
            return True
        if self.direction == 'left':
            search_sign = 1.0
            target_delta = -math.radians(self.config.return_reacquire_turn_deg)
        elif self.direction == 'right':
            search_sign = -1.0
            target_delta = math.radians(self.config.return_reacquire_turn_deg)
        else:
            search_sign = -1.0
            target_delta = 0.0
        print(
            'RETURN_HOME_REACQUIRE '
            f'direction={self.direction or "unknown"} '
            f'turn_deg={math.degrees(target_delta):+.1f}',
            flush=True,
        )
        self._event(
            'return_home_reacquire_start',
            direction=self.direction or '',
            turn_deg=round(math.degrees(target_delta), 1),
        )
        if abs(target_delta) > 1e-3:
            self._rotate_relative_arc(
                target_delta,
                self.config.return_heading_tol,
                self.config.shunt_turn_timeout,
            )
            self._stop_with_spin()
        forward_deadline = time.monotonic() + max(0.0, self.config.return_reacquire_forward_s)
        while time.monotonic() < forward_deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return False
            self.motion.publish(self.config.return_reacquire_linear, 0.0)
            self._spin_sleep(interval)
        target_delta_2 = math.copysign(
            math.radians(self.config.return_reacquire_turn2_deg),
            target_delta,
        ) if abs(target_delta) > 1e-3 else 0.0
        if abs(target_delta_2) > 1e-3:
            print(
                'RETURN_HOME_REACQUIRE_STAGE2 '
                f'turn_deg={math.degrees(target_delta_2):+.1f}',
                flush=True,
            )
            self._event(
                'return_home_reacquire_stage2',
                turn_deg=round(math.degrees(target_delta_2), 1),
            )
            self._rotate_relative_arc(
                target_delta_2,
                self.config.return_heading_tol,
                self.config.shunt_turn_timeout,
            )
            self._stop_with_spin()
        forward_deadline_2 = time.monotonic() + max(0.0, self.config.return_reacquire_forward2_s)
        while time.monotonic() < forward_deadline_2 and rclpy.ok():
            if not self.guard.runtime_ok():
                return False
            self.motion.publish(self.config.return_reacquire_linear, 0.0)
            self._spin_sleep(interval)
        deadline = time.monotonic() + max(0.1, self.config.return_reacquire_search_timeout)
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return False
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is None:
                self._spin_sleep(interval)
                continue
            cmd = follower.compute(image)
            if cmd.lane_found:
                print(
                    'RETURN_HOME_REACQUIRE_DONE '
                    f'linear={cmd.linear:.2f} angular={cmd.angular:+.2f}',
                    flush=True,
                )
                self._event(
                    'return_home_reacquire_done',
                    linear=round(cmd.linear, 3),
                    angular=round(cmd.angular, 3),
                )
                return True
            self.motion.publish(
                self.config.return_reacquire_linear,
                search_sign * abs(self.config.return_reacquire_angular),
            )
            self._spin_sleep(interval)
        print('RETURN_HOME_REACQUIRE_TIMEOUT', flush=True)
        self._event('return_home_reacquire_timeout')
        return False

    def _return_home_trajectory(self) -> Optional[str]:
        if self.motion_state is None:
            return None
        trajectory = self.motion_state.trajectory()
        self.motion_state.stop_recording()
        segments = reverse_plan(
            trajectory,
            waypoint_stride=self.config.return_waypoint_stride,
            min_segment_dist=self.config.return_min_segment_dist,
        )
        print(
            f'RETURN_HOME: trajectory replay, {len(trajectory)} waypoints '
            f'-> {len(segments)} segments',
            flush=True,
        )
        self._event(
            'return_home_plan', waypoints=len(trajectory),
            segments=len(segments), mode='trajectory',
        )
        if segments:
            result = self._execute_return_segments(segments)
            self._stop_with_spin()
            return result
        print('RETURN_HOME: no usable trajectory, falling back to timed', flush=True)
        return None

    def _execute_return_segments(self, segments) -> str:
        total_deadline = time.monotonic() + self.config.return_total_timeout
        for idx, seg in enumerate(segments):
            if time.monotonic() >= total_deadline:
                print('RETURN_HOME_TIMEOUT total', flush=True)
                self.failure_reason = 'return_home total timeout'
                break
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during RETURN_HOME'
                return 'STOP'
            print(
                f'RETURN_SEG {idx + 1}/{len(segments)} '
                f'heading={math.degrees(seg.target_heading):.0f}deg '
                f'dist={seg.distance:.2f}m',
                flush=True,
            )
            self._rotate_to_heading(seg.target_heading)
            self._drive_distance(seg.distance)
        return 'STOP'

    def _rotate_to_heading(self, target_heading: float) -> None:
        """Closed-loop rotate until absolute yaw is within tolerance of target."""
        if self.config.ackermann_turn_mode == 'shunt':
            yaw = self.motion_state.absolute_yaw()
            if yaw is None:
                return
            target = angle_diff(target_heading, yaw)
            result = execute_shunt_turn(
                target,
                motion_state=self.motion_state,
                publish_velocity=self.motion.publish,
                spin_sleep=lambda: self._spin_sleep(1.0 / self.config.rate_hz),
                forward_linear=self.config.shunt_forward_linear,
                reverse_linear=self.config.shunt_reverse_linear,
                angular=self.config.return_turn_angular,
                heading_tol=self.config.return_heading_tol,
                timeout=self.config.shunt_turn_timeout,
                pulse_s=self.config.shunt_pulse_s,
                stop_s=self.config.shunt_stop_s,
                ok=rclpy.ok,
                runtime_ok=self.guard.runtime_ok,
            )
            print(
                'RETURN_SHUNT_TURN_DONE '
                f'reason={result.reason} turned_deg={math.degrees(result.turned_rad):.1f} '
                f'elapsed={result.elapsed_s:.1f} cycles={result.cycles}',
                flush=True,
            )
            self.motion.publish(0.0, 0.0)
            return
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.return_segment_timeout
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return
            yaw = self.motion_state.absolute_yaw()
            if yaw is None:
                self._spin_sleep(interval)
                continue
            err = angle_diff(target_heading, yaw)
            if abs(err) <= self.config.return_heading_tol:
                break
            # err>0 means target is ccw (left) -> positive angular.
            turn = self.config.return_turn_angular if err > 0 else -self.config.return_turn_angular
            self.motion.publish(self.config.return_turn_linear, turn)
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)

    def _drive_distance(self, distance: float) -> None:
        """Closed-loop drive forward until odom distance delta reaches target."""
        interval = 1.0 / self.config.rate_hz
        deadline = time.monotonic() + self.config.return_segment_timeout
        start = self.motion_state.traveled_distance()
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                return
            traveled = self.motion_state.traveled_distance() - start
            if traveled >= distance - self.config.return_dist_tol:
                break
            self.motion.publish(self.config.return_drive_linear, 0.0)
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)

    def _fixed_route_ready(self) -> bool:
        if not self.config.fixed_route_active():
            return False
        if self.motion_state is None:
            self.get_logger().warn('fixed2d route requested but motion_state is unavailable')
            return False
        return True

    def _map_route_ready(self) -> bool:
        if not self.config.map_route_active():
            return False
        if self.motion_state is None:
            self.get_logger().warn('map2d route requested but motion_state is unavailable')
            return False
        return True

    def _reset_map_frame(self) -> bool:
        if self.motion_state is None:
            return False
        pose = self.motion_state.pose()
        if pose is None:
            return False
        self._map_odom_start = pose
        self._map_start_pose = pose_from_mapping(
            self.config.map_start_pose,
            self.route_map.start_pose,
        )
        print(
            'MAP_FRAME_RESET '
            f'x={self._map_start_pose.x:.3f} '
            f'y={self._map_start_pose.y:.3f} '
            f'heading_deg={math.degrees(self._map_start_pose.yaw):.1f}',
            flush=True,
        )
        self._event(
            'map_frame_reset',
            x=round(self._map_start_pose.x, 3),
            y=round(self._map_start_pose.y, 3),
            heading_deg=round(math.degrees(self._map_start_pose.yaw), 1),
        )
        return True

    def _reanchor_map_frame(self, waypoint_name: str) -> None:
        """Snap map-frame x/y to a physically-confirmed landmark waypoint.

        Heading is intentionally kept: a single landmark cannot observe yaw,
        and the approach angle varies run to run. Skips with a warning if the
        implied correction exceeds qr_reanchor_max_jump (likely a mis-set
        waypoint, not drift).
        """
        if not self.config.qr_reanchor_enabled or self.motion_state is None:
            return
        wp = self.route_map.waypoints.get(waypoint_name)
        raw = self.motion_state.pose()
        cur = self._current_map_pose()
        if wp is None or raw is None or cur is None:
            return
        dx, dy = wp.x - cur.x, wp.y - cur.y
        jump = math.hypot(dx, dy)
        if jump > self.config.qr_reanchor_max_jump:
            print(
                f'MAP_REANCHOR_SKIPPED waypoint={waypoint_name} '
                f'jump={jump:.3f} > max={self.config.qr_reanchor_max_jump}',
                flush=True,
            )
            self._event('map_reanchor_skipped', waypoint=waypoint_name,
                        jump=round(jump, 3))
            return
        self._map_odom_start = raw
        self._map_start_pose = Pose2D(wp.x, wp.y, cur.yaw)
        print(
            f'MAP_REANCHOR waypoint={waypoint_name} '
            f'dx={dx:.3f} dy={dy:.3f} drift={jump:.3f}',
            flush=True,
        )
        self._event('map_reanchor', waypoint=waypoint_name,
                    dx=round(dx, 3), dy=round(dy, 3), drift=round(jump, 3))

    def _current_map_pose(self) -> Optional[Pose2D]:
        if self.motion_state is None:
            return None
        pose = self.motion_state.pose()
        if pose is None:
            return None
        if self._map_odom_start is None and not self._reset_map_frame():
            return None
        return map_pose_from_odom(pose, self._map_odom_start, self._map_start_pose)

    def _map_route_timeout(self, name: str) -> float:
        if name == 'RETURN_TO_P':
            return self.config.map_return_timeout
        if name == 'A_TO_QR':
            return max(self.config.map_route_timeout, self.config.drive_to_qr_timeout)
        if name == 'B_CORRIDOR':
            return max(self.config.map_route_timeout, self.config.corridor_timeout)
        if name.startswith('C_LOOP'):
            return max(self.config.map_route_timeout, self.config.loop_timeout)
        return self.config.map_route_timeout

    def _map_route_linear_limits(self, name: str) -> Tuple[float, float]:
        if name.startswith('C_LOOP'):
            return self.config.map_loop_linear, self.config.map_loop_turn_linear
        return self.config.map_max_linear, min(
            self.config.steering_turn_linear,
            self.config.map_max_linear,
        )

    def _execute_map_route(
        self,
        name: str,
        allow_qr: bool = False,
        lane_color: Optional[str] = None,
        capture_markers: bool = False,
        target_marker: Optional[str] = None,
    ) -> Optional[str]:
        route = self.route_map.route(name)
        if len(route) < 2:
            self.failure_reason = f'missing map route: {name}'
            return 'STOP'
        print(f'MAP_ROUTE_START name={name}', flush=True)
        self._event('map_route_start', name=name, waypoints=len(route))
        follower = None
        if lane_color:
            follower = LaneFollower(
                self.vision_detector,
                lane_color,
                self.config.lane_follow_config(bias=0.0, side_mode='center'),
            )
            follower.reset()
        target_index = 0
        pose = self._current_map_pose()
        if pose is not None:
            first = route[0]
            if math.hypot(first.x - pose.x, first.y - pose.y) <= self.config.map_waypoint_tol * 2.0:
                target_index = 1
        target_min_dist = float('inf')  # closest approach to current target (overshoot recovery)
        loop_stall_s = 1.0
        pose_jump_k = 3.0
        pose_jump_margin = 0.10
        behind_skip_rad = math.radians(110.0)
        last_loop_start = time.monotonic()
        loop_dt = 0.0
        last_loop_phase = 'start'
        last_pose: Optional[Pose2D] = None
        last_pose_time: Optional[float] = None
        deadline = time.monotonic() + max(0.1, self._map_route_timeout(name))
        route_linear, route_turn_linear = self._map_route_linear_limits(name)
        interval = 1.0 / self.config.rate_hz
        last_log = 0.0
        last_qr_probe = 0.0
        last_capture_probe = 0.0
        target_marker_wp = self.route_map.waypoints.get(target_marker or '')
        start_time = time.monotonic()
        while target_index < len(route) and time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = f'max runtime exceeded during map route {name}'
                return 'STOP'
            pose = self._current_map_pose()
            if pose is None:
                last_loop_phase = 'spin_sleep_no_pose'
                self._spin_sleep(interval)
                continue
            loop_start = time.monotonic()
            loop_dt = loop_start - last_loop_start
            if loop_dt > loop_stall_s:
                print(
                    f'LOOP_STALL dt={loop_dt:.3f} phase={last_loop_phase}',
                    flush=True,
                )
                self._event(
                    'map_loop_stall',
                    route=name,
                    dt=round(loop_dt, 3),
                    phase=last_loop_phase,
                )
            last_loop_start = loop_start
            pose_time = time.monotonic()
            if last_pose is not None and last_pose_time is not None:
                pose_dt = max(0.0, pose_time - last_pose_time)
                pose_delta = math.hypot(pose.x - last_pose.x, pose.y - last_pose.y)
                pose_allowed = (
                    self.config.map_max_linear * pose_dt * pose_jump_k
                    + pose_jump_margin
                )
                if pose_delta > pose_allowed:
                    print(
                        'POSE_JUMP '
                        f'd={pose_delta:.3f} dt={pose_dt:.3f} '
                        f'allowed={pose_allowed:.3f}',
                        flush=True,
                    )
                    self._event(
                        'map_pose_jump',
                        route=name,
                        d=round(pose_delta, 3),
                        dt=round(pose_dt, 3),
                        allowed=round(pose_allowed, 3),
                    )
                    last_pose = pose
                    last_pose_time = pose_time
                    last_loop_phase = 'pose_jump_stop'
                    self.motion.publish(0.0, 0.0)
                    self._spin_sleep(interval)
                    continue
            last_pose = pose
            last_pose_time = pose_time
            target = route[target_index]
            dx = target.x - pose.x
            dy = target.y - pose.y
            dist = math.hypot(dx, dy)
            target_heading = math.atan2(dy, dx)
            heading_err = angle_diff(target_heading, pose.yaw)
            if dist < target_min_dist:
                target_min_dist = dist
            reached = dist <= self.config.map_waypoint_tol
            # Overshoot recovery: if we already approached this point (got within
            # ~2x tol), it is now behind us (|heading_err| > 90deg) and we are
            # moving away from it, treat it as passed and advance to the next point
            # instead of looping back. An Ackermann car cannot reverse-turn, so
            # chasing a waypoint behind it produces a big circle (observed 06-13).
            passed = (
                not reached
                and abs(heading_err) > math.pi / 2
                and target_min_dist <= self.config.map_waypoint_tol * 2.0
                and dist > target_min_dist + 0.02
            )
            passed_reason = 'receding' if passed else ''
            advance_to = target_index + 1
            if (
                not reached
                and not passed
                and target_index > 0
                and abs(heading_err) > behind_skip_rad
            ):
                for candidate_index in range(target_index + 1, len(route)):
                    candidate = route[candidate_index]
                    cand_heading = math.atan2(candidate.y - pose.y, candidate.x - pose.x)
                    cand_err = angle_diff(cand_heading, pose.yaw)
                    if abs(cand_err) <= behind_skip_rad:
                        advance_to = candidate_index
                        passed = True
                        passed_reason = 'behind'
                        break
                if not passed and target_index + 1 < len(route):
                    advance_to = target_index + 1
                    passed = True
                    passed_reason = 'behind'
            if reached or passed:
                tag = 'MAP_WAYPOINT_REACHED' if reached else 'MAP_WAYPOINT_PASSED'
                print(
                    f'{tag} name={target.name} dist={dist:.3f} '
                    f'reason={passed_reason or "tol"} next_index={advance_to}',
                    flush=True,
                )
                self._event(
                    'map_waypoint_reached',
                    route=name,
                    waypoint=target.name,
                    x=round(pose.x, 3),
                    y=round(pose.y, 3),
                    passed=bool(passed),
                    passed_reason=passed_reason,
                    next_index=advance_to,
                )
                target_index = advance_to
                target_min_dist = float('inf')
                last_loop_phase = 'waypoint_advance'
                continue
            last_loop_phase = 'start_heading_guard'
            if self._map_start_heading_guard(name, target_index, target, heading_err):
                return 'STOP'
            now = time.monotonic()
            last_loop_phase = 'get_latest_image'
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is not None:
                last_loop_phase = 'map_obstacle_step'
                if self._map_obstacle_step(image, interval):
                    last_loop_phase = 'map_detour'
                    continue
                if allow_qr and now - last_qr_probe >= 0.25:
                    last_qr_probe = now
                    last_loop_phase = 'drive_probe_qr'
                    result = self._drive_probe_qr(image, now - start_time)
                    if result is not None:
                        print(f'MAP_ROUTE_DONE name={name} reason=qr', flush=True)
                        self._event('map_route_done', name=name, reason='qr')
                        return result
                in_marker_window = (
                    target_marker_wp is None
                    or math.hypot(target_marker_wp.x - pose.x, target_marker_wp.y - pose.y)
                    <= self.config.marker_capture_window_m
                )
                if capture_markers and in_marker_window and now - last_capture_probe >= 0.3:
                    last_capture_probe = now
                    print(
                        'MARKER_WINDOW_ACTIVE '
                        f'name={target_marker or ""} '
                        f'dist={0.0 if target_marker_wp is None else math.hypot(target_marker_wp.x - pose.x, target_marker_wp.y - pose.y):.3f}',
                        flush=True,
                    )
                    last_loop_phase = 'maybe_capture_marker'
                    self._maybe_capture_marker(image, target_marker)
            angular = max(
                -self.config.map_max_angular,
                min(self.config.map_max_angular, self.config.map_heading_kp * heading_err),
            )
            linear = route_linear
            if abs(heading_err) > 0.8:
                linear = route_turn_linear
            elif abs(heading_err) > 0.4:
                linear = max(route_turn_linear, linear * 0.5)
            if follower is not None and image is not None:
                cmd = follower.compute(image)
                if cmd.lane_found:
                    angular = max(
                        -self.config.map_max_angular,
                        min(self.config.map_max_angular, (angular + cmd.angular) * 0.5),
                    )
                    linear = min(linear, cmd.linear, route_linear)
            if now - last_log >= 1.0:
                last_log = now
                print(
                    'MAP_POSE '
                    f'x={pose.x:.3f} y={pose.y:.3f} '
                    f'yaw={math.degrees(pose.yaw):.1f} dt={loop_dt:.3f}',
                    flush=True,
                )
                print(
                    'MAP_TARGET '
                    f'waypoint={target.name} dist={dist:.3f} '
                    f'heading_err={heading_err:.3f} '
                    f'linear={linear:.3f} angular={angular:.3f} dt={loop_dt:.3f}',
                    flush=True,
                )
            self.motion.publish(linear, angular)
            last_loop_phase = 'spin_sleep'
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        reason = 'done' if target_index >= len(route) else 'timeout'
        print(f'MAP_ROUTE_DONE name={name} reason={reason}', flush=True)
        self._event('map_route_done', name=name, reason=reason)
        return None if reason == 'done' else 'STOP'

    def _map_start_heading_guard(
        self,
        route_name: str,
        target_index: int,
        target: Waypoint,
        heading_err: float,
    ) -> bool:
        if not self.config.map_start_heading_guard_enabled:
            return False
        if route_name != 'A_TO_QR' or target_index > 1:
            return False
        limit = self.config.map_start_heading_guard_rad
        if abs(heading_err) <= limit:
            return False
        self.motion.publish(0.0, 0.0)
        print(
            'MAP_START_HEADING_MISMATCH '
            f'route={route_name} waypoint={target.name} '
            f'heading_err={heading_err:.3f} limit={limit:.3f}',
            flush=True,
        )
        self._event(
            'map_start_heading_mismatch',
            route=route_name,
            waypoint=target.name,
            heading_err=round(heading_err, 4),
            limit=round(limit, 4),
        )
        self.failure_reason = (
            f'map start heading mismatch: {math.degrees(heading_err):.1f}deg '
            f'to {target.name}'
        )
        return True

    def _map_obstacle_step(self, image, interval: float) -> bool:
        if not (self.config.obstacle_avoid_enabled and self.config.map_detour_enabled):
            return False
        decision = self._analyze_frame(image)
        if not (
            decision.obstacle_danger
            and decision.obstacle_zone == 'center'
        ):
            return False
        now = time.monotonic()
        if now - self._last_map_obstacle_time < self.config.map_detour_cooldown_s:
            return False
        self._last_map_obstacle_time = now
        source = decision.obstacle.source if decision.obstacle else 'unknown'
        action = self._map_detour_action(image, decision)
        print(
            'MAP_OBSTACLE '
            f'source={source} zone={decision.obstacle_zone} '
            f'area={decision.obstacle_area_ratio:.4f} action={action}',
            flush=True,
        )
        self._event(
            'map_obstacle',
            source=source,
            zone=decision.obstacle_zone,
            area_ratio=round(decision.obstacle_area_ratio, 5),
            action=action,
        )
        if self.args.save_debug_images:
            overlay = self.vision_detector.draw_debug(image, decision)
            self._save_debug_image('map_obstacle_debug', overlay)
        self._run_map_detour(action, interval)
        return True

    def _map_detour_action(self, image, decision) -> str:
        if decision.obstacle is None:
            return 'detour_left'
        width = image.shape[1]
        return 'detour_right' if decision.obstacle.center_x < width / 2.0 else 'detour_left'

    def _run_map_detour(self, action: str, interval: float) -> None:
        sign = 1.0 if action == 'detour_left' else -1.0
        angular = sign * self.config.map_max_angular
        self.motion.publish(0.0, 0.0)
        self._spin_sleep(interval)
        self._timed_motion(
            'MAP_DETOUR_TURN_OUT',
            self.config.steering_turn_linear,
            angular,
            self.config.map_detour_turn_s,
        )
        self._timed_motion('MAP_DETOUR_PASS', self.config.map_max_linear, 0.0, self.config.map_detour_drive_s)
        self._timed_motion(
            'MAP_DETOUR_TURN_BACK',
            self.config.steering_turn_linear,
            -angular,
            self.config.map_detour_recover_s,
        )
        self._timed_motion('MAP_DETOUR_REJOIN', self.config.map_max_linear, 0.0, self.config.map_detour_drive_s * 0.7)
        self._timed_motion(
            'MAP_DETOUR_REALIGN',
            self.config.steering_turn_linear,
            -angular,
            self.config.map_detour_recover_s * 0.5,
        )

    def _execute_fixed_route(
        self,
        name: str,
        allow_qr: bool = False,
        lane_color: Optional[str] = None,
        capture_markers: bool = False,
    ) -> Optional[str]:
        route = self.config.fixed_routes.get(name)
        if not isinstance(route, list):
            self.failure_reason = f'missing fixed route: {name}'
            return 'STOP'
        print(f'FIXED_ROUTE_START name={name}', flush=True)
        self._event('fixed_route_start', name=name, segments=len(route))
        follower = None
        if lane_color:
            follower = LaneFollower(
                self.vision_detector,
                lane_color,
                self.config.lane_follow_config(bias=0.0, side_mode='center'),
            )
            follower.reset()
        for index, segment in enumerate(route, start=1):
            if not isinstance(segment, dict):
                self.failure_reason = f'fixed route {name} segment {index} must be object'
                return 'STOP'
            seg_type = str(segment.get('type') or '')
            print(
                f'FIXED_ROUTE_SEG route={name} index={index} '
                f'type={seg_type} data={json.dumps(segment, ensure_ascii=False)}',
                flush=True,
            )
            self._event('fixed_route_segment', name=name, index=index, segment=segment)
            if seg_type == 'drive':
                result = self._fixed_route_drive(segment, follower)
            elif seg_type == 'turn':
                result = self._fixed_route_turn(segment)
            elif seg_type == 'arc':
                result = self._fixed_route_arc(segment, capture_markers=capture_markers)
            elif seg_type == 'seek_qr':
                result = self._fixed_route_seek_qr(segment, allow_qr=allow_qr)
            else:
                self.failure_reason = f'unsupported fixed route segment type: {seg_type}'
                return 'STOP'
            if result:
                return result
        self._event('fixed_route_done', name=name)
        return None

    def _fixed_route_drive(self, segment: Dict[str, object], follower=None) -> Optional[str]:
        distance = max(0.0, float(segment.get('distance', 0.0)))
        timeout = float(segment.get('timeout', self.config.fixed_route_segment_timeout))
        interval = 1.0 / self.config.rate_hz
        start = self.motion_state.traveled_distance()
        deadline = time.monotonic() + max(0.1, timeout)
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during fixed route drive'
                return 'STOP'
            traveled = self.motion_state.traveled_distance() - start
            if traveled >= max(0.0, distance - self.config.fixed_route_dist_tol):
                reason = 'distance'
                break
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is not None and self._route_obstacle_step(image, interval):
                continue
            if follower is not None and image is not None:
                cmd = follower.compute(image)
                if cmd.lane_found:
                    self.motion.publish(
                        min(cmd.linear, self.config.fixed_route_linear),
                        cmd.angular,
                    )
                else:
                    self.motion.publish(self.config.fixed_route_linear, 0.0)
            else:
                self.motion.publish(self.config.fixed_route_linear, 0.0)
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        self._fixed_segment_done('drive', reason, distance=distance)
        return None

    def _fixed_route_turn(self, segment: Dict[str, object]) -> Optional[str]:
        angle_deg = float(segment.get('angle_deg', 0.0))
        timeout = float(segment.get('timeout', self.config.fixed_route_segment_timeout))
        target = math.radians(angle_deg)
        if self.config.ackermann_turn_mode == 'shunt':
            result = execute_shunt_turn(
                target,
                motion_state=self.motion_state,
                publish_velocity=self.motion.publish,
                spin_sleep=lambda: self._spin_sleep(1.0 / self.config.rate_hz),
                forward_linear=self.config.shunt_forward_linear,
                reverse_linear=self.config.shunt_reverse_linear,
                angular=self.config.fixed_route_turn_angular,
                heading_tol=self.config.fixed_route_heading_tol,
                timeout=float(segment.get('timeout', self.config.shunt_turn_timeout)),
                pulse_s=self.config.shunt_pulse_s,
                stop_s=self.config.shunt_stop_s,
                ok=rclpy.ok,
                runtime_ok=self.guard.runtime_ok,
            )
            self.motion.publish(0.0, 0.0)
            self._fixed_segment_done(
                'turn',
                result.reason,
                angle_deg=angle_deg,
                turned_deg=round(math.degrees(result.turned_rad), 1),
                elapsed_s=round(result.elapsed_s, 1),
                cycles=result.cycles,
            )
            return None if result.ok else 'STOP'
        direction = 1.0 if target >= 0.0 else -1.0
        interval = 1.0 / self.config.rate_hz
        self.motion_state.reset_yaw()
        deadline = time.monotonic() + max(0.1, timeout)
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during fixed route turn'
                return 'STOP'
            turned = self.motion_state.integrated_yaw()
            remaining = abs(target) - abs(turned)
            if remaining <= self.config.fixed_route_heading_tol:
                reason = 'yaw'
                break
            self.motion.publish(
                self.config.fixed_route_turn_linear,
                direction * self.config.fixed_route_turn_angular,
            )
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        self._fixed_segment_done('turn', reason, angle_deg=angle_deg)
        return None

    def _fixed_route_arc(
        self,
        segment: Dict[str, object],
        capture_markers: bool = False,
    ) -> Optional[str]:
        angle_deg = float(segment.get('angle_deg', 0.0))
        timeout = float(segment.get('timeout', self.config.loop_timeout))
        target = math.radians(angle_deg)
        direction = 1.0 if target >= 0.0 else -1.0
        interval = 1.0 / self.config.rate_hz
        self.motion_state.reset_yaw()
        deadline = time.monotonic() + max(0.1, timeout)
        last_capture_probe = 0.0
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during fixed route arc'
                return 'STOP'
            turned = self.motion_state.integrated_yaw()
            if abs(target) - abs(turned) <= self.config.fixed_route_heading_tol:
                reason = 'yaw'
                break
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is not None:
                if self._route_obstacle_step(image, interval):
                    continue
                if capture_markers and time.monotonic() - last_capture_probe >= 0.3:
                    last_capture_probe = time.monotonic()
                    self._maybe_capture_marker(image)
            self.motion.publish(
                self.config.fixed_route_linear,
                direction * self.config.fixed_route_turn_angular,
            )
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        self._fixed_segment_done('arc', reason, angle_deg=angle_deg)
        return None

    def _fixed_route_seek_qr(
        self,
        segment: Dict[str, object],
        allow_qr: bool,
    ) -> Optional[str]:
        max_distance = max(0.0, float(segment.get('max_distance', 0.0)))
        timeout = float(segment.get('timeout', self.config.drive_to_qr_timeout))
        interval = 1.0 / self.config.rate_hz
        start = self.motion_state.traveled_distance()
        start_time = time.monotonic()
        deadline = start_time + max(0.1, timeout)
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                self.failure_reason = 'max runtime exceeded during fixed route seek_qr'
                return 'STOP'
            traveled = self.motion_state.traveled_distance() - start
            if max_distance > 0.0 and traveled >= max_distance:
                reason = 'distance'
                break
            image = self.vision.get_latest(max_age=self.config.image_stale_timeout)
            if image is not None:
                if self._route_obstacle_step(image, interval):
                    continue
                if allow_qr:
                    result = self._drive_probe_qr(image, time.monotonic() - start_time)
                    if result is not None:
                        self._fixed_segment_done('seek_qr', 'qr', max_distance=max_distance)
                        return result
            self.motion.publish(self.config.fixed_route_linear, 0.0)
            self._spin_sleep(interval)
        self.motion.publish(0.0, 0.0)
        self._fixed_segment_done('seek_qr', reason, max_distance=max_distance)
        return 'SCAN_QR' if allow_qr else None

    def _route_obstacle_step(self, image, interval: float) -> bool:
        decision = self._analyze_frame(image)
        if not (decision.obstacle_danger and self.config.obstacle_avoid_enabled):
            return False
        source = decision.obstacle.source if decision.obstacle else 'unknown'
        print(
            'ROUTE_OBSTACLE '
            f'source={source} zone={decision.obstacle_zone} '
            f'area={decision.obstacle_area_ratio:.4f}',
            flush=True,
        )
        self._event(
            'route_obstacle',
            source=source,
            zone=decision.obstacle_zone,
            area_ratio=round(decision.obstacle_area_ratio, 5),
        )
        self._handle_drive_obstacle(image, decision, interval)
        return True

    def _fixed_segment_done(self, seg_type: str, reason: str, **fields: object) -> None:
        print(
            f'FIXED_ROUTE_SEG_DONE type={seg_type} reason={reason} '
            f'data={json.dumps(fields, ensure_ascii=False)}',
            flush=True,
        )
        self._event('fixed_route_segment_done', type=seg_type, reason=reason, **fields)

    def _state_stop(self) -> str:
        self._stop_with_spin()
        return 'DONE'

    def _timed_motion(
        self,
        label: str,
        linear: float,
        angular: float,
        duration: float,
    ) -> None:
        print(
            f'ACTION: {label} linear.x={linear:.3f} angular.z={angular:.3f} '
            f'duration={duration:.2f}s',
            flush=True,
        )
        self._event(
            'motion_action',
            label=label,
            linear_x=round(linear, 4),
            angular_z=round(angular, 4),
            duration_sec=round(duration, 3),
            no_motion=bool(self.args.no_motion),
        )
        deadline = time.monotonic() + duration
        interval = 1.0 / self.config.rate_hz
        while time.monotonic() < deadline and rclpy.ok():
            if not self.guard.runtime_ok():
                raise MissionAbort(f'max runtime exceeded during {label}')
            self.motion.publish(linear, angular)
            self._spin_sleep(interval)

    def _stop_with_spin(self) -> None:
        if rclpy is None or not rclpy.ok():
            return
        interval = 1.0 / self.config.rate_hz
        for _ in range(max(1, self.config.stop_repeat)):
            try:
                self.motion.stop_once()
                self._spin_sleep(interval)
            except Exception as exc:
                print(f'STOP_PUBLISH_FAILED error={exc}', flush=True)
                break

    def _settle_transition(self, label: str) -> None:
        """Brief settle stop between zones so motion damps before the next state."""
        settle = self.config.transition_settle_s
        if settle <= 0.0:
            return
        print(f'ZONE_TRANSITION {label} settle={settle:.2f}s', flush=True)
        self._event('zone_transition', transition=label, settle_s=round(settle, 3))
        end = time.monotonic() + settle
        while time.monotonic() < end and rclpy.ok():
            try:
                self.motion.stop_once()
                self._spin_sleep(min(0.1, end - time.monotonic()))
            except Exception as exc:
                print(f'STOP_PUBLISH_FAILED error={exc}', flush=True)
                break

    def _spin_sleep(self, duration: float) -> None:
        end = time.monotonic() + max(0.0, duration)
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=min(0.02, end - time.monotonic()))

    def _wait_until(
        self,
        predicate: Callable[[], bool],
        timeout: float,
        spin_step: float = 0.05,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            if predicate():
                return True
            if not self.guard.runtime_ok():
                return False
            rclpy.spin_once(self, timeout_sec=spin_step)
        return predicate()

    def _save_debug_image(
        self,
        stem: str,
        image: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = self.debug_dir / f'{stem}_{stamp}.jpg'
        ok = self.vision.save_image(path, image=image)
        if not ok:
            self.get_logger().warn(f'failed to save image: {path}')
            return None
        # Return a str (not PosixPath): callers feed this into _event()/json.dumps
        # and into captured_markers, which a Path object would crash (field 6-14).
        path = str(path)
        self.saved_images.append(path)
        print(f'SAVED_IMAGE: {path}', flush=True)
        return path

    def _print_summary(self) -> None:
        runtime_sec = time.monotonic() - self.started_at
        print('MISSION_SUMMARY', flush=True)
        print(f'  qr_content: {self.qr_content or ""}', flush=True)
        print(f'  qr_method: {self.qr_method or ""}', flush=True)
        print(f'  direction: {self.direction or ""}', flush=True)
        print(f'  direction_source: {self.direction_source or ""}', flush=True)
        print(f'  obstacle_event_count: {self.obstacle_event_count}', flush=True)
        print(f'  target_card_seen_count: {self.target_card_seen_count}', flush=True)
        print(f'  target_image: {self.target_image_path or ""}', flush=True)
        print(f'  llm_mode: {self.config.llm_mode}', flush=True)
        print(f'  llm_result: {self.llm_result or ""}', flush=True)
        print(f'  runtime_sec: {runtime_sec:.2f}', flush=True)
        print(f'  failure_reason: {self.failure_reason or ""}', flush=True)
        self._event(
            'mission_summary',
            qr_method=self.qr_method or '',
            direction=self.direction or '',
            direction_source=self.direction_source or '',
            obstacle_event_count=self.obstacle_event_count,
            target_card_seen_count=self.target_card_seen_count,
            target_image=self.target_image_path or '',
            llm_mode=self.config.llm_mode,
            failure_reason=self.failure_reason or '',
            saved_images=self.saved_images,
        )


def parse_direction(text: str) -> str:
    return parse_qr_direction(text)


def default_config_path() -> Optional[Path]:
    candidates = []
    if get_package_share_directory is not None:
        try:
            share_dir = Path(get_package_share_directory(PACKAGE_NAME))
            candidates.append(share_dir / 'config' / 'mission_defaults.json')
        except Exception:
            pass
    candidates.append(Path(__file__).resolve().parents[1] / 'config' / 'mission_defaults.json')
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_config(path: Optional[str]) -> MissionConfig:
    selected = Path(path) if path else default_config_path()
    if selected is None:
        return MissionConfig()
    with selected.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return MissionConfig.from_mapping(data)


def run_dry_run(config: MissionConfig, args: argparse.Namespace) -> int:
    print('auto_mission dry-run')
    print(f'  start_state: {args.start_state}')
    print(f'  no_motion: {args.no_motion}')
    print(f'  max_runtime: {config.max_runtime:.1f}s')
    print(f'  cruise_linear: {config.cruise_linear:.3f}')
    print(f'  loop: linear={config.loop_linear:.3f}, angular={config.loop_angular:.3f}')
    print(f'  return: linear={config.return_linear:.3f}, angular={config.return_angular:.3f}')
    print(f'  llm_mode: {config.llm_mode}')
    print(
        f'  route_mode: {config.route_mode}, '
        f'map_route_enabled={config.map_route_enabled}, '
        f'fixed_route_enabled={config.fixed_route_enabled}'
    )
    if args.force_direction:
        direction = args.force_direction
        direction_source = 'force_direction'
    else:
        sample_qr = args.mock_qr_content if args.mock_qr_content is not None else 'left'
        instruction = parse_qr_instruction(sample_qr)
        direction = instruction.direction
        direction_source = instruction.source
    flow = flow_from_start(args.start_state, direction)
    for state in flow:
        print(f'STATE: {state}')
        print(f'ZONE: {STATE_ZONES.get(state, "UNKNOWN")}')
        if state == 'DRIVE_TO_QR':
            if config.map_route_active():
                print_map_route('A_TO_QR')
            elif config.fixed_route_active():
                print_fixed_route(config, 'A_TO_QR')
            else:
                print(
                    f'ACTION: DRIVE_TO_QR linear.x={config.cruise_linear:.3f} '
                    f'approach-until-QR-visible max_distance={config.drive_to_qr_max_distance:.2f}m '
                    f'timeout={config.drive_to_qr_timeout:.1f}s'
                )
            print('VISION: dark-blue obstacle avoidance + QR board approach (OpenCV/YOLO)')
        elif state == 'SCAN_QR':
            print('ACTION: SCAN_QR stop and decode recent frames')
        elif state == 'DECIDE_DIRECTION':
            print(f'DIRECTION: {direction}')
            print(f'DIRECTION_SOURCE: {direction_source}')
            if direction not in ('left', 'right'):
                print('DECIDE_DIRECTION_RESULT: STOP_DIRECTION_UNKNOWN')
            else:
                if config.qr_align_after_scan_enabled:
                    print(
                        'ACTION: QR_ALIGN_HEADING '
                        f'target={config.qr_exit_heading_deg:.1f}deg '
                        f'mode={config.ackermann_turn_mode} '
                        f'tol={config.qr_align_heading_tol:.2f}rad'
                    )
                if config.map_route_active():
                    print_map_route('QR_TO_CORRIDOR')
                elif config.fixed_route_active():
                    print_fixed_route(config, 'QR_TO_CORRIDOR')
        elif state == 'CORRIDOR_FOLLOW':
            if config.map_route_active():
                print_map_route('B_CORRIDOR')
            elif config.fixed_route_active():
                print_fixed_route(config, 'B_CORRIDOR')
            else:
                print(
                    f'ACTION: CORRIDOR_FOLLOW yellow-lane base_linear={config.lane_base_linear:.3f} '
                    f'length={config.corridor_length_m:.1f}m timeout={config.corridor_timeout:.0f}s'
                )
        elif state in ('LOOP_LEFT', 'LOOP_RIGHT'):
            target_marker = 'MARKER_LEFT' if state == 'LOOP_LEFT' else 'MARKER_RIGHT'
            print(f'TARGET_MARKER: {target_marker} policy={config.target_marker_policy}')
            if config.map_route_active():
                print_map_route('C_LOOP_LEFT' if state == 'LOOP_LEFT' else 'C_LOOP_RIGHT')
            elif config.fixed_route_active():
                print_fixed_route(config, 'C_LOOP_LEFT' if state == 'LOOP_LEFT' else 'C_LOOP_RIGHT')
            elif config.lane_follow_enabled:
                print(
                    f'ACTION: {state} yellow-ring follow base_linear={config.lane_base_linear:.3f} '
                    f'complete={config.loop_complete_fraction * 360:.0f}deg '
                    f'timeout={config.loop_timeout:.0f}s (inline orange marker capture)'
                )
            else:
                sign = 1.0 if state == 'LOOP_LEFT' else -1.0
                print(
                    f'ACTION: {state} linear.x={config.loop_linear:.3f} '
                    f'angular.z={sign * config.loop_angular:.3f} '
                    f'duration={config.loop_duration:.2f}s'
                )
        elif state == 'CAPTURE_TARGET_IMAGE':
            print(f'ACTION: detect/crop image card, save under {config.debug_dir}')
        elif state == 'CALL_LLM_API':
            if config.llm_mode == 'placeholder':
                print('LLM_RESULT: LLM_PLACEHOLDER')
            elif config.llm_mode == 'disabled':
                print('LLM_RESULT: LLM_DISABLED')
            else:
                print(
                    'LLM_RESULT: OpenAI-compatible request at STOP state '
                    f'model={config.llm_model}'
                )
        elif state == 'RETURN_HOME':
            if config.return_mode == 'map2d_fallback':
                print('ACTION: RETURN_HOME map2d RETURN_TO_P, fallback=trajectory')
                print_map_route('RETURN_TO_P')
            elif config.return_mode == 'trajectory':
                print(
                    'ACTION: RETURN_HOME trajectory replay (reverse_plan), '
                    f'drive_linear={config.return_drive_linear:.3f} '
                    f'turn_angular={config.return_turn_angular:.3f} '
                    f'stride={config.return_waypoint_stride} '
                    f'total_timeout={config.return_total_timeout:.0f}s'
                )
            else:
                print(
                    f'ACTION: RETURN_HOME linear.x={config.return_linear:.3f} '
                    f'angular.z={config.return_angular:.3f} '
                    f'duration={config.return_duration:.2f}s'
                )
    print('dry-run: no ROS subscriptions or /cmd_vel messages created')
    return 0


def print_fixed_route(config: MissionConfig, name: str) -> None:
    route = config.fixed_routes.get(name, [])
    print(
        f'FIXED_ROUTE_START name={name} linear={config.fixed_route_linear:.3f} '
        f'turn_angular={config.fixed_route_turn_angular:.3f}'
    )
    for index, segment in enumerate(route, start=1):
        print(
            f'  FIXED_ROUTE_SEG {index}: '
            f'{json.dumps(segment, ensure_ascii=False, sort_keys=True)}'
        )


def print_map_route(name: str) -> None:
    route = DEFAULT_ROUTE_MAP.route(name)
    print(f'MAP_ROUTE_START name={name} waypoints={len(route)}')
    for index, waypoint in enumerate(route, start=1):
        print(
            f'  MAP_WAYPOINT {index}: '
            f'{waypoint.name} x={waypoint.x:.3f} y={waypoint.y:.3f}'
        )


def flow_from_start(start_state: str, direction: str) -> List[str]:
    state = start_state
    flow = []
    while state != 'DONE':
        flow.append(state)
        if state == 'IDLE':
            state = 'PREFLIGHT'
        elif state == 'PREFLIGHT':
            state = 'DRIVE_TO_QR'
        elif state == 'DRIVE_TO_QR':
            state = 'SCAN_QR'
        elif state == 'SCAN_QR':
            state = 'DECIDE_DIRECTION'
        elif state == 'DECIDE_DIRECTION':
            if direction in ('left', 'right'):
                state = 'CORRIDOR_FOLLOW'
            else:
                state = 'STOP'
        elif state == 'CORRIDOR_FOLLOW':
            if direction == 'left':
                state = 'LOOP_LEFT'
            elif direction == 'right':
                state = 'LOOP_RIGHT'
            else:
                state = 'STOP'
        elif state in ('LOOP_LEFT', 'LOOP_RIGHT'):
            state = 'CAPTURE_TARGET_IMAGE'
        elif state == 'CAPTURE_TARGET_IMAGE':
            state = 'CALL_LLM_API'
        elif state == 'CALL_LLM_API':
            state = 'RETURN_HOME'
        elif state == 'RETURN_HOME':
            state = 'STOP'
        elif state == 'STOP':
            state = 'DONE'
        else:
            state = 'STOP'
    flow.append('DONE')
    return flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the first OriginCar competition mission state machine.'
    )
    parser.add_argument('--config', help='Mission JSON config path.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print state/action flow without creating ROS IO.')
    parser.add_argument('--no-motion', action='store_true',
                        help='Suppress non-zero /cmd_vel while still running ROS IO.')
    parser.add_argument('--start-state', choices=STATE_SEQUENCE[:-1], default='IDLE',
                        help='State to start from for focused testing.')
    parser.add_argument('--max-runtime', type=float,
                        help='Override full mission timeout in seconds.')
    parser.add_argument('--drive-to-qr-timeout', type=float,
                        help='Override low-speed QR approach timeout.')
    parser.add_argument('--scan-qr-timeout', type=float,
                        help='Override QR scan timeout.')
    parser.add_argument('--capture-timeout', type=float,
                        help='Override target image capture timeout.')
    parser.add_argument('--save-debug-images', action='store_true',
                        help='Save QR debug frames in addition to target image.')
    parser.add_argument('--force-direction', choices=['left', 'right'],
                        help='Override QR direction parsing for debug runs.')
    parser.add_argument('--mock-qr-content',
                        help='Use this QR text instead of camera QR during SCAN_QR.')
    parser.add_argument('--debug-dir', help='Override debug image directory.')
    parser.add_argument('--cruise-linear', type=float,
                        help='Override low-speed forward linear.x.')
    parser.add_argument('--turn-angular', type=float,
                        help='Override general turn angular.z.')
    parser.add_argument('--loop-duration', type=float,
                        help='Override left/right loop placeholder duration.')
    parser.add_argument('--return-duration', type=float,
                        help='Override return-home placeholder duration.')
    parser.add_argument('--llm-mode',
                        choices=['placeholder', 'disabled', 'openai-compatible'],
                        help='Override image LLM mode.')
    parser.add_argument('--llm-api-url', help='OpenAI-compatible chat completions URL.')
    parser.add_argument('--llm-model', help='OpenAI-compatible model name.')
    parser.add_argument('--llm-api-key-env',
                        help='Environment variable containing the API key.')
    parser.add_argument('--llm-timeout', type=float,
                        help='Image LLM request timeout in seconds.')
    parser.add_argument('--llm-prompt', help='Prompt for target image analysis.')
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config.apply_args(args)
    config.validate()
    if args.dry_run:
        return run_dry_run(config, args)
    if rclpy is None:
        raise ValueError('rclpy is not available; run with --dry-run or on the ROS2 board')

    if SignalHandlerOptions is not None:
        rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    else:
        rclpy.init(args=None)
    node = AutoMissionNode(config, args)
    try:
        return node.run_mission()
    finally:
        try:
            node._stop_with_spin()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    try:
        return run(args)
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
