#!/usr/bin/env python3
"""Static 2D field map and waypoint routes for the hospital competition."""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float


def angle_diff(target: float, source: float) -> float:
    diff = target - source
    while diff > math.pi:
        diff -= 2.0 * math.pi
    while diff <= -math.pi:
        diff += 2.0 * math.pi
    return diff


def pose_from_mapping(data: object, fallback: Pose2D) -> Pose2D:
    if not isinstance(data, dict):
        return fallback
    return Pose2D(
        x=float(data.get('x', fallback.x)),
        y=float(data.get('y', fallback.y)),
        yaw=math.radians(float(data.get('heading_deg', math.degrees(fallback.yaw)))),
    )


class RouteMap:
    """Fixed 5m x 5m competition map.

    Coordinate frame:
    - origin: lower-left corner of area A on the provided field map
    - x: right
    - y: upward toward B/C
    """

    def __init__(self) -> None:
        self.width_m = 5.0
        self.height_m = 5.0
        self.start_pose = Pose2D(0.284, 0.298, math.radians(3.1))
        self.waypoints: Dict[str, Waypoint] = {}
        self.routes: Dict[str, List[str]] = {}
        self._build()

    def _add(self, name: str, x: float, y: float) -> None:
        self.waypoints[name] = Waypoint(name, x, y)

    def _build(self) -> None:
        a_points: List[Tuple[float, float]] = [
            (0.284, 0.298),
            (0.596, 0.320),
            (1.249, 0.360),
            (1.788, 0.450),
            (2.327, 0.507),
            (2.866, 0.690),
            (3.360, 0.873),
            (3.802, 1.092),
            (4.115, 1.317),
            (4.313, 1.541),
            (4.427, 1.813),
        ]
        for index, (x, y) in enumerate(a_points):
            self._add('P_START' if index == 0 else f'A_DASH_{index:02d}', x, y)
        self._add('QR_BOARD', a_points[-1][0], a_points[-1][1])

        connector = [
            ('QR_BACK_01', 3.916, 1.813),
            ('QR_BACK_02', 3.320, 1.813),
            ('QR_BACK_03', 2.838, 1.823),
            ('B_ENTRY', 2.384, 1.823),
            ('B_MID_01', 2.384, 2.283),
            ('B_MID_02', 2.384, 2.753),
            ('C_ENTRY', 2.384, 3.171),
        ]
        for name, x, y in connector:
            self._add(name, x, y)

        # User-provided red safe centerline around the C yellow passage.
        c_loop_right = [
            ('C_R_01', 2.838, 3.197),
            ('C_R_02', 3.405, 3.192),
            ('C_R_03', 3.802, 3.197),
            ('C_R_04', 4.030, 3.276),
            ('C_R_05', 4.171, 3.433),
            ('C_R_06', 4.257, 3.642),
            ('C_R_07', 4.245, 3.851),
            ('C_R_08', 4.171, 4.060),
            ('C_R_09', 4.041, 4.216),
            ('C_R_10', 3.859, 4.321),
            ('C_R_11', 3.575, 4.363),
            ('C_R_12', 3.121, 4.363),
            ('C_R_13', 2.554, 4.347),
            ('C_R_14', 1.986, 4.347),
            ('C_R_15', 1.476, 4.394),
            ('C_R_16', 1.192, 4.394),
            ('C_R_17', 0.908, 4.331),
            ('C_R_18', 0.738, 4.216),
            ('C_R_19', 0.624, 4.033),
            ('C_R_20', 0.545, 3.772),
            ('C_R_21', 0.545, 3.433),
            ('C_R_22', 0.624, 3.250),
            ('C_R_23', 0.851, 3.150),
            ('C_R_24', 1.419, 3.140),
            ('C_R_25', 1.986, 3.145),
            ('C_R_26', 2.213, 3.119),
            ('C_R_27', 2.384, 2.989),
            ('C_R_28', 2.384, 3.171),
        ]
        for name, x, y in c_loop_right:
            self._add(name, x, y)

        c_loop_left = [
            ('C_L_01', 2.213, 3.119),
            ('C_L_02', 1.986, 3.145),
            ('C_L_03', 1.419, 3.140),
            ('C_L_04', 0.851, 3.150),
            ('C_L_05', 0.624, 3.250),
            ('C_L_06', 0.545, 3.433),
            ('C_L_07', 0.545, 3.772),
            ('C_L_08', 0.624, 4.033),
            ('C_L_09', 0.738, 4.216),
            ('C_L_10', 0.908, 4.331),
            ('C_L_11', 1.192, 4.394),
            ('C_L_12', 1.476, 4.394),
            ('C_L_13', 1.986, 4.347),
            ('C_L_14', 2.554, 4.347),
            ('C_L_15', 3.121, 4.363),
            ('C_L_16', 3.575, 4.363),
            ('C_L_17', 3.859, 4.321),
            ('C_L_18', 4.041, 4.216),
            ('C_L_19', 4.171, 4.060),
            ('C_L_20', 4.245, 3.851),
            ('C_L_21', 4.257, 3.642),
            ('C_L_22', 4.171, 3.433),
            ('C_L_23', 4.030, 3.276),
            ('C_L_24', 3.802, 3.197),
            ('C_L_25', 3.405, 3.192),
            ('C_L_26', 2.838, 3.197),
            ('C_L_27', 2.384, 3.171),
        ]
        for name, x, y in c_loop_left:
            self._add(name, x, y)

        # Directional target marker positions from the user's blue circles.
        # RIGHT is the upper-right marker; LEFT is the upper-left marker.
        self._add('MARKER_RIGHT', 4.041, 4.216)
        self._add('MARKER_LEFT', 0.738, 4.216)

        return_to_p = [
            ('RETURN_DASH_01', 2.384, 1.793),
            ('RETURN_DASH_02', 2.327, 1.761),
            ('RETURN_DASH_03', 2.185, 1.604),
            ('RETURN_DASH_04', 1.986, 1.499),
            ('RETURN_DASH_05', 1.703, 1.343),
            ('RETURN_DASH_06', 1.362, 1.217),
            ('RETURN_DASH_07', 1.022, 1.082),
            ('RETURN_DASH_08', 0.681, 0.873),
            ('RETURN_DASH_09', 0.426, 0.637),
        ]
        for name, x, y in return_to_p:
            self._add(name, x, y)

        self.routes = {
            'A_TO_QR': ['P_START'] + [f'A_DASH_{i:02d}' for i in range(1, 10)] + ['QR_BOARD'],
            'QR_TO_CORRIDOR': ['QR_BOARD', 'QR_BACK_01', 'QR_BACK_02', 'QR_BACK_03', 'B_ENTRY'],
            'B_CORRIDOR': ['B_ENTRY', 'B_MID_01', 'B_MID_02', 'C_ENTRY'],
            # C_R_27 下探点造成 +127° 发卡弯且与返航衔接成 180° 掉头，路线中剔除
            'C_LOOP_RIGHT': ['C_ENTRY'] + [name for name, _, _ in c_loop_right if name != 'C_R_27'],
            'C_LOOP_LEFT': ['C_ENTRY'] + [name for name, _, _ in c_loop_left],
            'RETURN_TO_P': (
                # B 区通道复用去程中点（消除 C_ENTRY→B_ENTRY 1.348m 无航点长段）
                ['C_ENTRY', 'B_MID_02', 'B_MID_01', 'B_ENTRY']
                + [name for name, _, _ in return_to_p]
                + ['P_START']
            ),
        }

    def route(self, name: str) -> List[Waypoint]:
        return [self.waypoints[key] for key in self.routes.get(name, [])]

    def route_names(self) -> List[str]:
        return sorted(self.routes)


def map_pose_from_odom(
    odom_pose: Tuple[float, float, float],
    odom_start: Tuple[float, float, float],
    map_start: Pose2D,
) -> Pose2D:
    ox, oy, oyaw = odom_pose
    sx, sy, syaw = odom_start
    dx = ox - sx
    dy = oy - sy
    c = math.cos(map_start.yaw - syaw)
    s = math.sin(map_start.yaw - syaw)
    mx = map_start.x + c * dx - s * dy
    my = map_start.y + s * dx + c * dy
    myaw = map_start.yaw + angle_diff(oyaw, syaw)
    return Pose2D(mx, my, myaw)


DEFAULT_ROUTE_MAP = RouteMap()
