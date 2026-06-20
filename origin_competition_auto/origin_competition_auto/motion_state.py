#!/usr/bin/env python3
"""Odometry + IMU fusion helper for OriginCar missions.

Provides yaw (absolute + unwrapped-integrated), traveled distance, and pose
trajectory recording on top of /odom (nav_msgs/Odometry) and /imu/data
(sensor_msgs/Imu). Used by loop-completion (Phase C) and return-home (Phase F).

Both topics publish at ~20 Hz with RELIABLE QoS on the RDK X5, so the default
subscription QoS works. The pure math helpers (yaw_from_quaternion, unwrap,
reverse_plan) import without rclpy so they can be unit-tested offline.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:  # ROS is only present on the board; keep pure helpers importable anywhere.
    import rclpy  # noqa: F401
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu
    _HAS_ROS = True
except Exception:  # pragma: no cover - offline/dev machine
    Odometry = object  # type: ignore
    Imu = object  # type: ignore
    _HAS_ROS = False


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Planar yaw (rad, range (-pi, pi]) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b wrapped to (-pi, pi]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


@dataclass
class Segment:
    """One return-home move: rotate to target_heading, then drive distance (m)."""

    target_heading: float
    distance: float


def reverse_plan(
    trajectory: List[Tuple[float, float, float]],
    waypoint_stride: int = 1,
    min_segment_dist: float = 0.02,
) -> List[Segment]:
    """Build a reversed segment plan to drive back along an outbound trajectory.

    trajectory is an ordered list of (x, y, yaw) poses recorded on the way out.
    The plan walks the downsampled waypoints in reverse; each segment points at
    the next reversed waypoint and records the straight-line distance to it.
    Pure function (no ROS) so it is unit-testable offline.
    """
    if not trajectory or len(trajectory) < 2:
        return []
    stride = max(1, int(waypoint_stride))
    pts = trajectory[::stride]
    # trajectory[0] (home) is always kept by [::stride]; ensure the final
    # outbound point (current pose) survives downsampling so the reversed plan
    # starts where the robot actually is.
    if pts[-1] != trajectory[-1]:
        pts.append(trajectory[-1])
    reversed_pts = list(reversed(pts))
    segments: List[Segment] = []
    for i in range(len(reversed_pts) - 1):
        x0, y0, _ = reversed_pts[i]
        x1, y1, _ = reversed_pts[i + 1]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist < min_segment_dist:
            continue
        segments.append(Segment(target_heading=math.atan2(dy, dx), distance=dist))
    return segments


class MotionState:
    """Subscribes /odom + /imu/data and exposes fused yaw / distance / trajectory.

    Construct with a live rclpy Node. Subscriptions use the node's default QoS,
    which matches the board's RELIABLE odom/imu publishers. integrated_yaw()
    accumulates unwrapped heading change since reset() for loop completion;
    traveled_distance() accumulates planar arc length from odom positions.
    """

    def __init__(
        self,
        node,
        odom_topic: str = '/odom',
        imu_topic: str = '/imu/data',
        use_imu_yaw: bool = True,
        waypoint_min_dist: float = 0.05,
        waypoint_min_yaw: float = 0.10,
        stale_timeout: float = 1.0,
    ) -> None:
        self._node = node
        self._use_imu_yaw = use_imu_yaw
        self._waypoint_min_dist = waypoint_min_dist
        self._waypoint_min_yaw = waypoint_min_yaw
        self._stale_timeout = stale_timeout

        self._odom_yaw: Optional[float] = None
        self._imu_yaw: Optional[float] = None
        self._pos: Optional[Tuple[float, float]] = None
        self._odom_time: Optional[float] = None
        self._imu_time: Optional[float] = None

        self._integrated_yaw = 0.0
        self._last_yaw_for_integ: Optional[float] = None
        self._distance = 0.0
        self._last_pos_for_dist: Optional[Tuple[float, float]] = None

        self._recording = False
        self._trajectory: List[Tuple[float, float, float]] = []
        self._last_wp: Optional[Tuple[float, float, float]] = None

        self._odom_sub = None
        self._imu_sub = None
        if _HAS_ROS:
            self._odom_sub = node.create_subscription(
                Odometry, odom_topic, self._odom_cb, 10)
            if use_imu_yaw:
                self._imu_sub = node.create_subscription(
                    Imu, imu_topic, self._imu_cb, 10)

    def _now(self) -> float:
        if _HAS_ROS and self._node is not None:
            return self._node.get_clock().now().nanoseconds / 1e9
        import time
        return time.monotonic()

    def _odom_cb(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pos = (float(p.x), float(p.y))
        self._odom_yaw = yaw_from_quaternion(
            float(q.x), float(q.y), float(q.z), float(q.w))
        self._odom_time = self._now()
        self._on_update()

    def _imu_cb(self, msg) -> None:
        q = msg.orientation
        self._imu_yaw = yaw_from_quaternion(
            float(q.x), float(q.y), float(q.z), float(q.w))
        self._imu_time = self._now()
        self._on_update()

    def _yaw_source(self) -> Optional[float]:
        """Prefer fresh IMU yaw; fall back to odom yaw."""
        now = self._now()
        if (
            self._use_imu_yaw
            and self._imu_yaw is not None
            and self._imu_time is not None
            and now - self._imu_time <= self._stale_timeout
        ):
            return self._imu_yaw
        return self._odom_yaw

    def _on_update(self) -> None:
        yaw = self._yaw_source()
        # Integrate unwrapped heading change.
        if yaw is not None:
            if self._last_yaw_for_integ is not None:
                self._integrated_yaw += angle_diff(yaw, self._last_yaw_for_integ)
            self._last_yaw_for_integ = yaw
        # Accumulate planar distance from odom positions.
        if self._pos is not None:
            if self._last_pos_for_dist is not None:
                dx = self._pos[0] - self._last_pos_for_dist[0]
                dy = self._pos[1] - self._last_pos_for_dist[1]
                self._distance += math.hypot(dx, dy)
            self._last_pos_for_dist = self._pos
        self._maybe_record(yaw)

    def _maybe_record(self, yaw: Optional[float]) -> None:
        if not self._recording or self._pos is None:
            return
        x, y = self._pos
        cur_yaw = yaw if yaw is not None else 0.0
        if self._last_wp is None:
            self._trajectory.append((x, y, cur_yaw))
            self._last_wp = (x, y, cur_yaw)
            return
        lx, ly, lyaw = self._last_wp
        moved = math.hypot(x - lx, y - ly)
        turned = abs(angle_diff(cur_yaw, lyaw))
        if moved >= self._waypoint_min_dist or turned >= self._waypoint_min_yaw:
            self._trajectory.append((x, y, cur_yaw))
            self._last_wp = (x, y, cur_yaw)

    # ---- public API ----
    def absolute_yaw(self) -> Optional[float]:
        return self._yaw_source()

    def integrated_yaw(self) -> float:
        return self._integrated_yaw

    def traveled_distance(self) -> float:
        return self._distance

    def pose(self) -> Optional[Tuple[float, float, float]]:
        if self._pos is None:
            return None
        yaw = self._yaw_source() or 0.0
        return (self._pos[0], self._pos[1], yaw)

    def has_odom(self) -> bool:
        return (
            self._odom_time is not None
            and self._now() - self._odom_time <= self._stale_timeout
        )

    def has_imu(self) -> bool:
        return (
            self._imu_time is not None
            and self._now() - self._imu_time <= self._stale_timeout
        )

    def reset(self) -> None:
        """Zero the yaw/distance integrators and clear the recorded trajectory."""
        self._integrated_yaw = 0.0
        self._last_yaw_for_integ = self._yaw_source()
        self._distance = 0.0
        self._last_pos_for_dist = self._pos
        self._trajectory = []
        self._last_wp = None

    def reset_yaw(self) -> None:
        """Zero only the integrated-yaw accumulator (for loop-completion).

        Leaves trajectory and traveled distance intact so return-home replay
        keeps the full outbound path.
        """
        self._integrated_yaw = 0.0
        self._last_yaw_for_integ = self._yaw_source()

    def distance_marker(self) -> float:
        """Snapshot current traveled distance for relative measurement."""
        return self._distance

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False

    def trajectory(self) -> List[Tuple[float, float, float]]:
        return list(self._trajectory)


def main(argv: Optional[List[str]] = None) -> int:
    """Live debug: print integrated yaw / distance / trajectory length at 2 Hz.

    Push the robot by hand (wheels off ground) to verify sign conventions:
    turning the robot left/ccw should make integrated_yaw increase.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Debug MotionState odom/imu fusion.')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--imu-topic', default='/imu/data')
    parser.add_argument('--no-imu', action='store_true')
    parser.add_argument('--duration', type=float, default=20.0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if not _HAS_ROS:
        print('MOTION_STATE_ERROR: rclpy not available', flush=True)
        return 2

    import time
    rclpy.init(args=None)
    node = rclpy.create_node('motion_state_debug')
    ms = MotionState(
        node,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        use_imu_yaw=not args.no_imu,
    )
    ms.reset()
    ms.start_recording()
    deadline = time.monotonic() + args.duration
    next_print = 0.0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now >= next_print:
                next_print = now + 0.5
                ay = ms.absolute_yaw()
                print(
                    'MOTION_STATE '
                    f'odom={int(ms.has_odom())} imu={int(ms.has_imu())} '
                    f'abs_yaw={ay:.3f} ' if ay is not None else 'MOTION_STATE abs_yaw=None ',
                    end='', flush=True,
                )
                print(
                    f'integ_yaw={math.degrees(ms.integrated_yaw()):.1f}deg '
                    f'dist={ms.traveled_distance():.3f}m '
                    f'waypoints={len(ms.trajectory())}',
                    flush=True,
                )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())





