#!/usr/bin/env python3
"""Drive a fixed square to validate distance/yaw trajectory execution."""

import argparse
import atexit
import math
import pathlib
import signal
import sys
import time
from typing import List, Optional, Tuple

if __package__ in (None, ''):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.utilities import remove_ros_args

    HAS_ROS = True
except ImportError:
    rclpy = None
    Twist = None
    HAS_ROS = False

    class Node:  # type: ignore[no-redef]
        pass

    def remove_ros_args(args: List[str]) -> List[str]:  # type: ignore[no-redef]
        return args

from origin_competition_auto.ackermann_turn import (
    estimated_turn_time_s,
    execute_shunt_turn,
)
from origin_competition_auto.motion_state import MotionState, angle_diff


class SquareDriveNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('square_drive_test')
        self.args = args
        self.publisher = self.create_publisher(Twist, args.topic, 10)
        self.stop_requested = False
        self.motion_state = MotionState(
            self,
            odom_topic=args.odom_topic,
            imu_topic=args.imu_topic,
            use_imu_yaw=not args.no_imu,
            stale_timeout=args.stale_timeout,
        )

    def wait_ready(self) -> bool:
        deadline = time.monotonic() + self.args.wait_timeout
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if (
                self.publisher.get_subscription_count() > 0
                and self.motion_state.has_odom()
                and (self.args.no_imu or self.motion_state.has_imu())
            ):
                return True
        if self.publisher.get_subscription_count() <= 0:
            self.get_logger().error(f'no subscriber on {self.args.topic}')
        if not self.motion_state.has_odom():
            self.get_logger().error(f'no fresh odom on {self.args.odom_topic}')
        if not self.args.no_imu and not self.motion_state.has_imu():
            self.get_logger().error(f'no fresh imu on {self.args.imu_topic}')
        return False

    def publish_velocity(self, linear: float, angular: float) -> None:
        msg = Twist()
        if not self.args.no_motion:
            msg.linear.x = float(linear)
            msg.angular.z = float(angular)
        self.publisher.publish(msg)

    def stop(self) -> None:
        for _ in range(max(1, self.args.stop_repeat)):
            self.publish_velocity(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / self.args.rate)

    def request_stop(self) -> None:
        self.stop_requested = True
        self.stop()

    def spin_sleep(self) -> None:
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(1.0 / self.args.rate)

    def drive_side(self, index: int) -> bool:
        target = max(0.0, self.args.side)
        print(f'SQUARE_SIDE_START index={index} target={target:.3f}', flush=True)
        start = self.motion_state.distance_marker()
        deadline = time.monotonic() + self.args.segment_timeout
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok() and not self.stop_requested:
            traveled = self.motion_state.traveled_distance() - start
            if traveled >= max(0.0, target - self.args.dist_tol):
                reason = 'distance'
                break
            self.publish_velocity(self.args.linear, 0.0)
            self.spin_sleep()
        self.stop()
        traveled = self.motion_state.traveled_distance() - start
        print(
            f'SQUARE_SIDE_DONE index={index} reason={reason} traveled={traveled:.3f}',
            flush=True,
        )
        return reason == 'distance'

    def turn_corner(self, index: int) -> bool:
        sign = 1.0 if self.args.direction == 'left' else -1.0
        target = sign * (math.pi / 2.0)
        print(
            f'SQUARE_TURN_START index={index} mode={self.args.turn_mode} '
            f'target_deg={math.degrees(target):.0f}',
            flush=True,
        )
        if self.args.turn_mode == 'shunt':
            result = execute_shunt_turn(
                target,
                motion_state=self.motion_state,
                publish_velocity=self.publish_velocity,
                spin_sleep=self.spin_sleep,
                forward_linear=self.args.shunt_forward_linear,
                reverse_linear=self.args.shunt_reverse_linear,
                angular=self.args.angular,
                heading_tol=self.args.heading_tol,
                timeout=self.args.turn_timeout,
                pulse_s=self.args.shunt_pulse_s,
                stop_s=self.args.shunt_stop_s,
                ok=rclpy.ok,
            )
            self.stop()
            print(
                f'SQUARE_TURN_DONE index={index} mode=shunt '
                f'reason={result.reason} turned_deg={math.degrees(result.turned_rad):.1f} '
                f'elapsed={result.elapsed_s:.1f} yaw_rate={result.yaw_rate_rad_s:.3f}rad/s '
                f'estimated_90_s={estimated_turn_time_s(math.pi / 2.0, result.yaw_rate_rad_s):.1f} '
                f'cycles={result.cycles}',
                flush=True,
            )
            return result.ok
        self.motion_state.reset_yaw()
        start_time = time.monotonic()
        deadline = start_time + self.args.turn_timeout
        reason = 'timeout'
        while time.monotonic() < deadline and rclpy.ok() and not self.stop_requested:
            turned = self.motion_state.integrated_yaw()
            remaining = abs(target) - abs(turned)
            if remaining <= self.args.heading_tol:
                reason = 'yaw'
                break
            self.publish_velocity(self.args.turn_linear, sign * self.args.angular)
            self.spin_sleep()
        self.stop()
        elapsed = max(1e-6, time.monotonic() - start_time)
        turned = self.motion_state.integrated_yaw()
        yaw_rate = abs(turned) / elapsed
        estimated_90_s = (math.pi / 2.0) / yaw_rate if yaw_rate > 1e-6 else 0.0
        print(
            f'SQUARE_TURN_DONE index={index} reason={reason} '
            f'turned_deg={math.degrees(turned):.1f} '
            f'elapsed={elapsed:.1f} yaw_rate={yaw_rate:.3f}rad/s '
            f'estimated_90_s={estimated_90_s:.1f}',
            flush=True,
        )
        return reason == 'yaw'

    def run_square(self) -> int:
        if not self.wait_ready():
            self.stop()
            return 2
        self.motion_state.reset()
        start_pose = self._wait_pose()
        if start_pose is None:
            self.get_logger().error('no start pose available')
            self.stop()
            return 2
        print(
            f'SQUARE_START side={self.args.side:.3f} direction={self.args.direction}',
            flush=True,
        )
        failed = False
        for index in range(1, 5):
            if not self.drive_side(index):
                failed = True
                if self.args.stop_on_timeout:
                    break
            if not self.turn_corner(index):
                failed = True
                if self.args.stop_on_timeout:
                    break
        self.stop()
        final_pose = self.motion_state.pose()
        self._print_final_error(start_pose, final_pose)
        return 1 if failed else 0

    def _wait_pose(self) -> Optional[Tuple[float, float, float]]:
        deadline = time.monotonic() + self.args.wait_timeout
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            pose = self.motion_state.pose()
            if pose is not None:
                return pose
        return None

    def _print_final_error(
        self,
        start_pose: Tuple[float, float, float],
        final_pose: Optional[Tuple[float, float, float]],
    ) -> None:
        if final_pose is None:
            print('SQUARE_DONE final_pose=missing', flush=True)
            return
        dx = final_pose[0] - start_pose[0]
        dy = final_pose[1] - start_pose[1]
        final_error = math.hypot(dx, dy)
        yaw_error = angle_diff(final_pose[2], start_pose[2])
        print(
            'SQUARE_DONE '
            f'dx={dx:.3f} dy={dy:.3f} '
            f'final_error={final_error:.3f} '
            f'yaw_error_deg={math.degrees(yaw_error):.1f}',
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Drive a fixed square using odom distance and IMU/odom yaw.'
    )
    parser.add_argument('--side', type=float, default=1.0,
                        help='Square side length in meters.')
    parser.add_argument('--linear', type=float, default=0.05,
                        help='Forward linear speed.')
    parser.add_argument('--angular', type=float, default=0.40,
                        help='Turn angular speed.')
    parser.add_argument('--turn-linear', type=float, default=0.03,
                        help='Forward linear speed while turning; use 0 for pivot turn.')
    parser.add_argument('--turn-mode', choices=['shunt', 'arc'], default='shunt',
                        help='shunt uses forward/reverse steering pulses; arc is forward-only.')
    parser.add_argument('--shunt-forward-linear', type=float, default=0.060,
                        help='Forward speed for shunt turn pulses.')
    parser.add_argument('--shunt-reverse-linear', type=float, default=0.060,
                        help='Reverse speed magnitude for shunt turn pulses.')
    parser.add_argument('--shunt-pulse-s', type=float, default=1.0,
                        help='Duration of each forward/reverse shunt pulse.')
    parser.add_argument('--shunt-stop-s', type=float, default=0.20,
                        help='Stop pause between shunt pulses.')
    parser.add_argument('--heading-tol', type=float, default=0.08,
                        help='Turn completion tolerance in radians.')
    parser.add_argument('--dist-tol', type=float, default=0.03,
                        help='Side distance completion tolerance in meters.')
    parser.add_argument('--segment-timeout', type=float, default=20.0,
                        help='Timeout for each side.')
    parser.add_argument('--turn-timeout', type=float, default=45.0,
                        help='Timeout for each 90 degree turn.')
    parser.add_argument('--direction', choices=['left', 'right'], default='right',
                        help='Turn direction for square corners.')
    parser.add_argument('--topic', default='/cmd_vel')
    parser.add_argument('--odom-topic', default='/odom')
    parser.add_argument('--imu-topic', default='/imu/data')
    parser.add_argument('--no-imu', action='store_true',
                        help='Use odom yaw only; do not require IMU.')
    parser.add_argument('--stale-timeout', type=float, default=1.0)
    parser.add_argument('--wait-timeout', type=float, default=5.0)
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--stop-repeat', type=int, default=20)
    parser.add_argument('--stop-on-timeout', action='store_true',
                        help='Abort the sequence after the first timeout.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the planned sequence without creating ROS IO.')
    parser.add_argument('--no-motion', action='store_true',
                        help='Create ROS IO but publish zero velocities only.')
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.side <= 0.0:
        raise ValueError('--side must be > 0')
    if not 0.0 < abs(args.linear) <= 0.08:
        raise ValueError('--linear must be in (0, 0.08]')
    if not 0.0 <= abs(args.turn_linear) <= 0.08:
        raise ValueError('--turn-linear must be in [0, 0.08]')
    if not 0.0 < abs(args.shunt_forward_linear) <= 0.08:
        raise ValueError('--shunt-forward-linear must be in (0, 0.08]')
    if not 0.0 < abs(args.shunt_reverse_linear) <= 0.08:
        raise ValueError('--shunt-reverse-linear must be in (0, 0.08]')
    if not 0.0 < abs(args.angular) <= 0.5:
        raise ValueError('--angular must be in (0, 0.5]')
    if args.heading_tol <= 0.0 or args.dist_tol <= 0.0:
        raise ValueError('--heading-tol and --dist-tol must be > 0')
    if args.segment_timeout <= 0.0 or args.turn_timeout <= 0.0:
        raise ValueError('--segment-timeout and --turn-timeout must be > 0')
    if args.shunt_pulse_s <= 0.0 or args.shunt_stop_s < 0.0:
        raise ValueError('--shunt-pulse-s must be > 0 and --shunt-stop-s must be >= 0')
    if args.rate <= 0.0:
        raise ValueError('--rate must be > 0')
    if args.stop_repeat < 1:
        raise ValueError('--stop-repeat must be >= 1')


def print_plan(args: argparse.Namespace) -> None:
    sign = 1 if args.direction == 'left' else -1
    print('square_drive_test plan')
    print(f'  side: {args.side:.3f} m')
    print(f'  linear: {args.linear:.3f} m/s')
    print(f'  turn_linear: {args.turn_linear:.3f} m/s')
    print(f'  turn_mode: {args.turn_mode}')
    print(f'  shunt_forward_linear: {args.shunt_forward_linear:.3f} m/s')
    print(f'  shunt_reverse_linear: {args.shunt_reverse_linear:.3f} m/s')
    print(f'  shunt_pulse_s: {args.shunt_pulse_s:.2f} s')
    print(f'  shunt_stop_s: {args.shunt_stop_s:.2f} s')
    print('  shunt angular.z: same sign for forward and reverse pulses')
    print(f'  angular: {args.angular:.3f} rad/s')
    print(f'  direction: {args.direction}')
    for index in range(1, 5):
        print(f'  {index}: drive {args.side:.3f} m, turn {sign * 90} deg')


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    print_plan(args)
    if args.dry_run:
        print('dry-run: no ROS node created, no /cmd_vel messages published')
        return 0
    if not HAS_ROS:
        print('SQUARE_ERROR rclpy is not available; run this on the ROS2 board')
        return 2

    rclpy.init(args=None)
    node = SquareDriveNode(args)
    def _signal_stop(signum, _frame):
        node.get_logger().warn(f'signal {signum}; publishing stop')
        node.request_stop()
        raise KeyboardInterrupt
    previous_sigint = signal.signal(signal.SIGINT, _signal_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, _signal_stop)
    atexit.register(node.request_stop)
    try:
        return node.run_square()
    except KeyboardInterrupt:
        node.get_logger().warn('interrupted; publishing stop')
        node.request_stop()
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        node.request_stop()
        try:
            atexit.unregister(node.request_stop)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
