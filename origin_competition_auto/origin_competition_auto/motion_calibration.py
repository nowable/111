#!/usr/bin/env python3
"""Safe manual motion calibration helper for OriginCar."""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args


@dataclass(frozen=True)
class MotionStep:
    name: str
    linear_x: float
    angular_z: float
    duration: float


class MotionCalibrationNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__('motion_calibration')
        self.publisher = self.create_publisher(Twist, topic, 10)
        self.topic = topic

    def wait_for_subscriber(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.publisher.get_subscription_count() > 0:
                return True
        return self.publisher.get_subscription_count() > 0

    def publish_velocity(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.publisher.publish(msg)

    def publish_for(self, step: MotionStep, rate_hz: float) -> None:
        interval = 1.0 / rate_hz
        count = max(1, math.ceil(step.duration * rate_hz))
        self.get_logger().info(
            f'{step.name}: linear.x={step.linear_x:.3f}, '
            f'angular.z={step.angular_z:.3f}, duration={step.duration:.2f}s, '
            f'messages={count}'
        )
        for _ in range(count):
            self.publish_velocity(step.linear_x, step.angular_z)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(interval)

    def publish_stop(self, repeat: int, rate_hz: float) -> None:
        interval = 1.0 / rate_hz
        self.get_logger().info(f'stop: publishing zero velocity {repeat} times')
        for _ in range(max(1, repeat)):
            self.publish_velocity(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Publish short, bounded /cmd_vel motions for manual calibration.'
    )
    parser.add_argument(
        '--step',
        choices=['stop', 'forward', 'reverse', 'left', 'right', 'all'],
        default='stop',
        help='Calibration step to run. Default is stop for safety.',
    )
    parser.add_argument('--linear', type=float, default=0.03,
                        help='Base linear.x speed for motion steps.')
    parser.add_argument('--angular', type=float, default=0.2,
                        help='Base angular.z speed for turn steps.')
    parser.add_argument('--duration', type=float, default=0.5,
                        help='Motion duration in seconds for non-stop steps.')
    parser.add_argument('--stop-repeat', type=int, default=5,
                        help='Number of zero velocity messages after each step.')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Publish rate in Hz.')
    parser.add_argument('--topic', default='/cmd_vel',
                        help='Twist command topic.')
    parser.add_argument('--wait-subscriber', type=float, default=3.0,
                        help='Seconds to wait for a /cmd_vel subscriber.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the planned sequence without publishing.')
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.duration < 0.0:
        raise ValueError('--duration must be >= 0')
    if args.rate <= 0.0:
        raise ValueError('--rate must be > 0')
    if args.stop_repeat < 1:
        raise ValueError('--stop-repeat must be >= 1')
    if abs(args.linear) > 3.0:
        raise ValueError('--linear capped at 3.0 (fat-finger guard; well above chassis max)')
    if abs(args.angular) > 0.8:
        raise ValueError('--angular is intentionally limited to <= 0.8 for calibration')


def sequence_for(args: argparse.Namespace) -> List[MotionStep]:
    step = args.step
    linear = abs(args.linear)
    angular = abs(args.angular)
    duration = args.duration

    if step == 'stop':
        return []
    if step == 'forward':
        return [MotionStep('forward', linear, 0.0, duration)]
    if step == 'reverse':
        return [MotionStep('reverse', -linear, 0.0, duration)]
    if step == 'left':
        return [MotionStep('left', linear, angular, duration)]
    if step == 'right':
        return [MotionStep('right', linear, -angular, duration)]
    if step == 'all':
        return [
            MotionStep('forward', linear, 0.0, duration),
            MotionStep('reverse', -linear, 0.0, duration),
            MotionStep('left', linear, angular, duration),
            MotionStep('right', linear, -angular, duration),
        ]
    raise ValueError(f'unsupported step: {step}')


def print_plan(args: argparse.Namespace, steps: Iterable[MotionStep]) -> None:
    print('motion_calibration plan')
    print(f'  topic: {args.topic}')
    print(f'  rate: {args.rate} Hz')
    print(f'  stop_repeat: {args.stop_repeat}')
    if args.step == 'stop':
        print('  action: stop only')
        return
    for step in steps:
        print(
            f'  action: {step.name}, linear.x={step.linear_x:.3f}, '
            f'angular.z={step.angular_z:.3f}, duration={step.duration:.2f}s'
        )
        print('  then: stop')


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    steps = sequence_for(args)
    print_plan(args, steps)
    if args.dry_run:
        print('dry-run: no messages published')
        return 0

    rclpy.init(args=None)
    node = MotionCalibrationNode(args.topic)
    try:
        if not node.wait_for_subscriber(args.wait_subscriber):
            node.get_logger().error(
                f'no subscriber on {args.topic}; aborting motion publish'
            )
            node.publish_stop(args.stop_repeat, args.rate)
            return 2

        if args.step == 'stop':
            node.publish_stop(args.stop_repeat, args.rate)
            return 0

        node.publish_stop(args.stop_repeat, args.rate)
        for step in steps:
            node.publish_for(step, args.rate)
            node.publish_stop(args.stop_repeat, args.rate)
        node.get_logger().info('calibration sequence finished')
        return 0
    except KeyboardInterrupt:
        node.get_logger().warn('interrupted; publishing stop')
        node.publish_stop(args.stop_repeat, args.rate)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
