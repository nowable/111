#!/usr/bin/env python3
"""Read-only preflight checks for the OriginCar competition stack."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.utilities import remove_ros_args
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Float32
except ImportError:
    rclpy = None
    CompressedImage = None
    Float32 = None
    qos_profile_sensor_data = None

    class Node:  # type: ignore[no-redef]
        pass

    def remove_ros_args(args: List[str]) -> List[str]:
        return args


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


class SystemCheckNode(Node):
    def __init__(self, image_topic: str, voltage_topic: str) -> None:
        super().__init__('origin_system_check')
        self.image_count = 0
        self.latest_image_time: Optional[float] = None
        self.latest_voltage: Optional[float] = None
        self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(Float32, voltage_topic, self._voltage_callback, 10)

    def _image_callback(self, _msg: CompressedImage) -> None:
        self.image_count += 1
        self.latest_image_time = time.monotonic()

    def _voltage_callback(self, msg: Float32) -> None:
        self.latest_voltage = float(msg.data)


def service_active(service: str) -> CheckResult:
    try:
        proc = subprocess.run(
            ['systemctl', 'is-active', service],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return CheckResult(service, False, f'systemctl error: {exc}')
    status = proc.stdout.strip()
    return CheckResult(service, status == 'active', status or proc.stderr.strip())


def topic_count(node: Node, topic: str) -> Dict[str, int]:
    if hasattr(node, 'count_publishers') and hasattr(node, 'count_subscribers'):
        return {
            'publishers': int(node.count_publishers(topic)),
            'subscriptions': int(node.count_subscribers(topic)),
        }
    return {
        'publishers': len(node.get_publishers_info_by_topic(topic)),
        'subscriptions': len(node.get_subscriptions_info_by_topic(topic)),
    }


def wait_for_samples(node: SystemCheckNode, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.image_count > 0 and node.latest_voltage is not None:
            return


def wait_for_topic_graph(
    node: Node,
    topic: str,
    timeout: float,
    require_publisher: bool = False,
    require_subscription: bool = False,
) -> Dict[str, int]:
    deadline = time.monotonic() + max(0.0, timeout)
    counts = topic_count(node, topic)
    while time.monotonic() < deadline and rclpy.ok():
        publisher_ok = (not require_publisher) or counts['publishers'] > 0
        subscription_ok = (not require_subscription) or counts['subscriptions'] > 0
        if publisher_ok and subscription_ok:
            return counts
        rclpy.spin_once(node, timeout_sec=0.05)
        counts = topic_count(node, topic)
    return counts


def run_checks(args: argparse.Namespace) -> List[CheckResult]:
    if rclpy is None:
        return [CheckResult('rclpy', False, 'rclpy is not available')]

    rclpy.init(args=None)
    node = SystemCheckNode(args.image_topic, args.voltage_topic)
    results: List[CheckResult] = []
    try:
        results.append(service_active('origincar-base.service'))
        results.append(service_active('origincar-camera.service'))
        wait_for_samples(node, args.sample_timeout)

        image_counts = wait_for_topic_graph(
            node,
            args.image_topic,
            args.discovery_timeout,
            require_publisher=True,
        )
        results.append(
            CheckResult(
                args.image_topic,
                image_counts['publishers'] > 0 and node.image_count > 0,
                (
                    f'publishers={image_counts["publishers"]} '
                    f'subscriptions={image_counts["subscriptions"]} '
                    f'samples={node.image_count}'
                ),
            )
        )

        cmd_counts = wait_for_topic_graph(
            node,
            args.cmd_vel_topic,
            args.discovery_timeout,
            require_subscription=True,
        )
        cmd_ok = cmd_counts['subscriptions'] > 0
        if not args.allow_cmd_vel_publisher:
            cmd_ok = cmd_ok and cmd_counts['publishers'] == 0
        results.append(
            CheckResult(
                args.cmd_vel_topic,
                cmd_ok,
                (
                    f'publishers={cmd_counts["publishers"]} '
                    f'subscriptions={cmd_counts["subscriptions"]}'
                ),
            )
        )

        yolo_counts = wait_for_topic_graph(
            node,
            args.yolo_topic,
            args.discovery_timeout,
            require_publisher=args.require_yolo,
        )
        yolo_ok = True
        if args.require_yolo:
            yolo_ok = yolo_counts['publishers'] > 0
        results.append(
            CheckResult(
                args.yolo_topic,
                yolo_ok,
                (
                    f'publishers={yolo_counts["publishers"]} '
                    f'subscriptions={yolo_counts["subscriptions"]} '
                    f'required={args.require_yolo}'
                ),
            )
        )

        voltage_ok = node.latest_voltage is not None
        if node.latest_voltage is not None and args.min_voltage > 0.0:
            voltage_ok = node.latest_voltage >= args.min_voltage
        detail = (
            'no_sample'
            if node.latest_voltage is None
            else f'{node.latest_voltage:.2f}V min={args.min_voltage:.2f}V'
        )
        results.append(CheckResult(args.voltage_topic, voltage_ok, detail))
        return results
    finally:
        node.destroy_node()
        rclpy.shutdown()


def print_results(results: List[CheckResult], as_json: bool) -> int:
    failed = [result for result in results if not result.ok]
    if as_json:
        print(
            json.dumps(
                {
                    'ok': not failed,
                    'checks': [asdict(result) for result in results],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    else:
        for result in results:
            status = 'PASS' if result.ok else 'FAIL'
            print(f'{status} {result.name}: {result.detail}', flush=True)
        print('SYSTEM_CHECK_OK' if not failed else 'SYSTEM_CHECK_FAIL', flush=True)
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run read-only checks before OriginCar mission testing.'
    )
    parser.add_argument('--image-topic', default='/image')
    parser.add_argument('--cmd-vel-topic', default='/cmd_vel')
    parser.add_argument('--voltage-topic', default='/PowerVoltage')
    parser.add_argument('--yolo-topic', default='/hobot_dnn_detection')
    parser.add_argument('--sample-timeout', type=float, default=3.0)
    parser.add_argument('--discovery-timeout', type=float, default=3.0,
                        help='Seconds to wait for ROS graph discovery.')
    parser.add_argument('--min-voltage', type=float, default=10.5)
    parser.add_argument('--require-yolo', action='store_true',
                        help='Fail if YOLO detection topic has no publisher.')
    parser.add_argument('--allow-cmd-vel-publisher', action='store_true',
                        help='Do not fail when a /cmd_vel publisher already exists.')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    ros_stripped_args = remove_ros_args(args=sys.argv if argv is None else argv)
    args = parser.parse_args(ros_stripped_args[1:])
    if args.sample_timeout < 0.1:
        parser.error('sample-timeout must be >= 0.1')
    results = run_checks(args)
    return print_results(results, args.json)


if __name__ == '__main__':
    raise SystemExit(main())
