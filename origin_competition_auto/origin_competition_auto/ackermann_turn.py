#!/usr/bin/env python3
"""Near-pivot turning helper for front-steer Ackermann-style bases."""

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class ShuntTurnResult:
    reason: str
    turned_rad: float
    elapsed_s: float
    yaw_rate_rad_s: float
    cycles: int

    @property
    def ok(self) -> bool:
        return self.reason == 'yaw'


def execute_shunt_turn(
    target_angle_rad: float,
    *,
    motion_state,
    publish_velocity: Callable[[float, float], None],
    spin_sleep: Callable[[], None],
    forward_linear: float,
    reverse_linear: float,
    angular: float,
    heading_tol: float,
    timeout: float,
    pulse_s: float,
    stop_s: float,
    ok: Optional[Callable[[], bool]] = None,
    runtime_ok: Optional[Callable[[], bool]] = None,
) -> ShuntTurnResult:
    """Turn in place approximately using forward/reverse steering pulses.

    Positive target means left/ccw yaw; negative target means right/cw yaw.
    The reverse pulse keeps the same angular.z sign because Twist angular.z is
    the requested body yaw direction, not the raw steering-wheel direction. An
    Ackermann controller should convert negative linear velocity to the opposite
    steering angle internally.
    """
    target = float(target_angle_rad)
    sign = 1.0 if target >= 0.0 else -1.0
    forward = abs(float(forward_linear))
    reverse = -abs(float(reverse_linear))
    steer = abs(float(angular))
    pulse = max(0.01, float(pulse_s))
    stop = max(0.0, float(stop_s))
    deadline = time.monotonic() + max(0.01, float(timeout))
    start_time = time.monotonic()
    cycles = 0
    reason = 'timeout'

    def alive() -> bool:
        return (ok is None or ok()) and time.monotonic() < deadline

    def guard_ok() -> bool:
        return runtime_ok is None or runtime_ok()

    def turned() -> float:
        return float(motion_state.integrated_yaw())

    def complete() -> bool:
        current = turned()
        if abs(target - current) <= heading_tol:
            return True
        return target * current > 0.0 and abs(current) >= max(0.0, abs(target) - heading_tol)

    def publish_stop_for(duration: float) -> bool:
        end = min(deadline, time.monotonic() + duration)
        while time.monotonic() < end and (ok is None or ok()):
            publish_velocity(0.0, 0.0)
            spin_sleep()
            if not guard_ok():
                return False
        return True

    def pulse_motion(linear: float, angular_cmd: float, duration: float) -> bool:
        end = min(deadline, time.monotonic() + duration)
        while time.monotonic() < end and (ok is None or ok()):
            if not guard_ok():
                return False
            if complete():
                return True
            publish_velocity(linear, angular_cmd)
            spin_sleep()
        return complete()

    motion_state.reset_yaw()
    try:
        while alive():
            if not guard_ok():
                reason = 'runtime'
                break
            if complete():
                reason = 'yaw'
                break
            cycles += 1
            if pulse_motion(forward, sign * steer, pulse):
                reason = 'yaw'
                break
            publish_velocity(0.0, 0.0)
            if not publish_stop_for(stop):
                reason = 'runtime'
                break
            if pulse_motion(reverse, sign * steer, pulse):
                reason = 'yaw'
                break
            publish_velocity(0.0, 0.0)
            if not publish_stop_for(stop):
                reason = 'runtime'
                break
    finally:
        publish_velocity(0.0, 0.0)
    elapsed = max(1e-6, time.monotonic() - start_time)
    final_turn = turned()
    return ShuntTurnResult(
        reason=reason,
        turned_rad=final_turn,
        elapsed_s=elapsed,
        yaw_rate_rad_s=abs(final_turn) / elapsed,
        cycles=cycles,
    )


def estimated_turn_time_s(angle_rad: float, yaw_rate_rad_s: float) -> float:
    if yaw_rate_rad_s <= 1e-6:
        return 0.0
    return abs(float(angle_rad)) / yaw_rate_rad_s
