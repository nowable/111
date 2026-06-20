#!/usr/bin/env python3
"""Yellow lane-following controller for OriginCar corridor and ring driving.

Pure image-in / command-out design: LaneFollower.compute(image) returns a
LaneCommand with linear/angular velocities derived from the centroid of the
yellow mask in a bottom ROI band. No ROS dependency, so it is unit-testable
offline against synthetic masks and saved frames.

Used by Phase C states:
- CORRIDOR_FOLLOW: drive the straight 1m yellow corridor (Zone B).
- LOOP_LEFT / LOOP_RIGHT: follow the 0.5m yellow ring around the forbidden
  green clinic (Zone C), biased toward the inner edge per direction.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class LaneFollowConfig:
    roi_y_ratio: float = 0.6        # ROI starts at this fraction of image height
    roi_height_ratio: float = 0.35  # ROI band height as fraction of image height
    kp: float = 0.6                 # proportional gain on normalized offset
    kd: float = 0.1                 # derivative gain on offset change
    base_linear: float = 0.03       # forward speed when lane is found (<=0.08)
    max_angular: float = 0.5        # angular clamp (matches MotionCommander)
    min_mask_area_ratio: float = 0.01  # below this the lane is considered lost
    bias: float = 0.0               # steering bias; >0 hugs right, <0 hugs left
    side_mode: str = 'center'       # 'center' | 'left_edge' | 'right_edge'
    mask_erode_px: int = 0          # erode the color mask by this kernel (px) before
                                    # centroid; removes the thin yellow grid-WALL lines
                                    # (黄白相间网格) so only the solid lane survives. 0=off.


@dataclass
class LaneCommand:
    linear: float
    angular: float
    lane_found: bool
    offset_norm: float       # -1 (lane far left) .. +1 (lane far right)
    mask_area_ratio: float


class LaneFollower:
    """Steer toward the yellow lane centroid in a bottom ROI band.

    detector must expose color_mask(image, profile_name, roi_y_ratio) -> (mask,
    y_offset), as implemented by vision_detector.VisionDetector. Sharing that
    method guarantees the follower and the detector use identical HSV thresholds.
    """

    def __init__(self, detector, color_name: str, config: Optional[LaneFollowConfig] = None) -> None:
        self.detector = detector
        self.color_name = color_name
        self.config = config or LaneFollowConfig()
        self._prev_offset: Optional[float] = None

    def reset(self) -> None:
        self._prev_offset = None

    def _clamp_angular(self, value: float) -> float:
        limit = self.config.max_angular
        if value > limit:
            return limit
        if value < -limit:
            return -limit
        return value

    def compute(self, image: np.ndarray) -> LaneCommand:
        cfg = self.config
        h, w = image.shape[:2]
        # Build the yellow mask over a bottom band [roi_y .. roi_y+roi_h].
        roi_y = int(h * cfg.roi_y_ratio)
        roi_h = max(1, int(h * cfg.roi_height_ratio))
        y1 = min(h, roi_y + roi_h)
        band = image[roi_y:y1, :]
        mask, _ = self.detector.color_mask(band, self.color_name, roi_y_ratio=None)

        # Erode away the thin yellow grid-wall lines so the solid lane dominates
        # the centroid (field 6-14: car tracked the 黄白相间 grid wall, not the lane).
        if cfg.mask_erode_px > 0:
            k = int(cfg.mask_erode_px)
            mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)

        band_area = float(max(1, mask.shape[0] * mask.shape[1]))
        mask_px = float((mask > 0).sum())
        area_ratio = mask_px / band_area
        if area_ratio < cfg.min_mask_area_ratio:
            self._prev_offset = None
            return LaneCommand(0.0, 0.0, False, 0.0, area_ratio)

        offset = self._lane_offset(mask, w)
        d_offset = 0.0 if self._prev_offset is None else (offset - self._prev_offset)
        self._prev_offset = offset
        # Negative angular steers right (matches Twist convention used elsewhere):
        # lane to the right (offset>0) -> turn right (negative angular).
        angular = -(cfg.kp * offset + cfg.kd * d_offset) + cfg.bias
        angular = self._clamp_angular(angular)
        return LaneCommand(cfg.base_linear, angular, True, offset, area_ratio)

    def _lane_offset(self, mask: np.ndarray, image_width: int) -> float:
        """Normalized horizontal offset of the lane in [-1, +1].

        center mode: column-weighted centroid of all yellow pixels.
        left_edge / right_edge: track the left- or right-most yellow column,
        used on the ring so the robot hugs the inner edge of the lane.
        """
        cols = np.where(mask.max(axis=0) > 0)[0]
        if cols.size == 0:
            return 0.0
        mode = self.config.side_mode
        if mode == 'left_edge':
            target_col = float(cols.min())
        elif mode == 'right_edge':
            target_col = float(cols.max())
        else:
            # weighted centroid over column sums (robust to thin/thick lanes)
            col_weights = mask.sum(axis=0).astype(np.float64)
            total = col_weights.sum()
            if total <= 0:
                return 0.0
            target_col = float((np.arange(mask.shape[1]) * col_weights).sum() / total)
        half = image_width / 2.0
        return (target_col - half) / half


def main(argv=None) -> int:
    """CLI: run lane following on a saved image and print the command.

    Reuses VisionDetector for color_mask so thresholds match the mission.
    """
    import argparse
    import json
    import sys
    import cv2

    try:
        from origin_competition_auto.vision_detector import (
            VisionDetector, load_detector_config)
    except ImportError:
        from vision_detector import VisionDetector, load_detector_config

    parser = argparse.ArgumentParser(description='Debug yellow lane following.')
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--config', help='Detector JSON config (HSV profiles).')
    parser.add_argument('--color', default='yellow')
    parser.add_argument('--side-mode', default='center',
                        choices=['center', 'left_edge', 'right_edge'])
    parser.add_argument('--bias', type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    image = cv2.imread(args.input_file)
    if image is None:
        print(f'INPUT_IMAGE_ERROR: {args.input_file}', flush=True)
        return 2
    detector = VisionDetector(load_detector_config(args.config))
    follower = LaneFollower(
        detector, args.color,
        LaneFollowConfig(bias=args.bias, side_mode=args.side_mode))
    cmd = follower.compute(image)
    print(json.dumps({
        'lane_found': cmd.lane_found,
        'linear': round(cmd.linear, 4),
        'angular': round(cmd.angular, 4),
        'offset_norm': round(cmd.offset_norm, 4),
        'mask_area_ratio': round(cmd.mask_area_ratio, 4),
    }, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



