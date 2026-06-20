#!/usr/bin/env python3
"""Reusable OpenCV and YOLO-style vision helpers for OriginCar missions."""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ColorProfile:
    """HSV threshold band for a named color (OpenCV HSV: H 0-179, S/V 0-255).

    When wrap_hue is True the hue band wraps past 179 back through 0, which is
    needed for red/orange that straddle the 0 boundary. In that case the mask is
    the OR of [h_min..179] and [0..h_max].
    """

    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255
    wrap_hue: bool = False

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> 'ColorProfile':
        values = {}
        for key in cls.__dataclass_fields__:
            if key in data:
                values[key] = data[key]
        return cls(**values)

    def lower_upper(self) -> Tuple[np.ndarray, np.ndarray]:
        """Primary HSV lower/upper bound pair (non-wrapped portion)."""
        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        return lower, upper

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        if not self.wrap_hue:
            lower, upper = self.lower_upper()
            return cv2.inRange(hsv, lower, upper)
        lower_a = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper_a = np.array([179, self.s_max, self.v_max], dtype=np.uint8)
        lower_b = np.array([0, self.s_min, self.v_min], dtype=np.uint8)
        upper_b = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        return cv2.bitwise_or(
            cv2.inRange(hsv, lower_a, upper_a),
            cv2.inRange(hsv, lower_b, upper_b),
        )


def default_color_profiles() -> Dict[str, ColorProfile]:
    """Field-day starting HSV bands. Tune on real frames; these are blind guesses."""
    return {
        # Dark-blue triangular obstacles (NOT black). H~100-130 is blue in OpenCV.
        'dark_blue': ColorProfile(h_min=100, h_max=130, s_min=80, s_max=255, v_min=40, v_max=255),
        # Yellow floor lane / corridor / ring.
        'yellow': ColorProfile(h_min=20, h_max=35, s_min=80, s_max=255, v_min=80, v_max=255),
        # Orange image/text marker boards at ring corners.
        'orange': ColorProfile(h_min=8, h_max=20, s_min=110, s_max=255, v_min=120, v_max=255),
        # Light-green forbidden clinic floor (used as an exclusion/exit cue).
        'green': ColorProfile(h_min=40, h_max=85, s_min=40, s_max=255, v_min=60, v_max=255),
        # Legacy black band kept for backward compatibility with vision_tune.
        'black': ColorProfile(h_min=0, h_max=179, s_min=20, s_max=255, v_min=0, v_max=85),
    }


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    source: str = 'opencv'

    @property
    def area(self) -> int:
        return max(0, self.bbox[2]) * max(0, self.bbox[3])

    @property
    def center_x(self) -> float:
        return self.bbox[0] + self.bbox[2] / 2.0

    @property
    def center_y(self) -> float:
        return self.bbox[1] + self.bbox[3] / 2.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class VisionDecision:
    obstacle_danger: bool = False
    obstacle_zone: str = 'clear'
    obstacle_area_ratio: float = 0.0
    obstacle: Optional[Detection] = None
    target_card: Optional[Detection] = None
    detections: Optional[List[Detection]] = None

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data['obstacle'] = self.obstacle.to_dict() if self.obstacle else None
        data['target_card'] = self.target_card.to_dict() if self.target_card else None
        data['detections'] = [
            detection.to_dict() for detection in (self.detections or [])
        ]
        return data


@dataclass
class DetectorConfig:
    obstacle_roi_y_ratio: float = 0.45
    obstacle_min_area_ratio: float = 0.015
    obstacle_danger_area_ratio: float = 0.035
    obstacle_center_band_ratio: float = 0.36
    black_v_max: int = 85
    black_s_min: int = 20
    target_min_area_ratio: float = 0.025
    target_white_s_max: int = 90
    target_white_v_min: int = 150
    yolo_min_confidence: float = 0.45
    # Named HSV color profiles; obstacle/lane/marker pick from these by name.
    color_profiles: Dict[str, ColorProfile] = field(default_factory=default_color_profiles)
    obstacle_color: str = 'dark_blue'
    lane_color: str = 'yellow'
    marker_color: str = 'orange'
    green_color: str = 'green'
    # Optional triangle shape gate for obstacles (off by default; risky blind).
    require_triangle: bool = False
    triangle_min_vertices: int = 3
    triangle_max_vertices: int = 6
    yolo_obstacle_labels: Tuple[str, ...] = (
        'black_obstacle', 'dark_blue_obstacle', 'triangle', 'obstacle',
    )
    yolo_qr_labels: Tuple[str, ...] = ('qr_board', 'qrcode', 'qr')
    yolo_target_labels: Tuple[str, ...] = (
        'image_card', 'target_card', 'orange_marker', 'marker_board', 'marker',
    )

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> 'DetectorConfig':
        values = {}
        for key in cls.__dataclass_fields__:
            if key not in data:
                continue
            value = data[key]
            if key == 'color_profiles' and isinstance(value, dict):
                profiles = default_color_profiles()
                for name, raw in value.items():
                    if isinstance(raw, ColorProfile):
                        profiles[name] = raw
                    elif isinstance(raw, dict):
                        profiles[name] = ColorProfile.from_mapping(raw)
                value = profiles
            elif key.endswith('_labels') and isinstance(value, list):
                value = tuple(str(item) for item in value)
            values[key] = value
        return cls(**values)

    def profile(self, name: str) -> ColorProfile:
        return self.color_profiles.get(name) or ColorProfile()


class VisionDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def analyze(
        self,
        image: np.ndarray,
        yolo_detections: Optional[Sequence[Detection]] = None,
        color_obstacles: bool = True,
    ) -> VisionDecision:
        """color_obstacles=False skips HSV obstacle detection (YOLO primary)."""
        detections: List[Detection] = []
        if color_obstacles:
            detections.extend(self.detect_obstacles(image))
        detections.extend(self.detect_target_cards(image))
        if yolo_detections:
            detections.extend(
                detection for detection in yolo_detections
                if detection.confidence >= self.config.yolo_min_confidence
            )

        h, w = image.shape[:2]
        obstacle = self._best_obstacle(detections)
        target_card = self._best_target_card(detections)
        danger = False
        zone = 'clear'
        area_ratio = 0.0
        if obstacle:
            area_ratio = obstacle.area / float(max(1, w * h))
            zone = self._zone_for_bbox(obstacle.bbox, w)
            # Center obstacles use the SAME tunable danger threshold as the sides;
            # the bare min-area floor made avoidance fire far too early (field 6-17).
            danger = area_ratio >= self.config.obstacle_danger_area_ratio
        return VisionDecision(
            obstacle_danger=danger,
            obstacle_zone=zone,
            obstacle_area_ratio=area_ratio,
            obstacle=obstacle,
            target_card=target_card,
            detections=detections,
        )

    def color_mask(
        self,
        image: np.ndarray,
        profile_name: str,
        roi_y_ratio: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Return (mask, y_offset) for a named color profile.

        roi_y_ratio crops to the lower portion of the frame (0 = full frame).
        y_offset is the pixel row where the mask/ROI starts, so callers can map
        ROI-local coordinates back to full-image coordinates. Shared by the lane
        follower so steering uses the exact same thresholds as detection.
        """
        h = image.shape[0]
        y0 = int(h * roi_y_ratio) if roi_y_ratio else 0
        roi = image[y0:h, :] if y0 else image
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = self.config.profile(profile_name).mask(hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, y0

    def detect_color_regions(
        self,
        image: np.ndarray,
        profile_name: str,
        label: str,
        source: str,
        roi_y_ratio: Optional[float],
        min_area_ratio: float,
        require_triangle: bool = False,
    ) -> List[Detection]:
        h, w = image.shape[:2]
        mask, y0 = self.color_mask(image, profile_name, roi_y_ratio)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        min_area = min_area_ratio * w * h
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if require_triangle and not self._is_triangleish(contour):
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    label=label,
                    confidence=min(1.0, area / max(1.0, min_area * 3.0)),
                    bbox=(int(x), int(y + y0), int(bw), int(bh)),
                    source=source,
                )
            )
        return sorted(detections, key=lambda item: item.area, reverse=True)

    def _is_triangleish(self, contour: np.ndarray) -> bool:
        peri = cv2.arcLength(contour, True)
        if peri <= 0:
            return False
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        return (
            self.config.triangle_min_vertices
            <= len(approx)
            <= self.config.triangle_max_vertices
        )

    def detect_obstacles(self, image: np.ndarray) -> List[Detection]:
        """Detect the configured obstacle color (default dark_blue) in the lower ROI."""
        return self.detect_color_regions(
            image,
            profile_name=self.config.obstacle_color,
            label=f'{self.config.obstacle_color}_obstacle',
            source=f'opencv_{self.config.obstacle_color}',
            roi_y_ratio=self.config.obstacle_roi_y_ratio,
            min_area_ratio=self.config.obstacle_min_area_ratio,
            require_triangle=self.config.require_triangle,
        )

    def detect_black_obstacles(self, image: np.ndarray) -> List[Detection]:
        """Backward-compatible alias kept so vision_tune and old configs work.

        Uses the legacy black HSV band (black_s_min/black_v_max) directly rather
        than the named profile, preserving prior behavior exactly.
        """
        h, w = image.shape[:2]
        y0 = int(h * self.config.obstacle_roi_y_ratio)
        roi = image[y0:h, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, self.config.black_s_min, 0], dtype=np.uint8),
            np.array([179, 255, self.config.black_v_max], dtype=np.uint8),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: List[Detection] = []
        min_area = self.config.obstacle_min_area_ratio * w * h
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            detections.append(
                Detection(
                    label='black_obstacle',
                    confidence=min(1.0, area / max(1.0, min_area * 3.0)),
                    bbox=(int(x), int(y + y0), int(bw), int(bh)),
                    source='opencv_black',
                )
            )
        return sorted(detections, key=lambda item: item.area, reverse=True)

    def detect_target_cards(self, image: np.ndarray) -> List[Detection]:
        """Detect the configured marker color (default orange) image/text boards."""
        h, w = image.shape[:2]
        detections = self.detect_color_regions(
            image,
            profile_name=self.config.marker_color,
            label=f'{self.config.marker_color}_marker',
            source=f'opencv_{self.config.marker_color}',
            roi_y_ratio=None,
            min_area_ratio=self.config.target_min_area_ratio,
        )
        filtered: List[Detection] = []
        for detection in detections:
            _, _, bw, bh = detection.bbox
            aspect = bw / float(max(1, bh))
            if not 0.35 <= aspect <= 3.2:
                continue
            filtered.append(detection)
        return filtered

    def crop_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        pad_ratio: float = 0.08,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        x, y, bw, bh = detection.bbox
        pad = int(max(bw, bh) * pad_ratio)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        return image[y1:y2, x1:x2].copy()

    def draw_debug(
        self,
        image: np.ndarray,
        decision: VisionDecision,
    ) -> np.ndarray:
        output = image.copy()
        for detection in decision.detections or []:
            color = (0, 0, 255) if 'obstacle' in detection.label else (0, 255, 255)
            if detection.source.startswith('yolo'):
                color = (255, 0, 0)
            x, y, w, h = detection.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            text = f'{detection.label}:{detection.confidence:.2f}'
            cv2.putText(
                output,
                text,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        status = (
            f'obstacle={decision.obstacle_zone} '
            f'danger={int(decision.obstacle_danger)} '
            f'area={decision.obstacle_area_ratio:.3f}'
        )
        cv2.putText(
            output,
            status,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return output

    def _best_obstacle(self, detections: Iterable[Detection]) -> Optional[Detection]:
        labels = set(self.config.yolo_obstacle_labels)
        candidates = [
            detection for detection in detections
            if detection.label in labels or 'obstacle' in detection.label
        ]
        return max(candidates, key=lambda item: item.area, default=None)

    def _best_target_card(self, detections: Iterable[Detection]) -> Optional[Detection]:
        labels = set(self.config.yolo_target_labels)
        candidates = [
            detection for detection in detections
            if detection.label in labels
            or detection.label == 'image_card'
            or 'marker' in detection.label
        ]
        return max(candidates, key=lambda item: item.area, default=None)

    def _zone_for_bbox(self, bbox: Tuple[int, int, int, int], image_width: int) -> str:
        center_x = bbox[0] + bbox[2] / 2.0
        band = self.config.obstacle_center_band_ratio * image_width
        left = (image_width - band) / 2.0
        right = left + band
        if center_x < left:
            return 'left'
        if center_x > right:
            return 'right'
        return 'center'


def parse_ai_targets_msg(
    msg: object,
    min_confidence: float = 0.0,
) -> List[Detection]:
    detections: List[Detection] = []
    for target in getattr(msg, 'targets', []):
        label = str(getattr(target, 'type', '') or 'object')
        for roi in getattr(target, 'rois', []):
            confidence = float(getattr(roi, 'confidence', 0.0))
            if confidence < min_confidence:
                continue
            rect = getattr(roi, 'rect', None)
            if rect is None:
                continue
            bbox = (
                int(getattr(rect, 'x_offset', 0)),
                int(getattr(rect, 'y_offset', 0)),
                int(getattr(rect, 'width', 0)),
                int(getattr(rect, 'height', 0)),
            )
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    source='yolo_ai_msgs',
                )
            )
    return detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run local OpenCV vision detection on a saved image.'
    )
    parser.add_argument('--input-file', required=True, help='Image file to inspect.')
    parser.add_argument('--output', help='Optional debug overlay output path.')
    parser.add_argument('--crop-output', help='Optional target-card crop output path.')
    parser.add_argument('--config', help='Optional JSON detector config.')
    parser.add_argument('--json', action='store_true', help='Print JSON result.')
    return parser


def load_detector_config(path: Optional[str]) -> DetectorConfig:
    if not path:
        return DetectorConfig()
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    return DetectorConfig.from_mapping(data)


def run_cli(args: argparse.Namespace) -> int:
    image = cv2.imread(args.input_file)
    if image is None:
        print(f'INPUT_IMAGE_ERROR: {args.input_file}', flush=True)
        return 2
    detector = VisionDetector(load_detector_config(args.config))
    decision = detector.analyze(image)
    if args.json:
        print(json.dumps(decision.to_dict(), ensure_ascii=False, indent=2), flush=True)
    else:
        print(
            'VISION_DECISION '
            f'obstacle_danger={decision.obstacle_danger} '
            f'obstacle_zone={decision.obstacle_zone} '
            f'obstacle_area_ratio={decision.obstacle_area_ratio:.4f}',
            flush=True,
        )
        for detection in decision.detections or []:
            print(
                f'DETECTION {detection.label} {detection.confidence:.3f} '
                f'{detection.bbox} {detection.source}',
                flush=True,
            )
    if args.output:
        output = detector.draw_debug(image, decision)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, output)
        print(f'SAVED_DEBUG_IMAGE: {args.output}', flush=True)
    if args.crop_output and decision.target_card is not None:
        crop = detector.crop_detection(image, decision.target_card)
        Path(args.crop_output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.crop_output, crop)
        print(f'SAVED_TARGET_CROP: {args.crop_output}', flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_cli(args)


if __name__ == '__main__':
    raise SystemExit(main())
