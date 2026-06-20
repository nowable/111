#!/usr/bin/env python3
"""Generate fixed-route config snippets from the provided field map dimensions."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]
Segment = Dict[str, object]


@dataclass(frozen=True)
class MapRoutePreset:
    name: str
    field_width_m: float
    area_a_depth_m: float
    area_b_depth_m: float
    area_c_depth_m: float
    initial_heading_deg: float
    a_to_qr: Sequence[Point]
    qr_to_corridor: Sequence[Point]
    b_corridor: Sequence[Point]
    c_loop_radius_hint_m: float

    @property
    def field_height_m(self) -> float:
        return self.area_a_depth_m + self.area_b_depth_m + self.area_c_depth_m


def default_map_preset() -> MapRoutePreset:
    # Coordinates are in meters. Origin is the lower-left corner of A.
    # x grows to the right, y grows from A toward C.
    return MapRoutePreset(
        name='field_map_v1',
        field_width_m=5.0,
        area_a_depth_m=2.0,
        area_b_depth_m=0.5,
        area_c_depth_m=2.5,
        # Place the car at P with its nose aligned to the first visible dashed
        # route segment: almost rightward, slightly upward on the map.
        initial_heading_deg=3.1,
        # Pixel-sampled from the provided map so the route overlays the lower
        # black dashed line from P to QR as closely as the screenshot permits.
        a_to_qr=[
            (0.284, 0.246),
            (0.653, 0.266),
            (1.249, 0.303),
            (1.788, 0.392),
            (2.327, 0.507),
            (2.866, 0.669),
            (3.360, 0.857),
            (3.802, 1.082),
            (4.171, 1.290),
            (4.427, 1.520),
            (4.625, 1.708),
            (4.824, 1.813),
        ],
        # After QR, move back toward the center gate into the B yellow passage.
        qr_to_corridor=[
            (4.62, 1.70),
            (3.75, 1.88),
            (2.70, 1.96),
            (2.50, 2.12),
        ],
        # B is the 0.5 m strip between A and C; keep a little extra travel margin.
        b_corridor=[
            (2.50, 2.12),
            (2.50, 2.70),
        ],
        c_loop_radius_hint_m=0.70,
    )


def normalize_angle_deg(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def heading_deg(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def round_float(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def route_from_waypoints(
    waypoints: Sequence[Point],
    start_heading_deg: float,
    *,
    final_as_seek_qr: bool = False,
    min_turn_deg: float = 3.0,
) -> Tuple[List[Segment], float]:
    if len(waypoints) < 2:
        raise ValueError('at least two waypoints are required')

    segments: List[Segment] = []
    current_heading = start_heading_deg
    legs = list(zip(waypoints[:-1], waypoints[1:]))
    for index, (start, end) in enumerate(legs):
        target_heading = heading_deg(start, end)
        turn = normalize_angle_deg(target_heading - current_heading)
        if abs(turn) >= min_turn_deg:
            segments.append({'type': 'turn', 'angle_deg': round_float(turn, 1)})

        leg_distance = distance(start, end)
        if final_as_seek_qr and index == len(legs) - 1:
            segments.append({'type': 'seek_qr', 'max_distance': round_float(leg_distance)})
        else:
            segments.append({'type': 'drive', 'distance': round_float(leg_distance)})
        current_heading = target_heading

    return segments, current_heading


def build_fixed_routes(preset: MapRoutePreset) -> Dict[str, List[Segment]]:
    a_to_qr, heading_after_a = route_from_waypoints(
        preset.a_to_qr,
        preset.initial_heading_deg,
        final_as_seek_qr=True,
    )
    qr_to_corridor, heading_after_qr = route_from_waypoints(
        preset.qr_to_corridor,
        heading_after_a,
    )
    b_corridor, _heading_after_b = route_from_waypoints(
        preset.b_corridor,
        heading_after_qr,
    )
    return {
        'A_TO_QR': a_to_qr,
        'QR_TO_CORRIDOR': qr_to_corridor,
        'B_CORRIDOR': b_corridor,
        'C_LOOP_RIGHT': [
            {'type': 'arc', 'angle_deg': -330, 'radius_hint': preset.c_loop_radius_hint_m},
        ],
        'C_LOOP_LEFT': [
            {'type': 'arc', 'angle_deg': 330, 'radius_hint': preset.c_loop_radius_hint_m},
        ],
    }


def build_config_patch(preset: MapRoutePreset) -> Dict[str, object]:
    return {
        'route_mode': 'fixed2d',
        'fixed_route_enabled': True,
        'fixed_route_linear': 0.03,
        'fixed_route_turn_angular': 0.25,
        'fixed_route_heading_tol': 0.10,
        'fixed_route_dist_tol': 0.05,
        'fixed_route_segment_timeout': 12.0,
        'fixed_routes': build_fixed_routes(preset),
    }


def update_config(path: Path, patch: Dict[str, object]) -> None:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'{path} does not contain a JSON object')
    backup = path.with_suffix(path.suffix + '.bak')
    backup.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    data.update(patch)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'updated_config={path}', file=sys.stderr)
    print(f'backup={backup}', file=sys.stderr)


def svg_polyline(points: Iterable[Point], preset: MapRoutePreset, scale: float) -> str:
    coords = []
    for x, y in points:
        sx = x * scale
        sy = (preset.field_height_m - y) * scale
        coords.append(f'{sx:.1f},{sy:.1f}')
    return ' '.join(coords)


def write_svg(path: Path, preset: MapRoutePreset) -> None:
    scale = 140.0
    width = preset.field_width_m * scale
    height = preset.field_height_m * scale
    a_h = preset.area_a_depth_m * scale
    b_h = preset.area_b_depth_m * scale
    c_h = preset.area_c_depth_m * scale
    b_y = c_h
    a_y = c_h + b_h
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">
  <rect x="0" y="0" width="{width:.0f}" height="{c_h:.0f}" fill="#bfe4b3"/>
  <rect x="0" y="{b_y:.0f}" width="{width:.0f}" height="{b_h:.0f}" fill="#f5e58b"/>
  <rect x="0" y="{a_y:.0f}" width="{width:.0f}" height="{a_h:.0f}" fill="#b8cdea"/>
  <rect x="{1.0 * scale:.0f}" y="{0.55 * scale:.0f}" width="{3.0 * scale:.0f}" height="{0.8 * scale:.0f}" rx="18" fill="#bfe4b3" opacity="0.85"/>
  <path d="M {0.35 * scale:.0f} {1.10 * scale:.0f}
           H {4.65 * scale:.0f}
           V {2.20 * scale:.0f}
           H {0.35 * scale:.0f}
           Z" fill="none" stroke="#f3d86c" stroke-width="{0.45 * scale:.0f}" stroke-linejoin="round" opacity="0.85"/>
  <line x1="{2.5 * scale:.0f}" y1="{b_y:.0f}" x2="{2.5 * scale:.0f}" y2="{a_y:.0f}" stroke="#f3d86c" stroke-width="{0.8 * scale:.0f}" opacity="0.9"/>
  <polyline points="{svg_polyline(preset.a_to_qr, preset, scale)}" fill="none" stroke="#111" stroke-width="5" stroke-dasharray="14 10"/>
  <polyline points="{svg_polyline(preset.qr_to_corridor, preset, scale)}" fill="none" stroke="#d2691e" stroke-width="4" stroke-dasharray="8 8"/>
  <polyline points="{svg_polyline(preset.b_corridor, preset, scale)}" fill="none" stroke="#d2691e" stroke-width="4"/>
  <circle cx="{preset.a_to_qr[0][0] * scale:.1f}" cy="{(preset.field_height_m - preset.a_to_qr[0][1]) * scale:.1f}" r="12" fill="#0b7cc1"/>
  <text x="{preset.a_to_qr[0][0] * scale + 18:.1f}" y="{(preset.field_height_m - preset.a_to_qr[0][1]) * scale + 6:.1f}" font-size="22">P</text>
  <rect x="{preset.a_to_qr[-1][0] * scale - 18:.1f}" y="{(preset.field_height_m - preset.a_to_qr[-1][1]) * scale - 18:.1f}" width="36" height="36" fill="#ffec3d" stroke="#1e40ff" stroke-width="3" transform="rotate(45 {preset.a_to_qr[-1][0] * scale:.1f} {(preset.field_height_m - preset.a_to_qr[-1][1]) * scale:.1f})"/>
  <text x="10" y="26" font-size="20">Generated fixed2d route preview (meters)</text>
  <text x="10" y="{height - 18:.0f}" font-size="18">A_TO_QR black, QR_TO_CORRIDOR orange dashed, B_CORRIDOR orange</text>
</svg>
'''
    path.write_text(svg, encoding='utf-8')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate fixed2d route JSON from the competition field map.',
    )
    parser.add_argument('--output', type=Path, help='write JSON patch to this file')
    parser.add_argument('--svg', type=Path, help='write an SVG route preview')
    parser.add_argument(
        '--update-config',
        type=Path,
        help='merge generated route keys into a mission_defaults.json file',
    )
    parser.add_argument(
        '--initial-heading-deg',
        type=float,
        default=None,
        help='override P start heading; 0 means facing right on the map',
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    preset = default_map_preset()
    if args.initial_heading_deg is not None:
        preset = MapRoutePreset(
            name=preset.name,
            field_width_m=preset.field_width_m,
            area_a_depth_m=preset.area_a_depth_m,
            area_b_depth_m=preset.area_b_depth_m,
            area_c_depth_m=preset.area_c_depth_m,
            initial_heading_deg=args.initial_heading_deg,
            a_to_qr=preset.a_to_qr,
            qr_to_corridor=preset.qr_to_corridor,
            b_corridor=preset.b_corridor,
            c_loop_radius_hint_m=preset.c_loop_radius_hint_m,
        )
    patch = build_config_patch(preset)
    text = json.dumps(patch, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')
    if args.svg:
        write_svg(args.svg, preset)
        print(f'svg={args.svg}', file=sys.stderr)
    if args.update_config:
        update_config(args.update_config, patch)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
