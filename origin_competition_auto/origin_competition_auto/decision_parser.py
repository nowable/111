#!/usr/bin/env python3
"""Parse QR task instructions into mission decisions."""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote_plus, urlparse


LEFT_VALUES = {
    'left',
    'l',
    'lt',
    'ccw',
    'counterclockwise',
    'counter-clockwise',
    'anticlockwise',
    'anti-clockwise',
    'zuo',
}
RIGHT_VALUES = {
    'right',
    'r',
    'rt',
    'cw',
    'clockwise',
    'you',
}
LEFT_PHRASES = (
    'left',
    'turn left',
    'loop left',
    'left loop',
    'go left',
    '向左',
    '左转',
    '左圈',
    '左侧',
    '左绕',
    '转左',
    '逆时针',
)
RIGHT_PHRASES = (
    'right',
    'turn right',
    'loop right',
    'right loop',
    'go right',
    '向右',
    '右转',
    '右圈',
    '右侧',
    '右绕',
    '转右',
    '顺时针',
)
PRIORITY_KEYS = (
    'direction',
    'dir',
    'turn',
    'loop',
    'route',
    'side',
    'task',
    'action',
    'cmd',
    'command',
)


@dataclass
class QrInstruction:
    raw: str
    direction: str = ''
    source: str = 'not_found'
    confidence: float = 0.0
    ambiguous: bool = False
    reason: str = ''
    fields: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def parse_direction(text: str) -> str:
    return parse_qr_instruction(text).direction


def parse_qr_instruction(text: str) -> QrInstruction:
    raw = text or ''
    instruction = QrInstruction(raw=raw)
    if not raw.strip():
        instruction.reason = 'empty'
        return instruction

    candidates: List[Tuple[str, str, float, str]] = []
    fields = extract_fields(raw)
    instruction.fields = fields

    for key in PRIORITY_KEYS:
        if key in fields:
            direction = direction_from_value(fields[key])
            if direction:
                candidates.append((direction, f'field:{key}', 0.95, fields[key]))

    for key, value in fields.items():
        if key in PRIORITY_KEYS:
            continue
        direction = direction_from_value(value)
        if direction:
            candidates.append((direction, f'field:{key}', 0.75, value))

    normalized = normalize_text(raw)
    text_direction = direction_from_text(normalized, raw)
    if text_direction:
        candidates.append((text_direction, 'text', 0.55, raw))

    token_direction = direction_from_tokens(normalized)
    if token_direction:
        candidates.append((token_direction, 'token', 0.45, raw))

    if not candidates:
        if has_left_right_conflict(normalized, raw):
            instruction.ambiguous = True
            instruction.reason = 'both_left_and_right'
            instruction.source = 'text_conflict'
            return instruction
        instruction.reason = 'no_direction_match'
        return instruction

    directions = {candidate[0] for candidate in candidates}
    if len(directions) > 1:
        best = sorted(candidates, key=lambda item: item[2], reverse=True)[0]
        same_best = [item for item in candidates if item[2] == best[2]]
        if len({item[0] for item in same_best}) > 1:
            instruction.ambiguous = True
            instruction.reason = 'both_left_and_right'
            instruction.source = ','.join(f'{item[1]}={item[3]}' for item in candidates)
            return instruction
        instruction.direction = best[0]
        instruction.source = best[1]
        instruction.confidence = best[2]
        instruction.reason = 'resolved_by_priority'
        return instruction

    best = sorted(candidates, key=lambda item: item[2], reverse=True)[0]
    instruction.direction = best[0]
    instruction.source = best[1]
    instruction.confidence = best[2]
    instruction.reason = 'matched'
    return instruction


def extract_fields(raw: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    fields.update(fields_from_json(raw))
    fields.update(fields_from_url(raw))
    fields.update(fields_from_pairs(raw))
    return {normalize_key(key): value for key, value in fields.items() if normalize_key(key)}


def fields_from_json(raw: str) -> Dict[str, str]:
    stripped = raw.strip()
    if not stripped.startswith(('{', '[')):
        return {}
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    fields: Dict[str, str] = {}
    flatten_json(data, fields)
    return fields


def flatten_json(value: object, fields: Dict[str, str], prefix: str = '') -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_key = f'{prefix}.{key}' if prefix else str(key)
            flatten_json(item, fields, next_key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            next_key = f'{prefix}.{index}' if prefix else str(index)
            flatten_json(item, fields, next_key)
    else:
        fields[prefix] = str(value)
        short_key = prefix.rsplit('.', 1)[-1]
        fields.setdefault(short_key, str(value))


def fields_from_url(raw: str) -> Dict[str, str]:
    decoded = unquote_plus(raw.strip())
    parsed = urlparse(decoded)
    if not parsed.query:
        return {}
    fields = {}
    for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
        if values:
            fields[key] = values[-1]
    return fields


def fields_from_pairs(raw: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for match in re.finditer(
        r'(?P<key>[A-Za-z_][A-Za-z0-9_.-]{0,32})\s*[:=]\s*(?P<value>[^,;|\s]+)',
        raw,
    ):
        if match.group('key').lower() in ('http', 'https') and '://' in raw:
            continue
        fields[match.group('key')] = match.group('value')
    return fields


def normalize_key(key: str) -> str:
    key = (key or '').strip().lower()
    return key.rsplit('.', 1)[-1]


def normalize_text(text: str) -> str:
    return unquote_plus(text or '').strip().lower()


def direction_from_value(value: str) -> str:
    normalized = normalize_text(value)
    compact = re.sub(r'[\s_\-]+', '', normalized)
    if normalized in LEFT_VALUES or compact in LEFT_VALUES:
        return 'left'
    if normalized in RIGHT_VALUES or compact in RIGHT_VALUES:
        return 'right'
    return direction_from_text(normalized, value)


def direction_from_text(normalized: str, original: str) -> str:
    left = contains_any_phrase(normalized, original, LEFT_PHRASES)
    right = contains_any_phrase(normalized, original, RIGHT_PHRASES)
    if left and not right:
        return 'left'
    if right and not left:
        return 'right'
    return ''


def contains_any_phrase(normalized: str, original: str, phrases: Iterable[str]) -> bool:
    for phrase in phrases:
        if any('\u4e00' <= ch <= '\u9fff' for ch in phrase):
            if phrase in original:
                return True
            continue
        if re.search(rf'\b{re.escape(phrase)}\b', normalized):
            return True
    return False


def direction_from_tokens(normalized: str) -> str:
    tokens = [token for token in re.split(r'[^a-zA-Z0-9]+', normalized) if token]
    left = any(token in LEFT_VALUES for token in tokens)
    right = any(token in RIGHT_VALUES for token in tokens)
    if left and not right:
        return 'left'
    if right and not left:
        return 'right'
    return ''


def has_left_right_conflict(normalized: str, original: str) -> bool:
    text_left = contains_any_phrase(normalized, original, LEFT_PHRASES)
    text_right = contains_any_phrase(normalized, original, RIGHT_PHRASES)
    tokens = [token for token in re.split(r'[^a-zA-Z0-9]+', normalized) if token]
    token_left = any(token in LEFT_VALUES for token in tokens)
    token_right = any(token in RIGHT_VALUES for token in tokens)
    return (text_left or token_left) and (text_right or token_right)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Parse QR instruction text.')
    parser.add_argument('text', nargs='*', help='QR text samples to parse.')
    parser.add_argument('--json', action='store_true')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    samples = args.text or [sys.stdin.read()]
    results = [parse_qr_instruction(sample).to_dict() for sample in samples]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    else:
        for item in results:
            print(
                'QR_PARSE '
                f'direction={item["direction"] or ""} '
                f'source={item["source"]} '
                f'confidence={item["confidence"]:.2f} '
                f'ambiguous={item["ambiguous"]} '
                f'reason={item["reason"]} '
                f'raw={item["raw"]}',
                flush=True,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
