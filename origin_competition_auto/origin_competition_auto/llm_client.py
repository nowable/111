#!/usr/bin/env python3
"""Image LLM client used by the OriginCar mission pipeline."""

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LlmConfig:
    mode: str = 'placeholder'
    api_url: str = 'https://api.openai.com/v1/chat/completions'
    model: str = 'gpt-4.1-mini'
    api_key_env: str = 'OPENAI_API_KEY'
    timeout: float = 20.0
    prompt: str = (
        'Identify the main object or symbol in this competition image. '
        'Answer briefly in Chinese, and include uncertainty if the image is unclear.'
    )
    max_tokens: int = 160
    temperature: float = 0.0
    max_image_bytes: int = 3_000_000
    max_image_dim: int = 1024
    jpeg_quality: int = 85
    downscale_enabled: bool = True

    @classmethod
    def from_mapping(cls, data: Dict[str, object]) -> 'LlmConfig':
        values = {}
        for key in cls.__dataclass_fields__:
            if key in data:
                values[key] = data[key]
        return cls(**values)


class LlmClient:
    """Small OpenAI-compatible vision client with safe offline fallbacks."""

    def __init__(self, config: Optional[LlmConfig] = None) -> None:
        self.config = config or LlmConfig()

    def analyze(self, image_path: Optional[str]) -> str:
        mode = (self.config.mode or 'placeholder').lower()
        if mode in ('disabled', 'off', 'none'):
            return 'LLM_DISABLED'
        if mode in ('placeholder', 'mock', 'dry-run'):
            if image_path:
                return f'LLM_PLACEHOLDER image={image_path}'
            return 'LLM_PLACEHOLDER no_image'
        if mode in ('openai-compatible', 'openai', 'http'):
            return self._analyze_openai_compatible(image_path)
        return f'LLM_ERROR unsupported_mode={self.config.mode}'

    def _analyze_openai_compatible(self, image_path: Optional[str]) -> str:
        if not image_path:
            return 'LLM_ERROR no_image'
        path = Path(image_path)
        if not path.exists():
            return f'LLM_ERROR image_not_found={path}'
        api_key = os.environ.get(self.config.api_key_env, '')
        if not api_key:
            return f'LLM_ERROR missing_env={self.config.api_key_env}'
        try:
            data_url = self._image_data_url(path)
            payload = self._request_payload(data_url)
            return self._post_json(payload, api_key)
        except Exception as exc:
            return f'LLM_ERROR {type(exc).__name__}: {exc}'

    def _image_data_url(self, path: Path) -> str:
        data = path.read_bytes()
        mime = 'image/jpeg'
        if self.config.downscale_enabled:
            shrunk = self._downscale_jpeg(data)
            if shrunk is not None:
                data = shrunk
        else:
            mime = mimetypes.guess_type(str(path))[0] or 'image/jpeg'
        if len(data) > self.config.max_image_bytes:
            raise ValueError(
                f'image too large: {len(data)} bytes > {self.config.max_image_bytes}'
            )
        encoded = base64.b64encode(data).decode('ascii')
        return f'data:{mime};base64,{encoded}'

    def _downscale_jpeg(self, data: bytes) -> Optional[bytes]:
        """Resize so max(h, w) <= max_image_dim and re-encode JPEG.

        Returns None if cv2/numpy is unavailable so the caller falls back to the
        original bytes (the network call still works, just larger).
        """
        try:
            import cv2
            import numpy as np
        except Exception:
            return None
        try:
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            h, w = arr.shape[:2]
            longest = max(h, w)
            if longest > self.config.max_image_dim:
                scale = self.config.max_image_dim / float(longest)
                arr = cv2.resize(
                    arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(
                '.jpg', arr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)])
            if not ok:
                return None
            return buf.tobytes()
        except Exception:
            return None

    def _request_payload(self, data_url: str) -> Dict[str, object]:
        return {
            'model': self.config.model,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': self.config.prompt},
                        {'type': 'image_url', 'image_url': {'url': data_url}},
                    ],
                }
            ],
            'max_tokens': int(self.config.max_tokens),
            'temperature': float(self.config.temperature),
        }

    def _post_json(self, payload: Dict[str, object], api_key: str) -> str:
        body = json.dumps(payload).encode('utf-8')
        request = urllib.request.Request(
            self.config.api_url,
            data=body,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                response_body = response.read().decode('utf-8')
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:500]
            raise RuntimeError(f'HTTP {exc.code}: {detail}') from exc
        data = json.loads(response_body)
        return self._extract_text(data)

    def _extract_text(self, data: Dict[str, object]) -> str:
        choices = data.get('choices')
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get('message')
                if isinstance(message, dict):
                    content = message.get('content')
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get('text'), str):
                                parts.append(item['text'])
                        if parts:
                            return ''.join(parts).strip()
        output_text = data.get('output_text')
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        return f'LLM_ERROR unexpected_response={json.dumps(data)[:500]}'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Debug the image LLM client.')
    parser.add_argument('--image', help='Image path to analyze.')
    parser.add_argument('--mode', default='placeholder',
                        choices=['placeholder', 'disabled', 'openai-compatible'])
    parser.add_argument('--api-url', default=LlmConfig.api_url)
    parser.add_argument('--model', default=LlmConfig.model)
    parser.add_argument('--api-key-env', default=LlmConfig.api_key_env)
    parser.add_argument('--timeout', type=float, default=LlmConfig.timeout)
    parser.add_argument('--prompt', default=LlmConfig.prompt)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    config = LlmConfig(
        mode=args.mode,
        api_url=args.api_url,
        model=args.model,
        api_key_env=args.api_key_env,
        timeout=args.timeout,
        prompt=args.prompt,
    )
    result = LlmClient(config).analyze(args.image)
    print(f'LLM_RESULT: {result}', flush=True)
    return 0 if not result.startswith('LLM_ERROR') else 2


if __name__ == '__main__':
    raise SystemExit(main())
