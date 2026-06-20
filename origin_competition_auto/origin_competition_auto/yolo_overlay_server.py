#!/usr/bin/env python3
"""Serve a browser MJPEG stream of /image with YOLO overlays."""

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    try:
        from ai_msgs.msg import PerceptionTargets
    except ImportError:
        PerceptionTargets = None
except ImportError:
    rclpy = None
    CompressedImage = None
    PerceptionTargets = None

    class Node:  # type: ignore[no-redef]
        pass

try:
    from origin_competition_auto.vision_detector import (
        VisionDecision,
        VisionDetector,
        load_detector_config,
        parse_ai_targets_msg,
        Detection,
    )
    try:
        from origin_competition_auto.yolo_detector import YoloDetector
    except ImportError:
        YoloDetector = None
except ImportError:
    from vision_detector import (  # type: ignore[no-redef]
        VisionDecision,
        VisionDetector,
        load_detector_config,
        parse_ai_targets_msg,
        Detection,
    )
    try:
        from yolo_detector import YoloDetector  # type: ignore[no-redef]
    except ImportError:
        YoloDetector = None


HTML_INDEX = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OriginCar YOLO Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 12px 16px; background: #1b1b1b; position: sticky; top: 0; }
    .meta { font-size: 14px; opacity: 0.85; }
    .wrap { display: flex; justify-content: center; padding: 12px; }
    img { max-width: 100%; height: auto; border: 2px solid #333; border-radius: 8px; }
  </style>
</head>
<body>
  <header>
    <div><strong>OriginCar YOLO Overlay Stream</strong></div>
    <div class="meta" id="meta">loading...</div>
  </header>
  <div class="wrap">
    <img src="/stream.mjpg" alt="YOLO overlay stream">
  </div>
  <script>
    async function refreshStatus() {
      try {
        const resp = await fetch('/status');
        const data = await resp.json();
        document.getElementById('meta').textContent =
          `image_age=${data.image_age_s}s  detections=${data.detection_count}  last_yolo_age=${data.yolo_age_s}s`;
      } catch (err) {
        document.getElementById('meta').textContent = 'status unavailable';
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


class OverlayStreamNode(Node):
    def __init__(
        self,
        image_topic: str,
        yolo_topic: str,
        config_path: Optional[str],
        yolo_min_confidence: float,
        jpeg_quality: int,
    ) -> None:
        super().__init__('yolo_overlay_server')
        self.vision = VisionDetector(load_detector_config(config_path))
        self.yolo_min_confidence = yolo_min_confidence
        self.jpeg_quality = int(max(50, min(95, jpeg_quality)))
        self.local_yolo = None
        self._local_yolo_cache: List[Detection] = []
        self._local_yolo_cache_time = 0.0
        self._local_yolo_min_interval = 0.12
        self.latest_jpeg: Optional[bytes] = None
        self.latest_image_time = 0.0
        self.latest_yolo_time = 0.0
        self.latest_topic_detections: List[Detection] = []
        self._lock = threading.Lock()
        if config_path and YoloDetector is not None:
            try:
                raw = json.loads(Path(config_path).read_text(encoding='utf-8'))
                model_path = str(raw.get('yolo_local_model') or '').strip()
                conf = float(raw.get('yolo_local_conf', yolo_min_confidence))
                self._local_yolo_min_interval = float(raw.get('yolo_local_min_interval', 0.12))
                if model_path:
                    self.local_yolo = YoloDetector(model_path, conf_thres=conf)
                    print(f'[yolo_overlay_server] local YOLO backend={self.local_yolo.backend} model={model_path}', flush=True)
            except Exception as exc:
                print(f'[yolo_overlay_server] local YOLO load failed: {exc}', flush=True)
        self.create_subscription(
            CompressedImage,
            image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        if PerceptionTargets is not None:
            self.create_subscription(
                PerceptionTargets,
                yolo_topic,
                self._yolo_callback,
                qos_profile_sensor_data,
            )

    def _yolo_callback(self, msg: object) -> None:
        detections = parse_ai_targets_msg(msg, min_confidence=self.yolo_min_confidence)
        with self._lock:
            self.latest_topic_detections = detections
            self.latest_yolo_time = time.monotonic()

    def _image_callback(self, msg: CompressedImage) -> None:
        array = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            return
        with self._lock:
            detections = list(self.latest_topic_detections)
        detections.extend(self._local_yolo_detections(image))
        overlay = self.vision.draw_debug(image, VisionDecision(detections=detections))
        ok, encoded = cv2.imencode(
            '.jpg',
            overlay,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        now = time.monotonic()
        with self._lock:
            self.latest_jpeg = encoded.tobytes()
            self.latest_image_time = now
            self.latest_topic_detections = detections

    def _local_yolo_detections(self, image: np.ndarray) -> List[Detection]:
        if self.local_yolo is None:
            return []
        now = time.monotonic()
        if now - self._local_yolo_cache_time < self._local_yolo_min_interval:
            return list(self._local_yolo_cache)
        try:
            raw = self.local_yolo.detect(image)
        except Exception as exc:
            print(f'[yolo_overlay_server] local YOLO inference failed: {exc}', flush=True)
            self.local_yolo = None
            return []
        self._local_yolo_cache = [
            Detection(
                label=det.class_name,
                confidence=det.score,
                bbox=(int(det.x1), int(det.y1), int(det.x2 - det.x1), int(det.y2 - det.y1)),
                source='yolo_local',
            )
            for det in raw
        ]
        self._local_yolo_cache_time = now
        return list(self._local_yolo_cache)

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def get_status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            return {
                'image_age_s': round(max(0.0, now - self.latest_image_time), 2) if self.latest_image_time else None,
                'yolo_age_s': round(max(0.0, now - self.latest_yolo_time), 2) if self.latest_yolo_time else None,
                'detection_count': len(self.latest_topic_detections),
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Serve /image with YOLO overlays as MJPEG.')
    parser.add_argument('--image-topic', default='/image')
    parser.add_argument('--yolo-topic', default='/hobot_dnn_detection')
    parser.add_argument('--config', help='Mission config JSON path.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9902)
    parser.add_argument('--yolo-min-confidence', type=float, default=0.45)
    parser.add_argument('--jpeg-quality', type=int, default=80)
    return parser


def main(argv=None) -> int:
    if rclpy is None:
        print('rclpy is not available', flush=True)
        return 2

    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rclpy.init(args=None)
    node = OverlayStreamNode(
        image_topic=args.image_topic,
        yolo_topic=args.yolo_topic,
        config_path=args.config,
        yolo_min_confidence=args.yolo_min_confidence,
        jpeg_quality=args.jpeg_quality,
    )

    stop_event = threading.Event()

    def spin_loop() -> None:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)

    spin_thread = threading.Thread(target=spin_loop, name='YoloOverlaySpin', daemon=True)
    spin_thread.start()

    app = Flask(__name__)

    @app.route('/')
    def index():
        return HTML_INDEX

    @app.route('/status')
    def status():
        return jsonify(node.get_status())

    @app.route('/snapshot.jpg')
    def snapshot():
        frame = node.get_frame()
        if frame is None:
            return Response(status=503)
        return Response(frame, mimetype='image/jpeg')

    @app.route('/stream.mjpg')
    def stream():
        def generate():
            while not stop_event.is_set():
                frame = node.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    b'Content-Length: ' + str(len(frame)).encode('ascii') + b'\r\n\r\n' +
                    frame + b'\r\n'
                )
                time.sleep(0.05)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
        return 0
    finally:
        stop_event.set()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
