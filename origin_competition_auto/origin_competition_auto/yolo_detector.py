#!/usr/bin/env python3
"""YOLO11n detector for the competition, dual backend.

Backends (chosen by model file extension):
- ``.onnx``: onnxruntime, used for local development and tests.
- ``.bin``: RDK X5 BPU via hobot_dnn (pyeasy_dnn), used on the board.

Both backends produce the rdk_model_zoo 6-tensor protocol
(cls/DFL-box per stride 8/16/32, NHWC), decoded here with DFL + NMS.
"""

import math
from typing import List, NamedTuple, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always present on board
    cv2 = None

try:
    import onnxruntime as _ort
except ImportError:
    _ort = None

try:
    from hobot_dnn import pyeasy_dnn as _dnn
except ImportError:
    _dnn = None

CLASS_NAMES = ["black_obstacle", "qr_board", "image_card", "base", "marker"]
STRIDES = (8, 16, 32)
INPUT_SIZE = 640
REG_MAX = 16


class Detection(NamedTuple):
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def bgr_to_nv12(bgr: np.ndarray) -> np.ndarray:
    yuv420p = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    h, w = bgr.shape[:2]
    y = yuv420p[:h, :]
    uv = yuv420p[h:, :].reshape(2, -1)
    nv12 = np.concatenate([y.reshape(-1), uv.transpose(1, 0).reshape(-1)])
    return np.ascontiguousarray(nv12)


def decode_outputs(
    outputs: Sequence[np.ndarray],
    conf_thres: float,
    num_classes: int,
) -> List[np.ndarray]:
    """Decode 6 NHWC tensors -> (boxes_xyxy, scores, class_ids) in 640-space.

    outputs order: [cls_s8, box_s8, cls_s16, box_s16, cls_s32, box_s32];
    cls tensors hold raw logits, box tensors hold DFL logits (H, W, 64).
    """
    logit_thres = -math.log(1.0 / max(conf_thres, 1e-6) - 1.0)
    boxes, scores, class_ids = [], [], []
    for level, stride in enumerate(STRIDES):
        cls = outputs[level * 2].reshape(-1, num_classes)
        box = outputs[level * 2 + 1]
        grid_h, grid_w = box.shape[1], box.shape[2]
        box = box.reshape(-1, 4, REG_MAX)
        best = cls.max(axis=1)
        keep = np.flatnonzero(best >= logit_thres)
        if keep.size == 0:
            continue
        cls_keep = cls[keep]
        dfl = box[keep].astype(np.float32)
        dfl = dfl - dfl.max(axis=2, keepdims=True)
        exp = np.exp(dfl)
        ltrb = (exp / exp.sum(axis=2, keepdims=True)) @ np.arange(REG_MAX, dtype=np.float32)
        cy, cx = np.divmod(keep, grid_w)
        cx = (cx.astype(np.float32) + 0.5) * stride
        cy = (cy.astype(np.float32) + 0.5) * stride
        boxes.append(np.stack([
            cx - ltrb[:, 0] * stride, cy - ltrb[:, 1] * stride,
            cx + ltrb[:, 2] * stride, cy + ltrb[:, 3] * stride,
        ], axis=1))
        scores.append(1.0 / (1.0 + np.exp(-cls_keep.max(axis=1))))
        class_ids.append(cls_keep.argmax(axis=1))
    if not boxes:
        return [np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)]
    return [np.concatenate(boxes), np.concatenate(scores), np.concatenate(class_ids)]


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.35,
        iou_thres: float = 0.5,
        class_names: Sequence[str] = CLASS_NAMES,
    ) -> None:
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.class_names = list(class_names)
        self._session = None
        self._bpu_models = None
        if model_path.endswith('.onnx'):
            if _ort is None:
                raise RuntimeError('onnxruntime is required for .onnx models')
            self._session = _ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
            self.backend = 'onnx'
        elif model_path.endswith('.bin'):
            if _dnn is None:
                raise RuntimeError('hobot_dnn is required for .bin models')
            self._bpu_models = _dnn.load(model_path)
            self.backend = 'bpu'
        else:
            raise ValueError(f'Unsupported model extension: {model_path}')

    def _infer_onnx(self, bgr: np.ndarray) -> List[np.ndarray]:
        resized = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        name = self._session.get_inputs()[0].name
        return self._session.run(None, {name: x})

    def _infer_bpu(self, bgr: np.ndarray) -> List[np.ndarray]:
        resized = cv2.resize(bgr, (INPUT_SIZE, INPUT_SIZE))
        outputs = self._bpu_models[0].forward(bgr_to_nv12(resized))
        return [np.asarray(o.buffer) for o in outputs]

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """Run detection on a BGR frame; boxes returned in frame coordinates."""
        h, w = bgr.shape[:2]
        raw = self._infer_onnx(bgr) if self.backend == 'onnx' else self._infer_bpu(bgr)
        boxes, scores, class_ids = decode_outputs(
            raw, self.conf_thres, len(self.class_names))
        if boxes.shape[0] == 0:
            return []
        xywh = boxes.copy()
        xywh[:, 2] -= xywh[:, 0]
        xywh[:, 3] -= xywh[:, 1]
        keep = cv2.dnn.NMSBoxes(
            xywh.tolist(), scores.tolist(), self.conf_thres, self.iou_thres)
        results: List[Detection] = []
        sx = w / float(INPUT_SIZE)
        sy = h / float(INPUT_SIZE)
        for i in np.asarray(keep).reshape(-1):
            cid = int(class_ids[i])
            results.append(Detection(
                class_id=cid,
                class_name=self.class_names[cid] if cid < len(self.class_names) else str(cid),
                score=float(scores[i]),
                x1=float(np.clip(boxes[i, 0] * sx, 0, w)),
                y1=float(np.clip(boxes[i, 1] * sy, 0, h)),
                x2=float(np.clip(boxes[i, 2] * sx, 0, w)),
                y2=float(np.clip(boxes[i, 3] * sy, 0, h)),
            ))
        results.sort(key=lambda d: d.score, reverse=True)
        return results


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description='YOLO detector debug CLI')
    parser.add_argument('--model', required=True, help='*.onnx (local) or *.bin (board)')
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--save', default='', help='optional annotated output image path')
    args = parser.parse_args()

    detector = YoloDetector(args.model, conf_thres=args.conf)
    image = cv2.imread(args.input_file)
    start = time.monotonic()
    detections = detector.detect(image)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    print(f'YOLO_BACKEND: {detector.backend} time={elapsed_ms:.1f}ms')
    for det in detections:
        print(f'YOLO_DET: {det.class_name} {det.score:.3f} '
              f'({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})')
    if args.save:
        for det in detections:
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            cv2.putText(image, f'{det.class_name} {det.score:.2f}',
                        (p1[0], max(12, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imwrite(args.save, image)
        print(f'YOLO_SAVED: {args.save}')


if __name__ == '__main__':
    main()
