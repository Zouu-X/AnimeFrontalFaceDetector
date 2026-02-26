"""
基于 ONNX 的 anime 人脸检测 + 关键点推理，运行时仅依赖 onnxruntime / opencv / numpy。

使用前需先导出 .onnx 文件（在完整训练环境中执行一次）。
"""
from __future__ import annotations

import pathlib
from typing import Optional, Union

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# 与导出脚本常量一致
FACE_INPUT_H, FACE_INPUT_W = 608, 608
# YOLOv3 config: mean=[0,0,0], std=[255,255,255], bgr_to_rgb=True
FACE_MEAN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
FACE_STD = np.array([255.0, 255.0, 255.0], dtype=np.float32)

LANDMARK_INPUT_H, LANDMARK_INPUT_W = 256, 256
HEATMAP_H, HEATMAP_W = 64, 64
NUM_KEYPOINTS = 28
# YOLOv3 anchors/strides
YOLOV3_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],
    [(30, 61), (62, 45), (59, 119)],
    [(10, 13), (16, 30), (33, 23)],
]
YOLOV3_STRIDES = [32, 16, 8]
YOLOV3_NUM_CLASSES = 1
# HRNet config: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb
LANDMARK_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
LANDMARK_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

MIN_FACE_SCORE = 0.05
MAX_DET = 100


def _ensure_nchw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return np.ascontiguousarray(img.transpose(2, 0, 1))
    return img


def _resize_keep_ratio(
    img: np.ndarray, target_h: int, target_w: int
) -> tuple[np.ndarray, float, float, int, int, tuple[int, int]]:
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    if new_w <= 0 or new_h <= 0:
        new_w, new_h = target_w, target_h
        scale = 0.0
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    if pad_w > 0 or pad_h > 0:
        resized = cv2.copyMakeBorder(
            resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return resized, scale, scale, new_w, new_h, (pad_w, pad_h)


def _nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    if boxes.size == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _decode_yolov3_outputs(pred_maps: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    解码 YOLOv3 原始输出为 (bboxes, scores)：
    - bboxes: (N, 4) in input scale
    - scores: (N,)
    """
    all_bboxes = []
    all_scores = []
    for level, pred_map in enumerate(pred_maps):
        # pred_map: (B, C, H, W)
        if pred_map.ndim != 4:
            continue
        b, c, h, w = pred_map.shape
        num_attrib = 5 + YOLOV3_NUM_CLASSES
        num_anchors = c // num_attrib
        if num_anchors <= 0:
            continue
        pred = pred_map.transpose(0, 2, 3, 1).reshape(b, h, w, num_anchors, num_attrib)

        # 解析预测
        xy = _sigmoid(pred[..., 0:2])
        wh = pred[..., 2:4]
        obj = _sigmoid(pred[..., 4:5])
        cls = _sigmoid(pred[..., 5:])

        stride = YOLOV3_STRIDES[level]
        anchors = np.array(YOLOV3_ANCHORS[level], dtype=np.float32)  # (na, 2)

        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        center_x = (grid_x * stride + stride * 0.5)[None, :, :, None]
        center_y = (grid_y * stride + stride * 0.5)[None, :, :, None]

        anchor_w = anchors[:, 0].reshape(1, 1, 1, num_anchors)
        anchor_h = anchors[:, 1].reshape(1, 1, 1, num_anchors)

        bx = center_x + (xy[..., 0] - 0.5) * stride
        by = center_y + (xy[..., 1] - 0.5) * stride
        bw = (anchor_w * 0.5) * np.exp(wh[..., 0])
        bh = (anchor_h * 0.5) * np.exp(wh[..., 1])

        x1 = bx - bw
        y1 = by - bh
        x2 = bx + bw
        y2 = by + bh

        boxes = np.stack([x1, y1, x2, y2], axis=-1)  # (B, H, W, A, 4)
        scores = (obj * cls).squeeze(-1)  # (B, H, W, A)

        boxes = boxes.reshape(b, -1, 4)
        scores = scores.reshape(b, -1)
        all_bboxes.append(boxes)
        all_scores.append(scores)

    if not all_bboxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    bboxes = np.concatenate(all_bboxes, axis=1)[0]
    scores = np.concatenate(all_scores, axis=1)[0]
    return bboxes.astype(np.float32), scores.astype(np.float32)


def _decode_heatmaps_to_keypoints(heatmaps: np.ndarray) -> np.ndarray:
    # heatmaps: (28, 64, 64) or (1, 28, 64, 64)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    nkpt = heatmaps.shape[0]
    out = np.zeros((nkpt, 3), dtype=np.float32)
    for k in range(nkpt):
        hm = heatmaps[k]
        flat = hm.ravel()
        idx = np.argmax(flat)
        score = float(flat[idx])
        y = idx // hm.shape[1]
        x = idx % hm.shape[1]
        out[k, 0] = float(x)
        out[k, 1] = float(y)
        out[k, 2] = score
    return out


class LandmarkDetectorONNX:
    """
    仅依赖 ONNX 的 LandmarkDetector 替代实现。
    接口与 detector.LandmarkDetector 的 __call__ 兼容；predict_with_embedding 暂不支持（可返回空 embedding）。
    """

    def __init__(
        self,
        face_onnx_path: Union[str, pathlib.Path],
        landmark_onnx_path: Union[str, pathlib.Path],
        device: str = "cpu",
        box_scale_factor: float = 1.1,
    ):
        if ort is None:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime 或 onnxruntime-gpu")
        face_onnx_path = pathlib.Path(face_onnx_path)
        landmark_onnx_path = pathlib.Path(landmark_onnx_path)
        if not face_onnx_path.exists():
            raise FileNotFoundError(f"Face ONNX 不存在: {face_onnx_path}")
        if not landmark_onnx_path.exists():
            raise FileNotFoundError(f"Landmark ONNX 不存在: {landmark_onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device.lower() else ["CPUExecutionProvider"]
        self.face_sess = ort.InferenceSession(
            str(face_onnx_path), providers=providers
        )
        self.landmark_sess = ort.InferenceSession(
            str(landmark_onnx_path), providers=providers
        )
        if not self.face_sess.get_inputs():
            raise RuntimeError(f"Face ONNX 输入为空，可能导出失败或文件损坏: {face_onnx_path}")
        if not self.landmark_sess.get_inputs():
            raise RuntimeError(f"Landmark ONNX 输入为空，可能导出失败或文件损坏: {landmark_onnx_path}")
        self.face_input_name = self.face_sess.get_inputs()[0].name
        self.landmark_input_name = self.landmark_sess.get_inputs()[0].name
        self.box_scale_factor = box_scale_factor
        self.device = device

    @staticmethod
    def _load_image(image_or_path: Union[np.ndarray, str, pathlib.Path]) -> np.ndarray:
        if isinstance(image_or_path, np.ndarray):
            return image_or_path
        if isinstance(image_or_path, pathlib.Path):
            image_or_path = image_or_path.as_posix()
        img = cv2.imread(image_or_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_or_path}")
        return img

    def _preprocess_face(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        resized, scale_x, scale_y, _nw, _nh, _pad = _resize_keep_ratio(
            img, FACE_INPUT_H, FACE_INPUT_W
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) - FACE_MEAN) / FACE_STD
        x = _ensure_nchw(x)
        x = x[np.newaxis, ...]
        return x.astype(np.float32), scale_x, scale_y

    def _detect_faces(self, image: np.ndarray) -> list[np.ndarray]:
        inp, scale_x, scale_y = self._preprocess_face(image)
        outputs = self.face_sess.run(None, {self.face_input_name: inp})
        if len(outputs) > 1:
            bboxes, scores = _decode_yolov3_outputs(outputs)
            valid = (scores >= MIN_FACE_SCORE) & (bboxes[:, 0] < bboxes[:, 2]) & (bboxes[:, 1] < bboxes[:, 3])
            bboxes = bboxes[valid]
            scores = scores[valid]
            if bboxes.size == 0:
                return []
            bboxes[:, 0] /= scale_x
            bboxes[:, 1] /= scale_y
            bboxes[:, 2] /= scale_x
            bboxes[:, 3] /= scale_y
            keep = _nms_boxes(bboxes, scores, iou_threshold=0.45)
            bboxes = bboxes[keep]
            scores = scores[keep]
            boxes = np.concatenate([bboxes, scores[:, None]], axis=1)
            boxes = self._update_pred_box(boxes)
            return [b for b in boxes]

        out = outputs[0]
        out = out[0]
        valid = (out[:, 4] >= MIN_FACE_SCORE) & (out[:, 0] < out[:, 2]) & (out[:, 1] < out[:, 3])
        boxes = out[valid]
        if boxes.size == 0:
            return []
        boxes = boxes[:, :5]
        boxes[:, 0] /= scale_x
        boxes[:, 1] /= scale_y
        boxes[:, 2] /= scale_x
        boxes[:, 3] /= scale_y
        keep = _nms_boxes(boxes[:, :4], boxes[:, 4], iou_threshold=0.45)
        boxes = boxes[keep]
        boxes = self._update_pred_box(boxes)
        return [b for b in boxes]

    def _update_pred_box(self, pred_boxes: np.ndarray) -> list[np.ndarray]:
        boxes = []
        if pred_boxes.ndim == 1:
            pred_boxes = pred_boxes.reshape(1, -1)
        for pred_box in pred_boxes:
            box = pred_box[:4].copy()
            size = box[2:] - box[:2] + 1
            new_size = size * self.box_scale_factor
            center = (box[:2] + box[2:]) / 2
            tl = center - new_size / 2
            br = tl + new_size
            new_box = np.concatenate([tl, br])
            if len(pred_box) > 4:
                new_pred_box = np.concatenate([new_box, pred_box[4:5]])
            else:
                new_pred_box = new_box
            boxes.append(new_pred_box)
        return boxes

    def _preprocess_landmark_crop(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        crop = img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]
        if crop.size == 0:
            crop = np.zeros((LANDMARK_INPUT_H, LANDMARK_INPUT_W, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (LANDMARK_INPUT_W, LANDMARK_INPUT_H), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) - LANDMARK_MEAN) / LANDMARK_STD
        x = _ensure_nchw(x)[np.newaxis, ...].astype(np.float32)
        return x

    def _detect_landmarks(
        self, image: np.ndarray, boxes: list[dict]
    ) -> list[dict]:
        image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        preds = []
        for box_dict in boxes:
            bbox = box_dict["bbox"]
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
            inp = self._preprocess_landmark_crop(image, x0, y0, x1, y1)
            heatmaps = self.landmark_sess.run(None, {self.landmark_input_name: inp})[0]
            kpts = _decode_heatmaps_to_keypoints(heatmaps)
            scale_x = (x1 - x0 + 1) / HEATMAP_W
            scale_y = (y1 - y0 + 1) / HEATMAP_H
            kpts[:, 0] = kpts[:, 0] * scale_x + x0
            kpts[:, 1] = kpts[:, 1] * scale_y + y0
            score = float(bbox[4]) if len(bbox) > 4 else 1.0
            preds.append({
                "bbox": np.array([x0, y0, x1, y1, score], dtype=np.float32),
                "keypoints": kpts,
            })
        return preds

    def __call__(
        self,
        image_or_path: Union[np.ndarray, str, pathlib.Path],
        boxes: Optional[list[np.ndarray]] = None,
    ) -> list[dict]:
        image = self._load_image(image_or_path)
        if boxes is None:
            boxes = self._detect_faces(image)
        else:
            boxes = [np.asarray(b) for b in boxes]
        if not boxes:
            return []
        box_list = [{"bbox": b} for b in boxes]
        return self._detect_landmarks(image, box_list)

    def predict_with_embedding(
        self,
        image_or_path: Union[np.ndarray, str, pathlib.Path],
        boxes: Optional[list[np.ndarray]] = None,
        layer: str = "neck",
    ) -> list[dict]:
        result = self.__call__(image_or_path, boxes)
        for r in result:
            r["embedding"] = np.array([], dtype=np.float32)
        return result


def get_onnx_paths(face_name: str = "yolov3") -> tuple[pathlib.Path, pathlib.Path]:
    """返回默认的 face 与 landmark ONNX 路径（包内 onnx_models 目录）。"""
    pkg = pathlib.Path(__file__).parent.resolve()
    onnx_dir = pkg / "onnx_models"
    return (
        onnx_dir / f"face_detector_{face_name}.onnx",
        onnx_dir / "landmark_hrnetv2.onnx",
    )


def create_detector_onnx(
    face_name: str = "yolov3",
    face_onnx_path: Optional[Union[str, pathlib.Path]] = None,
    landmark_onnx_path: Optional[Union[str, pathlib.Path]] = None,
    device: str = "cpu",
    box_scale_factor: float = 1.1,
) -> LandmarkDetectorONNX:
    if face_onnx_path is None or landmark_onnx_path is None:
        fpath, lpath = get_onnx_paths(face_name)
        face_onnx_path = face_onnx_path or fpath
        landmark_onnx_path = landmark_onnx_path or lpath
    return LandmarkDetectorONNX(
        face_onnx_path=face_onnx_path,
        landmark_onnx_path=landmark_onnx_path,
        device=device,
        box_scale_factor=box_scale_factor,
    )
