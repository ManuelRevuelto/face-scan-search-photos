"""Detección y reconocimiento facial con los modelos ONNX del pack buffalo_l de InsightFace.

Se usan los modelos directamente con onnxruntime (sin el paquete `insightface`, que en Windows
requiere compilar C++):
  - det_10g.onnx   -> detector SCRFD (cajas + 5 puntos faciales)
  - w600k_r50.onnx -> ArcFace ResNet50, embedding de 512 dimensiones
"""
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # HEIC opcional
    pass

log = logging.getLogger(__name__)

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DET_MODEL = "det_10g.onnx"
REC_MODEL = "w600k_r50.onnx"

# Posición canónica de ojos, nariz y comisuras en una cara alineada de 112x112 (ArcFace)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class DetectedFace:
    bbox: tuple[float, float, float, float]  # relativa a la imagen (0-1)
    det_score: float
    embedding: np.ndarray  # float32[512], norma 1

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


def load_image(data: bytes, max_side: int) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)  # respeta la orientación de la cámara
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def ensure_models(model_dir: Path) -> Path:
    if (model_dir / DET_MODEL).exists() and (model_dir / REC_MODEL).exists():
        return model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir / "buffalo_l.zip"
    log.info("Descargando modelos (~280 MB) de %s", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            name = Path(member).name
            if name in (DET_MODEL, REC_MODEL):
                (model_dir / name).write_bytes(z.read(member))
    zip_path.unlink()
    return model_dir


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thresh]
    return keep


def _session(path: Path, providers):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # ArcFace declara batch=1 pero admite varias caras: evita el aviso
    return ort.InferenceSession(str(path), sess_options=opts, providers=providers)


class _Detector:
    """SCRFD: 3 niveles (strides 8/16/32), 2 anchors por punto, salidas score/bbox/kps."""

    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, path: Path, providers, size: int):
        self.session = _session(path, providers)
        self.input_name = self.session.get_inputs()[0].name
        self.size = size
        self._centers = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            self._centers[key] = np.repeat(c, self.NUM_ANCHORS, axis=0)
        return self._centers[key]

    def detect(self, bgr: np.ndarray, score_thresh: float, nms_thresh: float = 0.4):
        h, w = bgr.shape[:2]
        scale = self.size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (self.size, self.size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})

        n = len(self.STRIDES)
        all_scores, all_boxes, all_kps = [], [], []
        for i, stride in enumerate(self.STRIDES):
            scores = outs[i].reshape(-1)
            boxes = outs[i + n].reshape(-1, 4) * stride
            kps = outs[i + 2 * n].reshape(-1, 10) * stride
            centers = self._anchor_centers(self.size // stride, self.size // stride, stride)
            keep = np.nonzero(scores >= score_thresh)[0]
            if not keep.size:
                continue
            c = centers[keep]
            b = boxes[keep]
            all_boxes.append(np.hstack([c - b[:, :2], c + b[:, 2:]]))
            all_kps.append((np.tile(c, 5) + kps[keep]).reshape(-1, 5, 2))
            all_scores.append(scores[keep])
        if not all_scores:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)

        boxes = np.vstack(all_boxes) / scale
        kps = np.vstack(all_kps) / scale
        dets = np.hstack([boxes, np.concatenate(all_scores)[:, None]]).astype(np.float32)
        keep = _nms(dets, nms_thresh)
        return dets[keep], kps[keep]


class _Recognizer:
    def __init__(self, path: Path, providers):
        self.session = _session(path, providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, bgr: np.ndarray, kps_list: np.ndarray) -> np.ndarray:
        crops = []
        for kps in kps_list:
            m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
            crops.append(cv2.warpAffine(bgr, m, (112, 112), borderValue=0.0))
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


class FaceEngine:
    def __init__(self, model_dir: Path, det_size: int = 640, providers: list[str] | None = None,
                 max_side: int = 1600, min_det_score: float = 0.5):
        providers = providers or ["CPUExecutionProvider"]
        ensure_models(model_dir)
        self.max_side = max_side
        self.min_det_score = min_det_score
        self.detector = _Detector(model_dir / DET_MODEL, providers, det_size)
        self.recognizer = _Recognizer(model_dir / REC_MODEL, providers)

    def detect(self, img: Image.Image) -> list[DetectedFace]:
        bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        w, h = img.size
        dets, kps = self.detector.detect(bgr, self.min_det_score)
        if not len(dets):
            return []
        embeddings = self.recognizer.embed(bgr, kps)
        faces = []
        for (x1, y1, x2, y2, score), emb in zip(dets, embeddings):
            bbox = (max(x1, 0) / w, max(y1, 0) / h, min(x2, w) / w, min(y2, h) / h)
            faces.append(DetectedFace(tuple(float(v) for v in bbox), float(score), emb.astype(np.float32)))
        return faces

    def detect_bytes(self, data: bytes) -> tuple[Image.Image, list[DetectedFace]]:
        img = load_image(data, self.max_side)
        return img, self.detect(img)

    def reference_embedding(self, data: bytes) -> np.ndarray:
        """Embedding de la cara principal (la más grande) de una foto de referencia."""
        try:
            img = load_image(data, self.max_side)
        except OSError as e:
            raise ValueError(f"No se puede leer la imagen de referencia: {e}") from e
        faces = self.detect(img)
        if not faces:
            raise ValueError("No se ha detectado ninguna cara en la foto de referencia")
        return max(faces, key=lambda f: f.area).embedding
