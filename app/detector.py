"""Wrapper deteksi YOLOv8 (mendukung .pt maupun .onnx).

Mengembalikan list dict {bbox, conf, class} untuk satu gambar.
Model di-load SEKALI saat instansiasi (jangan load per-request).
"""
from ultralytics import YOLO


class Detector:
    def __init__(self, weights="weights/best.pt", conf=0.25, iou=0.45):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou

    def detect(self, image):
        """image: numpy HxWx3 (BGR) atau path. -> list deteksi."""
        r = self.model.predict(image, conf=self.conf, iou=self.iou, verbose=False)[0]
        out = []
        for b in r.boxes:
            out.append({
                "bbox": [round(v, 1) for v in b.xyxy[0].tolist()],
                "conf": float(b.conf[0]),
                "class": self.model.names[int(b.cls[0])],
            })
        return out
