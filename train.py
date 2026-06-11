"""Training YOLOv8s multi-class (person + PPE) — Tahap 2.

Contoh:
  python train.py --data data.yaml --model yolov8s.pt --epochs 50 --imgsz 640 --seed 42

Augmentasi (Section 2.4): mosaic on + close_mosaic=10, HSV jitter, scale/translate,
hflip on, vflip off.
"""
import argparse
import random

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        scale=0.5, translate=0.1,
        fliplr=0.5, flipud=0.0,
        project="runs", name="yolov8s_ppe",
    )
    # Salin manual runs/yolov8s_ppe/weights/best.pt -> weights/best.pt setelah selesai.
    print("Training selesai. best.pt ada di runs/yolov8s_ppe/weights/best.pt")


if __name__ == "__main__":
    main()
