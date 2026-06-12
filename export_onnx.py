"""Export best.pt -> best.onnx — Tahap 6 / Must Have 5.

ONNX Runtime dipakai untuk inference di production (lebih ringan dari torch,
cocok untuk free-tier CPU). Preprocessing tetap ditangani Ultralytics sehingga
output ONNX identik dengan .pt (mitigasi risiko letterbox/normalize mismatch).

Jalankan:
  python export_onnx.py --weights weights/best.pt --imgsz 640
Butuh paket `onnx` (hanya saat export; runtime cukup onnxruntime).
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    model = YOLO(args.weights)
    path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, dynamic=False)
    print(f"ONNX tersimpan: {path}")


if __name__ == "__main__":
    main()
