"""Eval reproducible (seed tetap) — Tahap 3 / Must Have 4.

Contoh:
  python eval.py --data data.yaml --model weights/best.pt --split test --seed 42

Output: cetak tabel metrik + tulis eval_report.json.
"""
import argparse
import json
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
    ap.add_argument("--model", default="weights/best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    m = YOLO(args.model)
    r = m.val(data=args.data, split=args.split, iou=0.5, verbose=False)

    report = {
        "mAP@0.5": round(float(r.box.map50), 4),
        "mAP@0.5:0.95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class_mAP@0.5": {
            m.names[i]: round(float(v), 4) for i, v in enumerate(r.box.maps)
        },
        "seed": args.seed,
    }
    print(json.dumps(report, indent=2))
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'METRIC':<16}{'VALUE':>10}")
    for k in ["mAP@0.5", "mAP@0.5:0.95", "precision", "recall"]:
        print(f"{k:<16}{report[k]:>10.4f}")


if __name__ == "__main__":
    main()
