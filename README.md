# Human Safety Monitoring System (CV) — INaAI 2026

Sistem monitoring keselamatan berbasis kamera: **person detection, tracking
(ByteTrack), PPE detection (helmet/vest), dan people counting**. Satu model
YOLOv8s multi-class → FastAPI + ONNX → deploy ke Render. Eval reproducible.

## Status modul
- [x] T0 Scaffold repo & environment
- [ ] T1 Data (remap 7 kelas, cek leakage)
- [ ] T2 Train YOLOv8s → `best.pt`
- [ ] T3 Eval (`eval_report.json`)
- [ ] T4 Tracking ByteTrack
- [ ] T5 PPE logic + counting
- [ ] T6 FastAPI + ONNX
- [ ] T6.5 Deploy Render + smoke test publik
- [ ] T7 Visualisasi + edge case
- [ ] T8 Technical Report + tag `submission-final`

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate
pip install -r requirements.txt
```

## Data (Tahap 1)
```bash
# 1. Download Construction Site Safety (Roboflow/Kaggle snehilsanyal) ke ./construction-site-safety
# 2. Remap label 10 kelas -> 7 kelas
python remap_labels.py --root construction-site-safety
# 3. Cek tidak ada kebocoran antar split
python check_leakage.py --root construction-site-safety
```

## Training (Tahap 2)
```bash
python train.py --data data.yaml --model yolov8s.pt --epochs 50 --imgsz 640 --seed 42
# salin runs/yolov8s_ppe/weights/best.pt -> weights/best.pt
```

## Eval reproducible (Tahap 3)
```bash
python eval.py --data data.yaml --model weights/best.pt --split test --seed 42
# -> eval_report.json
```

## Run API lokal (Tahap 6)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

## Smoke test
```bash
python smoke_test.py                              # lokal
python smoke_test.py https://YOUR_APP.onrender.com  # publik
```

## Endpoint
| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status model |
| POST | `/detect` | deteksi person |
| POST | `/ppe` | status compliant/violation per orang |
| POST | `/count` | jumlah orang |
| POST | `/track` | tracking video (TODO) |
| POST | `/annotate` | media teranotasi (TODO) |

## Reproduce
Seed=42 di `train.py` & `eval.py`. `eval_report.json` berisi metrik final.

## Atribusi
- Dataset: **Construction Site Safety Image Dataset** — Roboflow Universe Projects (CC BY 4.0).
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- Detection: Ultralytics YOLOv8. Tracking: ByteTrack.
- Lihat `AI_USAGE_LOG.md` untuk penggunaan AI.
