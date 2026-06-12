# Human Safety Monitoring System (CV) — INaAI 2026

Sistem monitoring keselamatan berbasis kamera: **person detection, tracking
(ByteTrack), PPE detection (helmet/vest), dan people counting**. Satu model
YOLOv8s multi-class → FastAPI + ONNX → deploy ke Railway. Eval reproducible.

## Status modul
- [x] T0 Scaffold repo & environment
- [x] T1 Data (remap 7 kelas, cek leakage 2-lapis MD5+scene)
- [x] T2 Train YOLOv8s → `best.pt` (Colab T4, mAP@0.5 0.78)
- [x] T3 Eval (`eval_report.json`, seed=42 reproducible)
- [x] T4 Tracking ByteTrack
- [x] T5 PPE logic + counting (`ppe_demo.py` 8/8)
- [x] T6 FastAPI + ONNX (smoke PASS, p50 <120ms)
- [~] T6.5 Deploy Railway — config siap (`railway.json`, `Dockerfile`, lihat `DEPLOY.md`); tinggal push & connect
- [x] T7 Visualisasi + edge case (`/annotate` label+conf, `edge_case_test.py`, `report/EDGE_CASES.md`)
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

## PPE logic + counting (Tahap 5)
```bash
python ppe_demo.py                       # 8 unit test logika PPE (8/8 PASS)
python ppe_demo.py --images a.jpg b.jpg  # + uji gambar nyata
```

## Export ONNX & run API (Tahap 6)
```bash
# 1. export best.pt -> best.onnx (butuh paket onnx, sekali saja)
python export_onnx.py --weights weights/best.pt --imgsz 640
# 2. jalankan API (image-endpoint pakai ONNX Runtime, tracking pakai .pt)
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Web UI dashboard : http://localhost:8000/        (upload & lihat hasil teranotasi)
# Swagger API docs : http://localhost:8000/docs
```

## Smoke test & benchmark
```bash
python smoke_test.py                                # lokal (semua endpoint + error-case)
python smoke_test.py https://YOUR_APP.up.railway.app  # publik
python benchmark.py --n 30                           # p50/p95 latency /detect & /ppe
```
Latency lokal (CPU): **/detect p50 114 ms, /ppe p50 116 ms** (<500 ms target);
**/track ~9,6 dtk klip → 8,8 dtk** (vid_stride=2, imgsz=480).

## Dashboard Streamlit (UI alternatif)
```bash
streamlit run streamlit_app.py   # http://localhost:8501
```
Mode: PPE Compliance, Deteksi, Hitung orang, Robustness Test, Tracking video.
Memakai modul `app/` langsung (tak butuh server API berjalan).

## Deploy publik (Railway) — Tahap 6.5
Konfigurasi siap pakai: `Dockerfile`, `requirements-api.txt`, `railway.json`.
Langkah lengkap (push → Deploy from GitHub → verifikasi) ada di **`DEPLOY.md`**.
```bash
# ringkas:
git add -f weights/best.pt weights/best.onnx && git commit -am "deploy" && git push
# lalu di railway.app: New Project > Deploy from GitHub repo (baca Dockerfile + railway.json)
# Settings > Networking > Generate Domain (target port 8000)
python smoke_test.py https://<APP>.up.railway.app   # verifikasi publik
```

## Endpoint
| Method | Path | Input | Fungsi |
|---|---|---|---|
| GET | `/health` | – | status model (ONNX/PT) |
| POST | `/detect` | image | deteksi person + semua bbox |
| POST | `/ppe` | image | status compliant/violation per orang + severity |
| POST | `/count` | image | jumlah orang |
| POST | `/track` | video | tracking person, ID konsisten + jumlah unik |
| POST | `/annotate` | image | PNG teranotasi (hijau=compliant/merah=violation) |

Error handling: 400 (kosong/rusak), 415 (tipe tak didukung), 500 (inference gagal).

## Reproduce
Seed=42 di `train.py` & `eval.py`. `eval_report.json` berisi metrik final.

## Atribusi
- Dataset: **Construction Site Safety Image Dataset** — Roboflow Universe Projects (CC BY 4.0).
  https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety
- Detection: Ultralytics YOLOv8. Tracking: ByteTrack.
- Lihat `AI_USAGE_LOG.md` untuk penggunaan AI.
