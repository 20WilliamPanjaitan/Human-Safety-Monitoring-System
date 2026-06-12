# Model Provenance — `weights/best.pt`

Dokumen ini menjamin jejak **training → bobot → eval** konsisten dan dapat
dipertanggungjawabkan. Disertakan di Technical Report (Tahap 8).

## Identitas bobot final

| Atribut | Nilai |
|---|---|
| File | `weights/best.pt` |
| Arsitektur | YOLOv8s (multi-class: person + PPE, 7 kelas) |
| Ukuran | 59.510.127 byte (~59,5 MB) |
| MD5 | `2bdce55a6d22c71edb84bb9d7e1db887` |
| Kelas | `Person, Hardhat, NO-Hardhat, Safety-Vest, NO-Safety-Vest, Mask, NO-Mask` |

## Sumber training — penting (ada dua percobaan)

Model final **bukan** dari `train.py` lokal, melainkan dari notebook Colab GPU.
Keduanya memakai konfigurasi & seed identik; hanya hardware dan jumlah epoch
yang berbeda.

| | Run lokal (`train.py`) | **Run final (Colab)** |
|---|---|---|
| Artefak | `runs/yolov8s_ppe2/` | `notebooks/train_colab_T4.ipynb` → `weights/best.pt` |
| Hardware | CPU | **Tesla T4 (15 GB), CUDA** |
| Epoch | **5** (di-abort, terlalu lambat) | **50** (penuh) |
| MD5 bobot | `4fd26cb699819e15420369e4076de60b` | `2bdce55a6d22c71edb84bb9d7e1db887` ✅ dipakai |
| Status | dibuang | **dipakai untuk eval & deploy** |

> Run lokal 5-epoch inilah yang dulu menghasilkan `eval_report.json` versi lama
> (mAP@0.5 ≈ 0.62). Setelah `best.pt` diganti dengan model Colab 50-epoch,
> `eval_report.json` di-generate ulang (lihat angka di bawah).

## Lingkungan training (Colab)

- Ultralytics **8.3.0**, PyTorch **2.10.0+cu128**, Python **3.12.13**, NumPy **1.26.4**
- GPU: Tesla T4 (14913 MiB)
- Seed: **42** (`random`, `numpy`, `torch`, `cudnn.deterministic=True`)

## Hyperparameter (identik di `train.py` & notebook)

```
model=yolov8s.pt  epochs=50  imgsz=640  batch=16  seed=42
optimizer=auto    lr0=0.01   warmup_epochs=3   close_mosaic=10
hsv_h=0.015  hsv_s=0.7  hsv_v=0.4  scale=0.5  translate=0.1
fliplr=0.5   flipud=0.0
```

## Data

- Dataset: Construction Site Safety (Roboflow), di-remap 10→7 kelas via
  `remap_labels.py`.
- Split: bawaan **2605 / 114 / 82** (train/valid/test).
- Leakage: `check_leakage.py` (2-lapis) → 0 duplikat MD5; **19 scene video
  tersebar antar split** → dilaporkan sebagai limitasi (metrik test sedikit
  optimistis). Lihat bagian Limitasi report.

## Hasil eval final (test split, seed 42)

Direproduksi lokal dari `weights/best.pt` final → ditulis ke `eval_report.json`.

| Metrik | Nilai |
|---|---|
| mAP@0.5 | **0.7776** |
| mAP@0.5:0.95 | 0.4563 |
| precision | 0.8830 |
| recall | 0.7332 |

Per-class AP@0.5 terlemah: `NO-Hardhat` & `NO-Mask` (kelas pelanggaran, objek
kecil/jarang) — lihat `eval_report.json` untuk angka lengkap.

## Reproduksi

```bash
# 1. (sekali) training di Colab T4 — jalankan notebooks/train_colab_T4.ipynb
#    -> unduh runs/yolov8s_ppe/weights/best.pt menjadi weights/best.pt
# 2. verifikasi integritas bobot
python -c "import hashlib; print(hashlib.md5(open('weights/best.pt','rb').read()).hexdigest())"
#    -> 2bdce55a6d22c71edb84bb9d7e1db887
# 3. eval reproducible (hasil identik tiap run)
python eval.py --data data.yaml --model weights/best.pt --split test --seed 42
```
