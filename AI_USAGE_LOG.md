# AI Usage Log — Human Safety Monitoring System

Disclosure pemakaian AI selama Fase 2 (wajib, bagian dari Technical Report).
Filosofi: AI dipakai sebagai asisten; setiap output di-review, diuji, dan bisa
dijelaskan saat Live Defense.

## Tools yang digunakan
| Tool | Versi | Untuk apa |
|---|---|---|
| Claude Code (Anthropic) | Opus 4.8 | Scaffold repo, generate/refactor kode, debugging, deployment config, dokumentasi, report |
| Ultralytics YOLOv8 | 8.3.0 | Training, inference, eval `val()`, export ONNX |
| ByteTrack | built-in Ultralytics | Person tracking |

## Pola penggunaan
- **Agentic / multi-step:** AI menyusun struktur, menghasilkan skeleton kode,
  lalu mengeksekusi & menguji. Setiap perubahan diverifikasi manual.
- **Debugging:** diagnosis error runtime/deploy (lihat Verifikasi).
- **Refactor ke production-quality:** menyederhanakan & merapikan kode hasil AI.
- **Dokumentasi:** README, struktur proyek, Technical Report, panduan deploy.
- **TIDAK** memakai LLM untuk synthetic data / golden eval — semua metrik dari
  training & eval nyata, sehingga aturan review 30% sampel tidak berlaku.

## History prompt penting
**Fase awal (T0–T6) — scaffold & pipeline:**
1. "Buatkan roadmap & tahapan teknis Fase 2 terstruktur dan terukur" → `ROADMAP-FASE2.md`.
2. "Generate scaffold T0 (struktur folder, requirements, Dockerfile, skeleton)".
3. Pipeline data/model: `remap_labels.py`, `check_leakage.py`, `train.py`, `eval.py`.
4. Modul inti: `detector.py`, `ppe_logic.py` (containment + bukti positif NO-*), `tracker.py`, `counting.py`, `main.py` (FastAPI).

**Fase lanjutan (sesi ini):**
5. "Buat UI Streamlit modern & profesional" → `streamlit_app.py` + tema.
6. "Jelaskan fungsi confidence threshold & IoU (NMS)" (penjelasan teknis).
7. "Jalankan Tahap 7 — Visualisasi & Edge Case Test" → `edge_case_test.py`,
   `report/EDGE_CASES.md`, aset demo (pair compliant/violation, edge cases).
8. "Integrasikan edge case ke front end" → tab **Robustness Test** di Streamlit.
9. "Coba tab Robustness Test di browser" → verifikasi via Playwright (browser nyata).
10. "Analisis seluruh artefak → file MD penjelasan tiap file/folder" → `PROJECT_STRUCTURE.md`.
11. "Apakah project memenuhi kriteria Peserta_AIEngineer.pdf?" → audit Must/Nice/Extraordinary.
12. "Siapkan deployment" → `Dockerfile` (torch CPU-only), `requirements-api.txt`,
    `railway.json`, `DEPLOY.md`. (Awalnya Render, lalu diubah ke Railway.)
13. Debug deploy: error `import cv2` (`numpy._core.multiarray failed to import`).
14. "Susun Technical Report lengkap sesuai ketentuan" → `report/technical_report.{tex,pdf}`.

## Verifikasi & judgment (menolak/menyesuaikan saran AI)
- **ABI NumPy:** mendiagnosis `opencv-python-headless==4.10.0.84` (Linux) di-build
  terhadap NumPy 2.x sementara `numpy==1.26.4` (dibutuhkan torch 2.3.1) terpasang
  → `import cv2` gagal. Solusi: pin `opencv-python-headless==4.9.0.80` (ABI NumPy 1.x).
- **Streamlit API:** memperbaiki `use_container_width` (tak ada di v1.37) →
  `use_column_width`; memperbaiki konstruksi tuple `variants` yang sempat keliru.
- **Kualitas aset:** menolak gambar demo "compliant" pertama (subjek terlalu
  kecil), memindai test set untuk subjek besar & jelas.
- **Kejujuran laporan:** saat low-light ternyata TIDAK menurunkan deteksi,
  narasi `EDGE_CASES.md` diperbaiki agar sesuai data (model robust), bukan
  mengklaim degradasi yang tak terjadi.
- **Pengujian:** `ppe_demo.py` 8/8 unit test PASS; `smoke_test.py` semua endpoint
  + error-case PASS (lokal); `eval.py` reproducible (seed 42, identik antar-run).
