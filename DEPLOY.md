# Deploy ke Render — Inference Endpoint (Tahap 6.5 / Must Have 5)

Men-deploy **FastAPI** (`app/main.py`) sebagai inference endpoint publik yang
bisa diakses juri. Endpoint yang tersedia: `/health /detect /ppe /count /track
/annotate`, plus Swagger UI di `/docs` dan dashboard di `/`.

## Apa yang sudah disiapkan di repo

| File | Peran |
|---|---|
| **`Dockerfile`** | Image produksi. Torch CPU-only (hindari CUDA ~2 GB), lalu `requirements-api.txt`, lalu `app/` + `weights/`. Start: `uvicorn app.main:app` di `$PORT`. |
| **`requirements-api.txt`** | Dependensi runtime ramping (tanpa streamlit/requests). |
| **`render.yaml`** | Blueprint Render: web service Docker, plan free, region Singapore, healthcheck `/health`. |
| **`weights/best.pt` + `best.onnx`** | Bobot ikut ter-commit (image-endpoint pakai ONNX, tracking pakai `.pt`). |

> Perintah produksi sudah diverifikasi lokal: `uvicorn app.main:app` → `smoke_test.py` **ALL PASS** (detect 147ms, ppe 111ms, count 103ms, track 7 ID, error-case 400).

## Langkah deploy

### 0. Commit & push (wajib — Render build dari repo)
Pastikan bobot & file deploy ikut ter-commit:
```bash
git add -A
git add -f weights/best.pt weights/best.onnx   # bypass .gitignore untuk bobot
git commit -m "Tahap 6.5: konfigurasi deploy Render (Docker + render.yaml)"
git push origin main
```

### 1. Buat service di Render
1. Login ke <https://dashboard.render.com> (daftar gratis, bisa via GitHub).
2. **New ▸ Blueprint** → pilih repo `Human-Safety-Monitoring-System`.
3. Render membaca `render.yaml` otomatis → **Apply**. Build Docker mulai
   (5–10 menit pertama kali; torch+ultralytics cukup besar).

   *(Alternatif tanpa Blueprint: New ▸ Web Service → Runtime **Docker** →
   pilih repo → plan Free → region Singapore → Create.)*

### 2. Verifikasi publik
Setelah status **Live**, URL berbentuk `https://human-safety-monitoring.onrender.com`.
```bash
curl https://<APP>.onrender.com/health          # -> {"status":"ok",...}
python smoke_test.py https://<APP>.onrender.com  # semua endpoint + error-case
```
Buka `https://<APP>.onrender.com/docs` (Swagger) dan `/` (dashboard) di browser.

### 3. Update README
Ganti placeholder URL di README dengan URL publik final.

## ⚠️ Catatan penting

- **Cold start:** plan free tidur setelah ~15 menit idle; request pertama bisa
  ~30–60 dtk. **Warm-up `/health` beberapa menit sebelum demo** ke juri.
- **Memori (risiko utama):** free tier = **512 MB RAM**. Torch + Ultralytics +
  model bisa mendekati batas, terutama saat `/track` (video) memproses banyak
  frame → berpotensi **OOM / restart**. Mitigasi:
  - Demo `/track` dengan **klip pendek** (`samples/clip.mp4`, ~10 dtk) — sudah
    di-tune (`vid_stride=2, imgsz=480`).
  - Jika sering OOM, naikkan ke plan **Starter (512 MB→ lebih)** sementara saat
    penjurian, atau jalankan tracking dengan klip lebih kecil.
  - Image endpoints (`/detect /ppe /count /annotate`) jauh lebih ringan & aman.
- **Fallback demo:** siapkan `ngrok http 8000` dari lokal (`uvicorn app.main:app`)
  bila platform bermasalah saat demo — endpoint & smoke test identik.

## Region
`singapore` dipilih untuk latency terdekat ke Indonesia. Bisa diubah di
`render.yaml` (`oregon`, `frankfurt`, `ohio`, `virginia`).
