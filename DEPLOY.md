# Deploy ke Railway — Inference Endpoint (Tahap 6.5 / Must Have 5)

Men-deploy **FastAPI** (`app/main.py`) sebagai inference endpoint publik yang
bisa diakses juri. Endpoint: `/health /detect /ppe /count /track /annotate`,
plus Swagger UI di `/docs` dan dashboard di `/`.

## Apa yang sudah disiapkan di repo

| File | Peran |
|---|---|
| **`Dockerfile`** | Image produksi. Torch CPU-only (hindari CUDA ~2 GB), lalu `requirements-api.txt`, lalu `app/` + `weights/`. Start: `uvicorn app.main:app` di `$PORT`. |
| **`requirements-api.txt`** | Dependensi runtime ramping (tanpa streamlit/requests). |
| **`railway.json`** | Konfigurasi Railway: builder **DOCKERFILE**, healthcheck `/health`, restart on-failure. |
| **`weights/best.pt` + `best.onnx`** | Bobot ikut ter-commit (image-endpoint pakai ONNX, tracking pakai `.pt`). |

> Perintah produksi sudah diverifikasi lokal: `uvicorn app.main:app` → `smoke_test.py` **ALL PASS** (detect 147ms, ppe 111ms, count 103ms, track 7 ID, error-case 400). Railway membaca `$PORT` otomatis; CMD di Dockerfile memakai `${PORT:-8000}`.

## Langkah deploy

### 0. Commit & push (wajib — Railway build dari repo)
```bash
git add -A
git add -f weights/best.pt weights/best.onnx   # bypass .gitignore untuk bobot
git commit -m "Tahap 6.5: konfigurasi deploy Railway (Docker + railway.json)"
git push origin main
```

### 1A. Deploy via dashboard (paling mudah)
1. Login ke <https://railway.app> (bisa via GitHub).
2. **New Project ▸ Deploy from GitHub repo** → pilih
   `Human-Safety-Monitoring-System`.
3. Railway mendeteksi `Dockerfile` + `railway.json` → build mulai otomatis
   (5–10 menit pertama kali; torch+ultralytics cukup besar).
4. Setelah build sukses, buka tab **Settings ▸ Networking ▸ Generate Domain**
   untuk mendapat URL publik. Jika diminta target port, isi **8000**
   (sesuai `EXPOSE` di Dockerfile).

### 1B. Alternatif via Railway CLI
```bash
npm i -g @railway/cli
railway login
railway init           # buat / pilih project
railway up             # build & deploy dari Dockerfile
railway domain         # generate URL publik
```

### 2. Verifikasi publik
```bash
curl https://<APP>.up.railway.app/health          # -> {"status":"ok",...}
python smoke_test.py https://<APP>.up.railway.app  # semua endpoint + error-case
```
Buka `https://<APP>.up.railway.app/docs` (Swagger) dan `/` (dashboard) di browser.

### 3. Update README
Ganti placeholder URL di README dengan URL publik final.

## ⚠️ Catatan penting

- **Port:** Railway menyuntik `PORT` saat runtime; Dockerfile sudah memakainya
  (`${PORT:-8000}`). Tak perlu set manual. Jika domain 502, cek **target port =
  8000** di Settings ▸ Networking.
- **Memori:** model + torch cukup berat. Railway trial memberi RAM lebih lega
  dari free tier lain, tapi `/track` (video) tetap paling berat — demo pakai
  **klip pendek** (`samples/clip.mp4`, ~10 dtk; sudah di-tune `vid_stride=2,
  imgsz=480`). Image endpoints (`/detect /ppe /count /annotate`) ringan & aman.
- **Biaya/limit:** Railway trial = kredit terbatas + sleep saat kredit habis.
  Warm-up `/health` sebelum demo ke juri.
- **Fallback demo:** siapkan `ngrok http 8000` dari lokal
  (`uvicorn app.main:app`) bila platform bermasalah — endpoint & smoke test
  identik.
