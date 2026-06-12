# Image produksi untuk inference endpoint (FastAPI) — Tahap 6.5 deploy Render.
# Catatan: Render free tier = CPU. Torch dipasang dari index CPU-only agar
# image tidak menarik paket CUDA (~2 GB) yang sia-sia di CPU.
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0 dibutuhkan OpenCV walau pakai headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 1) Torch + torchvision CPU-only (pinned, dipakai ultralytics & ByteTrack)
RUN pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Sisa dependensi runtime API (tanpa streamlit/requests — bukan untuk server)
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 3) Kode + bobot (image-endpoint pakai best.onnx bila ada, tracking pakai best.pt)
COPY app/ ./app/
COPY weights/ ./weights/

EXPOSE 8000

# Render menyuntik $PORT; default 8000 untuk run lokal di container.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
