# Image produksi untuk dashboard Streamlit (streamlit_app.py) — deploy Railway.
# Catatan: build CPU-only. Torch dipasang dari index CPU-only agar
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

# 2) Sisa dependensi runtime dashboard (streamlit + ultralytics + opencv).
#    ultralytics menarik opencv-python (reguler, ABI numpy 2.x) yang menimpa
#    opencv-python-headless pinned -> "numpy._core.multiarray failed to import".
#    Solusi: install, uninstall opencv non-headless, lalu reinstall (headless).
COPY requirements-streamlit.txt .
RUN pip install -r requirements-streamlit.txt
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir -r requirements-streamlit.txt

# 3) Kode + konfigurasi + bobot
#    (image-endpoint pakai best.onnx bila ada, tracking pakai best.pt)
COPY app/ ./app/
COPY weights/ ./weights/
COPY .streamlit/ ./.streamlit/
COPY streamlit_app.py edge_case_test.py eval_report.json ./

EXPOSE 8501

# Railway menyuntik $PORT; default 8501 untuk run lokal di container.
# --server.address 0.0.0.0 wajib agar bisa diakses dari luar container.
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
