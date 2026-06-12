# Image produksi untuk dashboard Streamlit (streamlit_app.py) — deploy Railway.
# Catatan: build CPU-only. Torch dipasang dari index CPU-only agar
# image tidak menarik paket CUDA (~2 GB) yang sia-sia di CPU.
FROM python:3.11-slim

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# libgl1 + libglib2.0-0 dibutuhkan OpenCV walau pakai headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# constraints.txt mengunci numpy==1.26.4 di SEMUA langkah pip di bawah.
COPY constraints.txt .

# 1) Torch + torchvision CPU-only (pinned, dipakai ultralytics & ByteTrack).
#    Pakai constraint agar torch tak menarik numpy 2.x.
RUN pip install -c constraints.txt torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 2) Sisa dependensi runtime dashboard (streamlit + ultralytics + opencv).
#    ultralytics mendeklarasikan opencv-python (ABI numpy 2.x) yang bentrok di
#    namespace `cv2` dengan headless pinned -> "numpy._core.multiarray failed".
COPY requirements-streamlit.txt .
RUN pip install -c constraints.txt -r requirements-streamlit.txt
# Buang opencv non-headless, lalu paksa-pasang HANYA headless tanpa menyentuh
# numpy (--no-deps), supaya ABI cv2 cocok dengan numpy 1.26.4.
RUN pip uninstall -y opencv-python opencv-contrib-python
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    opencv-python-headless==4.9.0.80
# Gagalkan build SEKARANG bila ABI numpy/opencv masih mismatch (versi tercetak).
RUN python -c "import numpy, cv2; print('BUILD OK — numpy', numpy.__version__, '| cv2', cv2.__version__)"

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
