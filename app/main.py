"""FastAPI entrypoint — Human Safety Monitoring API (Tahap 6).

Endpoint: /health /detect /ppe /track /count /annotate
- Model di-load SEKALI saat startup (image: ONNX Runtime; video/track: .pt).
- Tiap response menyertakan latency_ms.
- Error handling: 400 (kosong/rusak), 415 (tipe tak didukung), 500 (inference gagal).
"""
import os
import tempfile
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.counting import count_image
from app.detector import Detector
from app.ppe_logic import ALL_PPE_CLASSES, assess_ppe, summarize
from app.tracker import track_summary

# Inference gambar pakai ONNX (ringan utk free-tier CPU); fallback ke .pt.
WEIGHTS = os.getenv("WEIGHTS", "weights/best.pt")
# Tracking butuh model torch (.pt) untuk ByteTrack.
TRACK_WEIGHTS = os.getenv("TRACK_WEIGHTS", "weights/best.pt")
PERSON_IDX = 0  # index kelas Person di data.yaml (utk filter tracking)

app = FastAPI(title="Human Safety Monitoring API", version="1.0")
det = Detector(WEIGHTS)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/", include_in_schema=False)
def ui():
    """Web UI dashboard (upload gambar/video, lihat hasil teranotasi)."""
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

PERSON_CLASSES = {"Person", "person"}
PPE_CLASSES = ALL_PPE_CLASSES

IMG_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/avi",
               "application/octet-stream"}


def read_image(file_bytes):
    if not file_bytes:
        raise HTTPException(400, "Empty file")
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


def _require_type(file: UploadFile, allowed):
    """415 jika content-type tak termasuk allowed (None = lewati cek)."""
    ct = (file.content_type or "").lower()
    if ct and allowed is not None and ct not in allowed:
        raise HTTPException(415, f"Unsupported media type: {ct}")


def _detect(img):
    """Bungkus inference; error model internal -> 500 (bukan 400)."""
    try:
        return det.detect(img)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Inference failed: {e}")


def _split_dets(dets):
    persons = [d for d in dets if d["class"] in PERSON_CLASSES]
    items = [d for d in dets if d["class"] in PPE_CLASSES]
    return persons, items


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(WEIGHTS), "version": "1.0"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    _require_type(file, IMG_TYPES)
    img = read_image(await file.read())
    dets = _detect(img)
    persons = [d for d in dets if d["class"] in PERSON_CLASSES]
    return {
        "num_persons": len(persons),
        "detections": dets,
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/ppe")
async def ppe(file: UploadFile = File(...)):
    t0 = time.time()
    _require_type(file, IMG_TYPES)
    img = read_image(await file.read())
    persons, items = _split_dets(_detect(img))
    report = assess_ppe(persons, items)
    return {
        **summarize(report),
        "persons": report,
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/count")
async def count(file: UploadFile = File(...)):
    t0 = time.time()
    _require_type(file, IMG_TYPES)
    img = read_image(await file.read())
    persons, _ = _split_dets(_detect(img))
    return {
        "count": count_image(persons),
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/track")
async def track(file: UploadFile = File(...)):
    """Tracking person pada video: track_id konsisten + jumlah unik (MH2)."""
    t0 = time.time()
    _require_type(file, VIDEO_TYPES)
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.close()
        try:
            summary = track_summary(tmp.name, weights=TRACK_WEIGHTS,
                                    classes=[PERSON_IDX])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Tracking failed: {e}")
    finally:
        os.unlink(tmp.name)
    return {**summary, "latency_ms": round((time.time() - t0) * 1000)}


def _annotate_image(img):
    """Gambar bbox person berwarna by-kepatuhan + kotak PPE. Return BGR frame."""
    persons, items = _split_dets(_detect(img))
    report = assess_ppe(persons, items)
    # PPE items (abu-abu tipis + label kelas & conf)
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        lbl = f"{it['class']} {it.get('conf', 0):.2f}"
        cv2.putText(img, lbl, (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    # Person: hijau=compliant, merah=violation, kuning=unverified saja
    for r in report:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        if r["violations"]:
            col, tag = (0, 0, 255), "VIOLATION:" + ",".join(r["violations"])
        elif r["unverified"]:
            col, tag = (0, 200, 255), "UNVERIFIED"
        else:
            col, tag = (0, 200, 0), "COMPLIANT"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


@app.post("/annotate")
async def annotate(file: UploadFile = File(...)):
    """Kembalikan gambar teranotasi (PNG) — bisa langsung dilihat di /docs."""
    _require_type(file, IMG_TYPES)
    img = read_image(await file.read())
    out = _annotate_image(img)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise HTTPException(500, "Encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")
