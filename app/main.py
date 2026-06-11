"""FastAPI entrypoint — Human Safety Monitoring API.

Endpoint: /health /detect /ppe /track /count /annotate
Model di-load SEKALI saat startup. Tiap response sertakan latency_ms.
TODO (Tahap 6): lengkapi /track, /annotate; pindah inference ke best.onnx.
"""
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from app.counting import count_image
from app.detector import Detector
from app.ppe_logic import HELMET_CLASSES, MASK_CLASSES, VEST_CLASSES, assess_ppe

WEIGHTS = os.getenv("WEIGHTS", "weights/best.pt")

app = FastAPI(title="Human Safety Monitoring API", version="1.0")
det = Detector(WEIGHTS)

PERSON_CLASSES = {"Person", "person"}
PPE_CLASSES = HELMET_CLASSES | VEST_CLASSES | MASK_CLASSES


def read_image(file_bytes):
    if not file_bytes:
        raise HTTPException(400, "Empty file")
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")
    return img


def _split_dets(dets):
    persons = [d for d in dets if d["class"] in PERSON_CLASSES]
    items = [d for d in dets if d["class"] in PPE_CLASSES]
    return persons, items


@app.get("/health")
def health():
    return {"status": "ok", "model": "yolov8s", "version": "1.0"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    img = read_image(await file.read())
    dets = det.detect(img)
    persons = [d for d in dets if d["class"] in PERSON_CLASSES]
    return {
        "num_persons": len(persons),
        "detections": dets,
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/ppe")
async def ppe(file: UploadFile = File(...)):
    t0 = time.time()
    img = read_image(await file.read())
    persons, items = _split_dets(det.detect(img))
    report = assess_ppe(persons, items)
    return {
        "num_persons": len(persons),
        "persons": report,
        "violations_total": sum(0 if p["compliant"] else 1 for p in report),
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/count")
async def count(file: UploadFile = File(...)):
    t0 = time.time()
    img = read_image(await file.read())
    persons, _ = _split_dets(det.detect(img))
    return {
        "count": count_image(persons),
        "latency_ms": round((time.time() - t0) * 1000),
    }


@app.post("/track")
async def track(file: UploadFile = File(...)):
    # TODO (Tahap 4/6): simpan video ke temp, jalankan tracker.track_video, agregasi per frame.
    raise HTTPException(501, "Not implemented yet")


@app.post("/annotate")
async def annotate(file: UploadFile = File(...)):
    # TODO (Tahap 7): render bbox berwarna (hijau compliant / merah violation), return base64.
    raise HTTPException(501, "Not implemented yet")
