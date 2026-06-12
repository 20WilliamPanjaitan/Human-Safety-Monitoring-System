"""Person tracking dengan ByteTrack (built-in Ultralytics) — Tahap 4 / MH2.

- track_video(): stream {track_id, bbox, class} per deteksi (dipakai API /track).
- render_tracked_video(): tulis video teranotasi dengan track_id + bbox berwarna.

Config tracker tuned ada di app/bytetrack_custom.yaml (track_buffer=60 untuk okulasi).
"""
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_models = {}

# Path absolut ke config tuned agar tak tergantung cwd.
_TRACKER_CFG = str(Path(__file__).with_name("bytetrack_custom.yaml"))


def _get_model(weights="weights/best.pt"):
    """Cache per-path; person tracking pakai COCO, PPE pakai custom best.pt."""
    if weights not in _models:
        _models[weights] = YOLO(weights)
    return _models[weights]


def _color(track_id):
    """Warna stabil & berbeda per track_id (BGR)."""
    hue = (int(track_id) * 47) % 180
    px = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def track_video(source, weights="weights/best.pt", classes=None):
    """Yield {track_id, bbox, class} per deteksi terkonfirmasi, streaming.

    classes: list index kelas yang dilacak (mis. [0] untuk person saja
    saat memakai model COCO). None = semua kelas.
    """
    model = _get_model(weights)
    results = model.track(
        source=source,
        tracker=_TRACKER_CFG,
        persist=True,
        stream=True,
        verbose=False,
        classes=classes,
    )
    for r in results:
        for b in r.boxes:
            if b.id is None:          # track belum terkonfirmasi
                continue
            yield {
                "track_id": int(b.id[0]),
                "bbox": [round(v, 1) for v in b.xyxy[0].tolist()],
                "class": model.names[int(b.cls[0])],
            }


def track_summary(source, weights="weights/best.pt", classes=None, max_frames=None,
                  vid_stride=2, imgsz=480):
    """Ringkasan tracking untuk API /track (JSON ringkas, bukan video).

    Return {num_frames, unique_ids, count_persons, tracks:[{track_id,class,frames}]}.
    `count_persons` = jumlah track_id unik = People Counting versi video (MH3).

    vid_stride: proses tiap-N frame (2 = lewati separuh) -> hemat waktu agar klip
        5-10 dtk selesai < 10 dtk di CPU. imgsz: resolusi inference lebih kecil
        (480 vs 640) untuk speed. Keduanya trade-off kecepatan vs ketelitian.
    """
    model = _get_model(weights)
    tracks = {}          # track_id -> {"class":..., "frames":n}
    num_frames = 0

    results = model.track(source=source, tracker=_TRACKER_CFG, persist=True,
                          stream=True, verbose=False, classes=classes,
                          vid_stride=vid_stride, imgsz=imgsz)
    for r in results:
        num_frames += 1
        for b in r.boxes:
            if b.id is None:
                continue
            tid = int(b.id[0])
            t = tracks.setdefault(tid, {"class": model.names[int(b.cls[0])], "frames": 0})
            t["frames"] += 1
        if max_frames and num_frames >= max_frames:
            break
    return {
        "num_frames": num_frames,
        "unique_ids": len(tracks),
        "count_persons": len(tracks),
        "tracks": [{"track_id": k, **v} for k, v in sorted(tracks.items())],
    }


def render_tracked_video(source, out_path="samples/clip_tracked.mp4",
                         weights="weights/best.pt", classes=None):
    """Render video teranotasi: bbox berwarna per-ID + label track_id & kelas.

    classes: lihat track_video. Return statistik {frames, unique_ids} untuk MH2.
    """
    model = _get_model(weights)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    writer = None
    seen_ids = set()
    frames = 0
    results = model.track(
        source=source,
        tracker=_TRACKER_CFG,
        persist=True,
        stream=True,
        verbose=False,
        classes=classes,
    )
    for r in results:
        frame = r.orig_img.copy()
        h, w = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, 25.0, (w, h))

        for b in r.boxes:
            if b.id is None:
                continue
            tid = int(b.id[0])
            seen_ids.add(tid)
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            cls = model.names[int(b.cls[0])]
            col = _color(tid)
            label = f"ID {tid} {cls}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        frames += 1

    if writer is not None:
        writer.release()
    return {"frames": frames, "unique_ids": len(seen_ids), "out": out_path}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="path video / glob frame")
    ap.add_argument("--out", default="samples/clip_tracked.mp4")
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="filter index kelas, mis. --classes 0 (person, COCO)")
    a = ap.parse_args()
    print(json.dumps(
        render_tracked_video(a.source, a.out, a.weights, a.classes), indent=2))
