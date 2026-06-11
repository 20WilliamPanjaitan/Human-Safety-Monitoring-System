"""Person tracking dengan ByteTrack (built-in Ultralytics).

TODO (Tahap 4): tuning track_buffer/match_thresh, render anotasi track_id.
"""
from ultralytics import YOLO

_model = None


def _get_model(weights="weights/best.pt"):
    global _model
    if _model is None:
        _model = YOLO(weights)
    return _model


def track_video(source, weights="weights/best.pt"):
    """Yield {track_id, bbox, class} per deteksi terkonfirmasi, streaming."""
    model = _get_model(weights)
    results = model.track(
        source=source,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
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
