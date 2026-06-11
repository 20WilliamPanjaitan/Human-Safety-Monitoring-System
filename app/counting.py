"""People counting — detection-based (bukan density estimation).

count_image: jumlah box person pada satu gambar.
count_video_unique: jumlah track_id unik melintas pada video.
"""

PERSON_CLASSES = {"Person", "person"}


def count_image(persons):
    return len([p for p in persons if p.get("class") in PERSON_CLASSES])


def count_video_unique(track_stream):
    seen = set()
    for det in track_stream:
        if det.get("class") in PERSON_CLASSES:
            seen.add(det["track_id"])
    return len(seen)
