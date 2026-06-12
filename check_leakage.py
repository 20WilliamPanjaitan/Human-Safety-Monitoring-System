"""Cek leakage antar split — Tahap 1 / Must Have 4.

Dua lapis pemeriksaan:

1. **MD5 byte-identik** — gambar yang persis sama (byte) muncul di >1 split.
   Menangkap duplikat mentah.

2. **Scene/video** — dataset Construction Site Safety berisi banyak FRAME VIDEO
   (nama: `<scene>_mp4-<frame>_jpg.rf.<hash>.jpg`, juga `_MOV-`, `_AVI-`).
   Roboflow me-rename + meng-augmentasi tiap frame sehingga hash-nya BEDA,
   jadi MD5 buta terhadapnya. Padahal dua frame dari video yang sama nyaris
   identik — kalau satu di train dan satu di test = kebocoran nyata
   (model "menghafal" scene). Lapis ini mengelompokkan gambar per-scene dan
   menandai scene yang tersebar di >1 split.

Contoh:
  python check_leakage.py --root construction-site-safety
"""
import argparse
import hashlib
import re
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ["train", "valid", "test"]

# <scene>_(mp4|mov|avi)-<frame>...  -> group(1) = identitas scene/video.
# Contoh: "construction-2-_mp4-84_jpg.rf.<hash>.jpg" -> scene "construction-2-"
_SCENE_RE = re.compile(r"^(.*?)_(mp4|mov|avi)-\d+", re.IGNORECASE)


def md5(path: Path):
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def scene_of(filename: str):
    """Kembalikan id scene jika nama file adalah frame video, else None."""
    m = _SCENE_RE.match(filename)
    return m.group(1).lower() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="construction-site-safety")
    args = ap.parse_args()

    root = Path(args.root)
    hash_to_splits = defaultdict(set)
    scene_to_splits = defaultdict(set)   # scene -> {split, ...}
    scene_counts = defaultdict(lambda: defaultdict(int))  # scene -> split -> n
    counts = defaultdict(int)

    for split in SPLITS:
        for img in (root / split / "images").glob("*"):
            if img.suffix.lower() not in IMG_EXT:
                continue
            counts[split] += 1
            hash_to_splits[md5(img)].add(split)
            scene = scene_of(img.name)
            if scene:
                scene_to_splits[scene].add(split)
                scene_counts[scene][split] += 1

    print("Jumlah gambar per split:", dict(counts))
    print(f"Scene video terdeteksi: {len(scene_to_splits)}\n")

    ok = True

    # --- Lapis 1: MD5 byte-identik ---
    dup_md5 = {h: s for h, s in hash_to_splits.items() if len(s) > 1}
    if dup_md5:
        ok = False
        print(f"[MD5]   LEAKAGE: {len(dup_md5)} gambar byte-identik di >1 split.")
        for h, s in list(dup_md5.items())[:10]:
            print(f"        {h[:10]}... -> {sorted(s)}")
    else:
        print("[MD5]   OK: 0 gambar byte-identik antar split.")

    # --- Lapis 2: scene/video tersebar antar split ---
    dup_scene = {sc: sp for sc, sp in scene_to_splits.items() if len(sp) > 1}
    if dup_scene:
        ok = False
        print(f"[SCENE] LEAKAGE: {len(dup_scene)} scene video tersebar di >1 split.")
        for sc, sp in sorted(dup_scene.items()):
            dist = ", ".join(f"{k}:{scene_counts[sc][k]}"
                             for k in SPLITS if scene_counts[sc][k])
            print(f"        '{sc}' -> {sorted(sp)}  ({dist})")
    else:
        print("[SCENE] OK: tiap scene video berada dalam satu split saja.")

    print("\n==> VERDICT:", "BERSIH (0 leakage)" if ok else "ADA LEAKAGE — perbaiki split.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
