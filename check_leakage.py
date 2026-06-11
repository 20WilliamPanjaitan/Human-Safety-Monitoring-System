"""Cek leakage antar split via hash gambar — Tahap 1 / Must Have 4.

Memastikan tidak ada gambar identik (byte-level MD5) yang muncul di lebih dari
satu split (train/valid/test). Frame duplikat = kebocoran data.

Contoh:
  python check_leakage.py --root construction-site-safety
"""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def md5(path: Path):
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="construction-site-safety")
    args = ap.parse_args()

    root = Path(args.root)
    hash_to_splits = defaultdict(set)
    counts = defaultdict(int)

    for split in ["train", "valid", "test"]:
        for img in (root / split / "images").glob("*"):
            if img.suffix.lower() in IMG_EXT:
                counts[split] += 1
                hash_to_splits[md5(img)].add(split)

    dups = {h: s for h, s in hash_to_splits.items() if len(s) > 1}

    print("Jumlah gambar per split:", dict(counts))
    if dups:
        print(f"LEAKAGE DITEMUKAN: {len(dups)} gambar muncul di >1 split.")
        for h, s in list(dups.items())[:10]:
            print(f"  {h[:10]}... -> {sorted(s)}")
    else:
        print("OK: 0 gambar duplikat antar split.")


if __name__ == "__main__":
    main()
