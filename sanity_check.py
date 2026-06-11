"""Sanity check visual — Tahap 1.

Render N gambar acak + bbox-nya (label sudah ter-remap ke 7 kelas) untuk
verifikasi manual bahwa label benar secara visual. Output disimpan sebagai
PNG teranotasi agar bisa dibuka & diperiksa mata.

Contoh:
  python sanity_check.py --root construction-site-safety --split train --n 5
"""
import argparse
import random
from pathlib import Path

import cv2

NAMES = {0: "Person", 1: "Hardhat", 2: "NO-Hardhat", 3: "Safety-Vest",
         4: "NO-Safety-Vest", 5: "Mask", 6: "NO-Mask"}
COLORS = {0: (255, 200, 0), 1: (0, 200, 0), 2: (0, 0, 255), 3: (0, 200, 200),
          4: (0, 100, 255), 5: (200, 0, 200), 6: (128, 0, 255)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def draw(img_path: Path, lbl_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ! gagal baca {img_path.name}")
        return
    h, w = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0])
            cx, cy, bw, bh = (float(x) for x in p[1:])
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            col = COLORS.get(c, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, NAMES.get(c, str(c)), (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    print(f"  -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="construction-site-safety")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    pick = random.sample(imgs, min(args.n, len(imgs)))

    out_dir = root / "_sanity"
    out_dir.mkdir(exist_ok=True)
    print(f"Render {len(pick)} gambar dari split '{args.split}':")
    for p in pick:
        draw(p, lbl_dir / (p.stem + ".txt"), out_dir / (p.stem + "_annot.png"))
    print(f"\nBuka folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
