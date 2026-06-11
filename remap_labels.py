"""Remap label Construction Site Safety (10 kelas) -> 7 kelas relevan — Tahap 1.

10 kelas asli (index Roboflow):
  0 Hardhat, 1 Mask, 2 NO-Hardhat, 3 NO-Mask, 4 NO-Safety Vest,
  5 Person, 6 Safety Cone, 7 Safety Vest, 8 machinery, 9 vehicle

7 kelas target (lihat data.yaml):
  0 Person, 1 Hardhat, 2 NO-Hardhat, 3 Safety-Vest, 4 NO-Safety-Vest,
  5 Mask, 6 NO-Mask

Baris label dengan kelas yang dibuang (Cone, machinery, vehicle) akan dihapus.
File .txt yang jadi kosong tetap dibiarkan (gambar background valid).

Contoh:
  python remap_labels.py --root construction-site-safety
"""
import argparse
from pathlib import Path

# index asli -> index baru; yang tidak ada di map = dibuang
OLD_TO_NEW = {
    5: 0,  # Person        -> 0
    0: 1,  # Hardhat        -> 1
    2: 2,  # NO-Hardhat     -> 2
    7: 3,  # Safety Vest    -> 3
    4: 4,  # NO-Safety Vest -> 4
    1: 5,  # Mask           -> 5
    3: 6,  # NO-Mask        -> 6
}


def remap_file(txt_path: Path):
    new_lines = []
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        old = int(parts[0])
        if old in OLD_TO_NEW:
            parts[0] = str(OLD_TO_NEW[old])
            new_lines.append(" ".join(parts))
    txt_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="construction-site-safety")
    args = ap.parse_args()

    root = Path(args.root)
    label_files = list(root.glob("**/labels/*.txt"))
    if not label_files:
        print(f"Tidak ada file label di {root}/**/labels/*.txt — cek path.")
        return
    for f in label_files:
        remap_file(f)
    print(f"Remap selesai untuk {len(label_files)} file label.")


if __name__ == "__main__":
    main()
