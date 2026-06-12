"""Uji & demo logika PPE + counting — Tahap 5 / Must Have 3.

Dua bagian:
  1. Unit test sintetis (deterministik, tanpa model) -> bukti logika benar.
  2. Uji pada gambar test nyata -> bukti integrasi dengan model.

Jalankan:
  python ppe_demo.py                      # unit test saja (cepat)
  python ppe_demo.py --images a.jpg b.jpg # + uji gambar nyata
"""
import argparse
import json

from app.counting import count_image
from app.ppe_logic import assess_ppe, summarize


def _person(x1, y1, x2, y2, conf=0.9):
    return {"bbox": [x1, y1, x2, y2], "conf": conf, "class": "Person"}


def _item(cls, x1, y1, x2, y2, conf=0.9):
    return {"bbox": [x1, y1, x2, y2], "conf": conf, "class": cls}


def unit_tests():
    """Skenario kanonik. Return jumlah (pass, total)."""
    p = _person(0, 0, 100, 200)
    cases = [
        # nama, persons, items, cek(report)
        ("Pakai helm+rompi -> compliant",
         [p], [_item("Hardhat", 10, 0, 90, 40), _item("Safety-Vest", 10, 60, 90, 150)],
         lambda r: r[0]["status"]["helmet"] == "ok"
                   and r[0]["status"]["vest"] == "ok"
                   and r[0]["compliant"] is True
                   and r[0]["violations"] == []),

        ("Tanpa helm (NO-Hardhat) -> violation helmet",
         [p], [_item("NO-Hardhat", 10, 0, 90, 40), _item("Safety-Vest", 10, 60, 90, 150)],
         lambda r: r[0]["status"]["helmet"] == "violation"
                   and r[0]["violations"] == ["helmet"]
                   and r[0]["compliant"] is False
                   and r[0]["severity"] == "low"),

        ("Helm & rompi sama-sama NO -> 2 violation, severity high",
         [p], [_item("NO-Hardhat", 10, 0, 90, 40), _item("NO-Safety-Vest", 10, 60, 90, 150)],
         lambda r: r[0]["violations"] == ["helmet", "vest"]
                   and r[0]["severity"] == "high"),

        ("PPE tak terdeteksi -> unknown, BUKAN violation",
         [p], [],
         lambda r: r[0]["status"]["helmet"] == "unknown"
                   and r[0]["violations"] == []
                   and r[0]["unverified"] == ["helmet", "vest"]
                   and r[0]["compliant"] is True),

        ("Tanpa masker saja -> TETAP compliant (mask tak wajib)",
         [p], [_item("Hardhat", 10, 0, 90, 40), _item("Safety-Vest", 10, 60, 90, 150),
               _item("NO-Mask", 30, 5, 70, 30)],
         lambda r: r[0]["status"]["mask"] == "violation"
                   and "mask" not in r[0]["violations"]
                   and r[0]["compliant"] is True),

        ("PPE orang lain (tak overlap) -> tak ter-asosiasi",
         [p], [_item("Hardhat", 500, 500, 560, 540)],
         lambda r: r[0]["status"]["helmet"] == "unknown"),

        ("Bukti positif menang atas negatif (conf lebih tinggi)",
         [p], [_item("Hardhat", 10, 0, 90, 40, conf=0.9),
               _item("NO-Hardhat", 12, 0, 88, 40, conf=0.3)],
         lambda r: r[0]["status"]["helmet"] == "ok"),
    ]

    npass = 0
    for name, persons, items, check in cases:
        report = assess_ppe(persons, items)
        ok = bool(check(report))
        npass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print("        ->", json.dumps(report, ensure_ascii=False))

    # counting
    persons5 = [_person(i * 100, 0, i * 100 + 80, 200) for i in range(5)]
    cnt_ok = count_image(persons5) == 5
    npass += cnt_ok
    print(f"  [{'PASS' if cnt_ok else 'FAIL'}] count_image(5 orang) == 5")

    total = len(cases) + 1
    return npass, total


def test_real_images(paths):
    from app.detector import Detector
    from app.main import _split_dets
    det = Detector()
    for path in paths:
        dets = det.detect(path)
        persons, items = _split_dets(dets)
        report = assess_ppe(persons, items)
        print(f"\n=== {path} ===")
        print("count person:", count_image(persons))
        print("summary:", json.dumps(summarize(report), ensure_ascii=False))
        for i, r in enumerate(report):
            print(f"  person {i}: status={r['status']} "
                  f"violations={r['violations']} severity={r['severity']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=[])
    args = ap.parse_args()

    print("UNIT TEST (sintetis):")
    npass, total = unit_tests()
    print(f"\n  {npass}/{total} PASS")

    if args.images:
        print("\nUJI GAMBAR NYATA:")
        test_real_images(args.images)

    raise SystemExit(0 if npass == total else 1)


if __name__ == "__main__":
    main()
