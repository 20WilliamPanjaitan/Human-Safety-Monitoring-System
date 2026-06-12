"""Logika kepatuhan APD — Tahap 5 / Must Have 3.

Mengubah deteksi multi-class (person + PPE) menjadi status compliant/violation
per orang, lewat asosiasi IoU person <-> PPE.

Dua keputusan desain penting:

1. **Bukti POSITIF pelanggaran, bukan sekadar ketiadaan.** Model dilatih
   mendeteksi kelas negatif eksplisit (`NO-Hardhat`, `NO-Safety-Vest`,
   `NO-Mask`). Kita pakai itu: helm "violation" hanya bila ada box `NO-Hardhat`
   yang overlap, bukan semata karena box `Hardhat` tak terdeteksi. Tiap kategori
   punya 3 status: `ok` / `violation` / `unknown` (tak terdeteksi). Ini menghindari
   false-violation saat helm cuma miss-detect.

2. **PPE wajib vs situasional.** Di lokasi konstruksi, helm & rompi umumnya wajib;
   masker situasional. Maka REQUIRED default = (helmet, vest); mask dipantau tapi
   tak memicu non-compliant. Ubah lewat argumen `required` bila skenario berbeda.
"""

# Kategori PPE -> kelas positif (dipakai) & negatif (pelanggaran), sesuai data.yaml.
# Alias huruf-kecil disertakan agar tahan jika nama kelas berubah.
PPE_CATEGORIES = {
    "helmet": {"pos": {"Hardhat", "helmet"}, "neg": {"NO-Hardhat", "no-hardhat"}},
    "vest":   {"pos": {"Safety-Vest", "Safety Vest", "vest"},
               "neg": {"NO-Safety-Vest", "NO-Safety Vest", "no-vest"}},
    "mask":   {"pos": {"Mask", "mask"}, "neg": {"NO-Mask", "no-mask"}},
}

# PPE yang memicu non-compliant bila dilanggar. Mask sengaja TIDAK wajib.
REQUIRED = ("helmet", "vest")

# Semua nama kelas PPE (pos+neg) — dipakai main.py untuk memisah deteksi.
ALL_PPE_CLASSES = {c for cat in PPE_CATEGORIES.values()
                   for grp in cat.values() for c in grp}

# Peta kelas -> (kategori, "pos"/"neg") untuk lookup cepat.
_CLASS_INDEX = {}
for _cat, _grp in PPE_CATEGORIES.items():
    for _c in _grp["pos"]:
        _CLASS_INDEX[_c] = (_cat, "pos")
    for _c in _grp["neg"]:
        _CLASS_INDEX[_c] = (_cat, "neg")


def iou(a, b):
    """Intersection-over-Union (dipakai di tempat lain / referensi)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def containment(small, big):
    """Fraksi kotak `small` (PPE) yang berada di dalam `big` (person).

    IoU tak cocok untuk asosiasi PPE<->person: helm/masker jauh lebih kecil dari
    kotak orang, sehingga IoU selalu rendah meski helm jelas DI kepala orang itu.
    Containment (irisan / luas PPE) mengukur "berapa banyak PPE ada di dalam
    orang" -> ~1.0 untuk helm di dalam kotak orang, berapa pun ukurannya.
    """
    x1, y1 = max(small[0], big[0]), max(small[1], big[1])
    x2, y2 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (small[2] - small[0]) * (small[3] - small[1])
    return inter / area if area > 0 else 0.0


def _severity(n_violations):
    """Priority alert (Extraordinary): makin banyak pelanggaran, makin tinggi."""
    if n_violations == 0:
        return "none"
    return "low" if n_violations == 1 else "high"


def assess_ppe(persons, ppe_items, thr=0.5, required=REQUIRED):
    """Tetapkan status kepatuhan PPE per orang.

    persons / ppe_items: list dict {bbox, conf, class}.
    thr: ambang CONTAINMENT asosiasi PPE<->person (default 0.5 = mayoritas kotak
         PPE ada di dalam orang). Lihat containment() untuk alasan vs IoU.
    required: kategori PPE yang wajib (pelanggaran -> non-compliant).

    Return list dict per orang:
      {bbox, status:{cat: ok/violation/unknown}, violations:[...],
       unverified:[...], compliant:bool, severity:none/low/high}
    """
    report = []
    for p in persons:
        # Bukti positif/negatif terkuat (by conf) per kategori untuk orang ini.
        best = {cat: {"pos": 0.0, "neg": 0.0} for cat in PPE_CATEGORIES}
        for item in ppe_items:
            idx = _CLASS_INDEX.get(item["class"])
            if idx is None:
                continue
            if containment(item["bbox"], p["bbox"]) <= thr:
                continue
            cat, polarity = idx
            conf = float(item.get("conf", 1.0))
            if conf > best[cat][polarity]:
                best[cat][polarity] = conf

        status = {}
        for cat in PPE_CATEGORIES:
            pos, neg = best[cat]["pos"], best[cat]["neg"]
            if pos == 0.0 and neg == 0.0:
                status[cat] = "unknown"        # tak ada bukti -> jangan asal vonis
            elif neg > pos:
                status[cat] = "violation"      # bukti negatif lebih kuat
            else:
                status[cat] = "ok"

        violations = [c for c in required if status[c] == "violation"]
        unverified = [c for c in required if status[c] == "unknown"]
        report.append({
            "bbox": p["bbox"],
            "status": status,
            "violations": violations,
            "unverified": unverified,            # wajib tapi tak terverifikasi
            "compliant": len(violations) == 0,   # = tak ada pelanggaran TERKONFIRMASI
            "severity": _severity(len(violations)),
        })
    return report


def summarize(report):
    """Agregat untuk priority alert tingkat-frame (Extraordinary)."""
    n = len(report)
    violators = [r for r in report if r["violations"]]
    return {
        "num_persons": n,
        "num_compliant": sum(1 for r in report if r["compliant"]),
        "num_violations": len(violators),
        "highest_severity": ("high" if any(r["severity"] == "high" for r in violators)
                             else "low" if violators else "none"),
    }
