"""Logika kepatuhan APD: asosiasi person <-> PPE via IoU matching.

Mengubah deteksi multi-class menjadi status compliant/violation per orang.
TODO (Tahap 5): tuning threshold IoU, tambah priority alert (Extraordinary).
"""

# Nama kelas dari data.yaml -> kategori PPE
HELMET_CLASSES = {"Hardhat", "helmet"}
VEST_CLASSES = {"Safety-Vest", "Safety Vest", "vest"}
MASK_CLASSES = {"Mask", "mask"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def assess_ppe(persons, ppe_items, thr=0.1):
    """Cocokkan tiap PPE box ke person box (IoU > thr).

    persons / ppe_items: list dict {bbox, conf, class}.
    -> list report per orang dengan status & violations.
    """
    report = []
    for p in persons:
        status = {"helmet": False, "vest": False, "mask": False}
        for item in ppe_items:
            if iou(p["bbox"], item["bbox"]) > thr:
                if item["class"] in HELMET_CLASSES:
                    status["helmet"] = True
                if item["class"] in VEST_CLASSES:
                    status["vest"] = True
                if item["class"] in MASK_CLASSES:
                    status["mask"] = True
        violations = [k for k, v in status.items() if not v]
        report.append({
            "bbox": p["bbox"],
            "status": status,
            "compliant": len(violations) == 0,
            "violations": violations,
        })
    return report
