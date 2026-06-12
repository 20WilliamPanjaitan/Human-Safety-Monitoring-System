# Edge Case Test — Tahap 7 (Visualisasi & Ketahanan)

Hasil uji ketahanan model PPE pada kondisi sulit + aset demo teranotasi. Direproduksi dengan `python edge_case_test.py`.

- Gambar subjek : `samples/worker.jpg`
- Gambar crowd  : `samples/crowd.jpg`
- Aset teranotasi: `samples/edge_cases/`

## Ringkasan Metrik

| Kondisi | Person | PPE box | Pelanggaran | Severity | Mean conf person |
|---|---|---|---|---|---|
| baseline (normal) | 2 | 3 | 1 | low | 0.597 |
| low_light (brightness 0.18x) | 2 | 3 | 1 | low | 0.578 |
| far_small (subjek 0.4x) | 2 | 2 | 1 | high | 0.748 |
| overlap / crowd | 7 | 5 | 0 | none | 0.539 |

## Observasi & Mitigasi per Edge Case

### low_light
- **Aset:** `samples/edge_cases/low_light.png`
- **Terukur:** person=2, PPE=3, pelanggaran=1, mean_conf=0.578
- **Observasi:** Brightness diturunkan drastis (0.18x) namun jumlah person & PPE TETAP, mean confidence person hanya turun tipis (~0.02). Model robust terhadap penurunan brightness global — kemungkinan besar berkat augmentasi HSV-value saat training. Bukan klaim ketahanan universal: degradasi nyata diperkirakan pada noise/motion-blur malam yang tak disimulasikan di sini.
- **Mitigasi:** Untuk kondisi malam ekstrem: CLAHE/histogram-equalization pra-proses, kamera IR/low-light, dan uji dengan footage malam nyata (bukan hanya brightness sintetis).

### far_small
- **Aset:** `samples/edge_cases/far_small.png`
- **Terukur:** person=2, PPE=2, pelanggaran=1, mean_conf=0.748
- **Observasi:** Subjek dikecilkan ke 0.4x: bbox person masih tertangkap (2/2) namun deteksi PPE turun (3 -> 2 box) — PPE kecil mulai miss. Mean conf person yang tersisa justru naik (hanya deteksi kuat yang lolos). Saat PPE miss, status jadi `unverified`, BUKAN false-violation.
- **Mitigasi:** Inference imgsz lebih besar (1280); tiling/SAHI untuk objek kecil; kamera lebih dekat ke zona kerja.

### overlap_crowd
- **Aset:** `samples/edge_cases/overlap_crowd.png`
- **Terukur:** person=7, PPE=5, pelanggaran=0, mean_conf=0.539
- **Observasi:** Kerumunan terdeteksi (7 person) tanpa false-violation. Risiko yang diketahui: NMS dapat menggabung/menghapus person yang sangat berdempetan (potensi under-count), dan 1 box PPE bisa ter-claim >1 orang (simplifikasi yang diterima untuk demo).
- **Mitigasi:** Naikkan IoU/NMS threshold di kerumunan; asosiasi PPE 1-ke-1 (Hungarian) bila butuh presisi; gunakan tracking untuk dedup antar frame.

## Catatan Desain (kenapa tak ada false-violation)

Logika kepatuhan (`app/ppe_logic.py`) memvonis `violation` hanya bila ada **bukti positif** kelas negatif (`NO-Hardhat`/`NO-Safety-Vest`) yang overlap. Saat PPE sekadar miss-detect (low light / jauh), status menjadi `unknown`/`unverified`, **bukan** pelanggaran palsu — keputusan desain yang membuat sistem konservatif & aman untuk audit.
