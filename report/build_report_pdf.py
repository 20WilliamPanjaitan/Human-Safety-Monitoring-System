"""Render Technical Report -> report/technical_report.pdf (reportlab).

Konten identik dengan report/technical_report.tex. Dipakai karena LaTeX tak
terpasang lokal; PDF ini siap-submit. Jalankan: python report/build_report_pdf.py
"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (HRFlowable, Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "technical_report.pdf")
CM_IMG = os.path.join(HERE, "confusion_matrix_normalized.png")

ACCENT = colors.HexColor("#b8860b")
INK = colors.HexColor("#1a1a1a")
MUT = colors.HexColor("#555555")
LINE = colors.HexColor("#cccccc")

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                      fontSize=8.6, leading=11.2, alignment=TA_JUSTIFY,
                      spaceAfter=4, textColor=INK)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=12,
                    textColor=ACCENT, spaceBefore=7, spaceAfter=3)
title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=15,
                       leading=17, textColor=INK, spaceAfter=2)
sub = ParagraphStyle("sub", fontName="Helvetica", fontSize=8.4, leading=10.5,
                     textColor=MUT, spaceAfter=2)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=7.4,
                     leading=9, textColor=MUT, spaceAfter=2)
small = ParagraphStyle("small", parent=body, fontSize=8.0, leading=10.2)


def P(t, st=body):
    return Paragraph(t, st)


story = []

# ── Header (tanpa cover) ──
story.append(P("Human Safety Monitoring System — Technical Report", title))
story.append(P("INaAI Competition 2026 &times; IT Del · AI Engineer Track · "
               "Domain 1: Human Safety Monitoring (Computer Vision)", sub))
story.append(P("William Panjaitan · "
               "github.com/20WilliamPanjaitan/Human-Safety-Monitoring-System · "
               "Juni 2026", sub))
story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT,
                        spaceBefore=3, spaceAfter=4))

# ── 1. Problem & Scope ──
story.append(P("1. Problem &amp; Scope", h2))
story.append(P(
    "Sistem monitoring keselamatan manusia berbasis kamera untuk area "
    "konstruksi. Diimplementasikan <b>4 dari 5</b> modul (minimum 3): "
    "<b>person detection</b>, <b>person tracking</b>, <b>PPE detection</b> "
    "(helmet/vest/mask), dan <b>people counting</b>. Face recognition sengaja "
    "tidak diimplementasi (privacy/PDP-by-design + batas waktu). Satu model "
    "<b>YOLOv8s multi-class (7 kelas)</b> menggerakkan semua modul, disajikan "
    "lewat <b>inference endpoint FastAPI</b> (ONNX Runtime untuk gambar, "
    "PyTorch untuk tracking) dan dashboard Streamlit. Target metrik mAP@0.5 "
    "&ge; 0.50 (go/no-go) &mdash; tercapai <b>0.78</b>."))

# ── 2. Data ──
story.append(P("2. Data", h2))
story.append(P(
    "Dataset <b>Construction Site Safety</b> (Roboflow Universe, lisensi "
    "<b>CC BY 4.0</b>). 10 kelas asli di-<i>remap</i> ke 7 kelas relevan "
    "(<i>Person, Hardhat, NO-Hardhat, Safety-Vest, NO-Safety-Vest, Mask, "
    "NO-Mask</i>) via <font face='Courier'>remap_labels.py</font>; kelas "
    "Safety-Cone/machinery/vehicle dibuang. Split memakai bawaan Roboflow "
    "<b>2605/114/82</b> (train/val/test). Kebocoran diaudit <b>2 lapis</b> "
    "(<font face='Courier'>check_leakage.py</font>): <b>0 duplikat byte-identik "
    "(MD5)</b> antar split; namun <b>19 scene video tersebar antar split</b> "
    "(frame dari klip sama ada di train &amp; test) &rarr; mAP test sedikit "
    "optimistis. Dilaporkan jujur sebagai limitasi, bukan di-<i>re-split</i> "
    "(yang akan memaksa re-training di window 22,5 jam). Ketujuh kelas muncul "
    "di split test; label diverifikasi visual "
    "(<font face='Courier'>sanity_check.py</font>)."))

# ── 3. Model & Training ──
story.append(P("3. Model &amp; Training", h2))
story.append(P(
    "<b>YOLOv8s</b> (Ultralytics 8.3.0), transfer-learning dari bobot COCO. "
    "Dilatih di Google Colab <b>Tesla T4</b>, <b>50 epoch</b>, imgsz 640, "
    "batch 16, <b>seed 42</b> (random/numpy/torch + cudnn.deterministic). "
    "Augmentasi: mosaic + close_mosaic=10, HSV jitter (h .015/s .7/v .4), "
    "scale .5, translate .1, fliplr .5, flipud 0. Bobot final "
    "<font face='Courier'>weights/best.pt</font> (MD5 2bdce55a&hellip;), "
    "diekspor ke <b>ONNX</b> untuk inference CPU yang lebih ringan; parity "
    "deteksi ONNX&harr;PT terverifikasi (5/5 identik kelas+conf@2dp). Provenans "
    "lengkap di <font face='Courier'>report/MODEL_PROVENANCE.md</font> (termasuk "
    "run lokal 5-epoch yang di-abort &amp; dibuang)."))

# ── 4. Evaluation ──
story.append(P("4. Evaluation", h2))
story.append(P(
    "Eval reproducible (<font face='Courier'>eval.py</font>, seed 42, hasil "
    "identik antar-run) pada split test:"))

ov = Table([
    ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"],
    ["0.778", "0.456", "0.883", "0.733"],
], colWidths=[3.1 * cm] * 4)
ov.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
    ("FONT", (0, 1), (-1, 1), "Helvetica", 9),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.4, LINE),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
]))
story.append(ov)
story.append(Spacer(1, 3))

pc = Table([
    ["Kelas", "Person", "Hardhat", "NO-Hardhat", "Safety-Vest",
     "NO-Safety-Vest", "Mask", "NO-Mask"],
    ["AP@0.5", "0.843", "0.879", "0.562", "0.857", "0.786", "0.761", "0.755"],
], colWidths=[1.9 * cm, 1.6 * cm, 1.6 * cm, 1.9 * cm, 1.9 * cm, 2.2 * cm,
              1.3 * cm, 1.5 * cm])
pc.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, -1), "Helvetica", 7.4),
    ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 7.4),
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.4),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0e6cc")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("GRID", (0, 0), (-1, -1), 0.4, LINE),
    ("TOPPADDING", (0, 0), (-1, -1), 2),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(pc)
story.append(Spacer(1, 3))
story.append(P(
    "<b>Kenapa mAP, bukan accuracy?</b> Deteksi tak punya \"accuracy\" "
    "tunggal: tiap gambar punya jumlah objek berbeda dan kualitas lokalisasi "
    "(IoU) penting. mAP mengintegrasikan kurva precision&ndash;recall lintas "
    "ambang confidence dan IoU per kelas &mdash; menangkap <i>missed "
    "detection</i> dan <i>false alarm</i> sekaligus. Accuracy "
    "(benar/total) tak terdefinisi untuk jumlah box yang variabel dan akan "
    "menutupi ketimpangan kelas (mis. kelas NO-* yang jarang). "
    "<b>Insight confusion matrix</b> (ternormalisasi): Hardhat&harr;NO-Hardhat "
    "nyaris tak pernah tertukar (silang 0.02) &mdash; saat model bilang "
    "\"no-helmet\", jarang itu false alarm; error dominan adalah "
    "<i>miss&rarr;background</i> (Person 0.35, NO-Hardhat 0.44), yaitu "
    "under-deteksi objek kecil/jarang, <b>bukan</b> salah klasifikasi "
    "safety-critical. Ini mode kegagalan yang diinginkan auditor: cenderung "
    "\"kelewat deteksi\", bukan \"keliru menyatakan patuh padahal melanggar\"."))

if os.path.exists(CM_IMG):
    img = Image(CM_IMG, width=6.4 * cm, height=6.4 * cm)
    img.hAlign = "CENTER"
    story.append(Spacer(1, 2))
    story.append(img)
    story.append(P("Gambar 1. Confusion matrix ternormalisasi (split test).", cap))

# ── 5. System Design ──
story.append(P("5. System Design", h2))
story.append(P(
    "Satu <i>core</i> bersama (<font face='Courier'>app/</font>) dipakai "
    "identik oleh REST API dan UI:", small))
story.append(P(
    "&bull; <b>Detector</b>: wrapper YOLOv8 (ONNX/PT), di-load sekali, "
    "<i>lazy</i> saat request pertama (startup ringan).<br/>"
    "&bull; <b>PPE logic</b>: asosiasi PPE&harr;person via <b>containment</b> "
    "(fraksi box PPE di dalam box orang), <b>bukan IoU</b> &mdash; IoU gagal "
    "untuk PPE kecil vs box orang yang tinggi. Vonis pakai <b>bukti positif "
    "kelas negatif</b> (NO-Hardhat/NO-Safety-Vest): status 3-nilai "
    "<i>ok/violation/unknown</i>. <i>violation</i> butuh box NO-* yang "
    "overlap; sekadar PPE tak terdeteksi &rarr; <i>unknown</i> (hindari "
    "pelanggaran palsu). Wajib = helmet+vest; mask dipantau, tak wajib. "
    "Multiple-violation + <b>severity</b> (none/low/high) + agregat "
    "<font face='Courier'>summarize()</font> per-frame (Extraordinary 1).<br/>"
    "&bull; <b>Tracking</b>: <b>ByteTrack</b> (Ultralytics), tuned "
    "(<font face='Courier'>track_buffer=60</font> utk okulasi singkat, "
    "<font face='Courier'>match_thresh=0.85</font> kurangi ID-switch); ID "
    "stabil antar-frame, jumlah ID unik = people counting versi video.<br/>"
    "&bull; <b>Counting</b>: per-gambar (jumlah box) &amp; per-video (track_id "
    "unik).", small))
story.append(P(
    "<b>API</b> (<font face='Courier'>FastAPI</font>): "
    "<font face='Courier'>/health /detect /ppe /count /track /annotate</font>, "
    "tiap respons memuat <font face='Courier'>latency_ms</font>, error handling "
    "400/415/500. <b>Latency</b> (CPU lokal): detect ~114 ms, ppe ~116 ms (p50, "
    "&lt;500 ms target); track klip 9,6 dtk &rarr; 8,8 dtk (vid_stride 2, imgsz "
    "480). <b>Visualisasi</b>: <font face='Courier'>/annotate</font> menggambar "
    "box berwarna-kepatuhan (hijau patuh / merah pelanggaran / kuning "
    "unverified) + label kelas&amp;conf; dashboard Streamlit menambah UI "
    "interaktif termasuk tab <i>Robustness Test</i>. <b>Deployment</b>: "
    "dikemas Docker (torch CPU-only) dan di-deploy ke <b>Railway</b> sebagai "
    "endpoint publik HTTPS (URL di README); "
    "<font face='Courier'>smoke_test.py</font> memvalidasi semua endpoint + "
    "error-case."))

# ── 6. AI Usage Log ──
story.append(P("6. AI Usage Log", h2))
story.append(P(
    "<b>Tools.</b> Claude Code (Anthropic, <b>Opus 4.8</b>) &mdash; asisten "
    "coding agentic; Ultralytics YOLOv8 <b>8.3.0</b> (train/inference/val); "
    "ByteTrack (built-in Ultralytics). <b>Tidak</b> ada synthetic data atau "
    "golden eval hasil LLM (semua metrik dari training/eval nyata), jadi aturan "
    "review 30% tak berlaku.", small))
story.append(P(
    "<b>Pola.</b> Agentic/multi-step: scaffolding, generate kode, debugging, "
    "refactor ke production-quality &mdash; tiap perubahan di-review &amp; "
    "diverifikasi manual. <b>Prompt utama Fase 2:</b> (i) bangun dashboard "
    "Streamlit modern di atas modul yang ada; (ii) jalankan uji edge-case "
    "Tahap 7 + generate aset/laporan; (iii) integrasikan edge-case sebagai tab "
    "Robustness di UI; (iv) uji tab di browser nyata; (v) analisis repo &rarr; "
    "dokumen struktur per-file; (vi) audit kelengkapan vs rubrik; (vii) "
    "siapkan deployment Docker (Render&rarr;Railway) &amp; perbaiki kegagalan "
    "ABI cv2/NumPy; (viii) tulis report ini. <b>Verifikasi &amp; judgment:</b> "
    "menolak/menyesuaikan saran AI &mdash; mendiagnosis mismatch "
    "opencv-headless 4.10 vs NumPy 1.x lalu pin 4.9.0.80; memperbaiki Streamlit "
    "<font face='Courier'>use_container_width</font>&rarr;"
    "<font face='Courier'>use_column_width</font> (v1.37). Semua kode "
    "dijalankan &amp; diuji (<font face='Courier'>ppe_demo.py</font> 8/8 unit, "
    "<font face='Courier'>smoke_test.py</font> semua endpoint PASS, eval "
    "reproducible). History prompt lengkap di "
    "<font face='Courier'>AI_USAGE_LOG.md</font>.", small))

# ── 7. Limitations ──
story.append(P("7. Limitations (disadari)", h2))
story.append(P(
    "&bull; <b>Scene leakage</b>: 19 scene video tersebar antar split &rarr; "
    "mAP test sedikit optimistis (diukur &amp; diungkap).<br/>"
    "&bull; <b>Kelas negatif jarang</b>: NO-Mask/NO-Hardhat recall terendah "
    "(instance sedikit, objek kecil) &mdash; diagonal confusion NO-Mask 0.39.<br/>"
    "&bull; <b>Asosiasi crowd</b>: 1 box PPE bisa di-claim &gt;1 orang yang "
    "overlap (simplifikasi diterima); NMS padat bisa under-count.<br/>"
    "&bull; <b>Tanpa ReID sejati</b>: ByteTrack berbasis gerak (track_buffer "
    "&asymp; 2,4 dtk @25 fps); ID bisa berubah setelah keluar-frame lama "
    "(&gt;~3 dtk) &mdash; ReID appearance-embedding belum ada.<br/>"
    "&bull; <b>Robustness sintetis</b>: low-light = brightness global &darr; "
    "(model terbukti robust, kemungkinan dari HSV-value aug); noise/blur malam "
    "nyata belum diuji. Far/small menurunkan deteksi PPE (&rarr; unverified, "
    "bukan pelanggaran palsu).<br/>"
    "&bull; <b>Tanpa face recognition</b>: sengaja di-descope demi "
    "privacy/PDP-by-design &amp; waktu; alerting berbasis identitas di luar "
    "scope.<br/>"
    "&bull; <b>Compute free-tier</b>: CPU-only; <font face='Courier'>/track</font> "
    "klip panjang berat memori (demo pakai klip pendek); ada cold-start pada "
    "request pertama.", small))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.5 * cm,
                        rightMargin=1.5 * cm, topMargin=1.3 * cm,
                        bottomMargin=1.2 * cm, title="Technical Report",
                        author="William Panjaitan")
doc.build(story)
print("PDF ->", OUT)
