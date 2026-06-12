"""Streamlit dashboard — Human Safety Monitoring System (INaAI 2026).

UI modern & profesional di atas modul inti yang sama dengan FastAPI:
  Detector (YOLOv8/ONNX) · assess_ppe (kepatuhan APD) · track_summary (ByteTrack).

Jalankan:  streamlit run streamlit_app.py
Model di-load SEKALI via st.cache_resource (tidak reload tiap interaksi).
"""
import json
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st

from app.counting import count_image
from app.detector import Detector
from app.ppe_logic import ALL_PPE_CLASSES, assess_ppe, summarize
from app.tracker import track_summary
from edge_case_test import make_far_small, make_low_light

# ───────────────────────────── Konfigurasi ──────────────────────────────
WEIGHTS = os.getenv("WEIGHTS",
                    "weights/best.onnx" if os.path.exists("weights/best.onnx")
                    else "weights/best.pt")
TRACK_WEIGHTS = os.getenv("TRACK_WEIGHTS", "weights/best.pt")
PERSON_IDX = 0
PERSON_CLASSES = {"Person", "person"}

st.set_page_config(
    page_title="Human Safety Monitoring",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Styling (modern) ───────────────────────────
st.markdown("""
<style>
  :root{
    --bg:#0f1419; --panel:#1a2230; --panel2:#222c3d; --line:#2c3850;
    --txt:#e6edf3; --muted:#8b9bb4; --accent:#ffb703; --accent2:#fb8500;
    --ok:#22c55e; --bad:#ef4444; --info:#38bdf8;
  }
  .stApp{background:radial-gradient(1200px 600px at 75% -10%,#1b2a3f,transparent),var(--bg);}
  #MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0;}
  .block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1280px;}

  /* Header bar */
  .app-head{display:flex;align-items:center;gap:15px;padding:6px 2px 20px;
    border-bottom:1px solid var(--line);margin-bottom:22px;}
  .app-logo{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;
    background:linear-gradient(135deg,var(--accent),var(--accent2));font-size:24px;
    box-shadow:0 6px 18px rgba(251,133,0,.35);}
  .app-title{font-size:20px;font-weight:800;color:var(--txt);line-height:1.1;}
  .app-sub{font-size:12.5px;color:var(--muted);margin-top:3px;}
  .live{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12.5px;
    color:var(--muted);background:var(--panel);border:1px solid var(--line);
    padding:8px 14px;border-radius:999px;}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--ok);
    box-shadow:0 0 10px var(--ok);}

  /* Metric cards */
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:6px 0 20px;}
  .metric{background:var(--panel);border:1px solid var(--line);border-radius:15px;
    padding:16px 18px;}
  .metric .n{font-size:30px;font-weight:800;line-height:1;}
  .metric .l{color:var(--muted);font-size:11px;text-transform:uppercase;
    letter-spacing:.5px;margin-top:8px;}
  .metric.acc .n{color:var(--accent);} .metric.ok .n{color:var(--ok);}
  .metric.bad .n{color:var(--bad);} .metric.info .n{color:var(--info);}

  /* Section card */
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;
    padding:20px 22px;margin-bottom:18px;}
  .card-h{font-size:12px;text-transform:uppercase;letter-spacing:.6px;
    color:var(--muted);font-weight:700;margin-bottom:14px;}

  /* Pills */
  .pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:12px;
    font-weight:600;}
  .pill.ok{background:rgba(34,197,94,.16);color:#4ade80;}
  .pill.bad{background:rgba(239,68,68,.16);color:#f87171;}
  .pill.unk{background:rgba(139,155,180,.16);color:#aab6c9;}
  .pill.low{background:rgba(234,179,8,.16);color:#facc15;}
  .pill.high{background:rgba(239,68,68,.16);color:#f87171;}
  .pill.none{background:rgba(34,197,94,.16);color:#4ade80;}

  /* Table */
  table.tbl{width:100%;border-collapse:collapse;font-size:13.5px;}
  table.tbl th{text-align:left;color:var(--muted);font-weight:600;font-size:11px;
    text-transform:uppercase;letter-spacing:.4px;padding:10px 12px;
    border-bottom:1px solid var(--line);}
  table.tbl td{padding:11px 12px;border-bottom:1px solid var(--line);color:var(--txt);}
  table.tbl tr:last-child td{border-bottom:none;}

  .meta{color:var(--muted);font-size:12px;margin-top:12px;display:flex;gap:20px;
    flex-wrap:wrap;}

  /* Buttons */
  .stButton>button, .stDownloadButton>button{
    background:linear-gradient(135deg,var(--accent),var(--accent2));color:#1a1300;
    border:none;border-radius:10px;padding:9px 22px;font-weight:700;width:100%;}
  .stButton>button:hover{filter:brightness(1.07);color:#1a1300;}

  /* Sidebar */
  section[data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line);}
  .side-metric{display:flex;justify-content:space-between;align-items:center;
    padding:9px 0;border-bottom:1px solid var(--line);font-size:13px;color:var(--muted);}
  .side-metric b{color:var(--txt);font-size:14px;font-weight:700;}
  .bar-wrap{margin:7px 0 11px;}
  .bar-lbl{display:flex;justify-content:space-between;font-size:11.5px;
    color:var(--muted);margin-bottom:4px;}
  .bar-bg{height:7px;background:var(--panel2);border-radius:999px;overflow:hidden;}
  .bar-fg{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));
    border-radius:999px;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Model & helpers ────────────────────────────
@st.cache_resource(show_spinner="Memuat model deteksi…")
def get_detector():
    return Detector(WEIGHTS)


@st.cache_data
def load_eval():
    try:
        with open("eval_report.json") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def split_dets(dets):
    persons = [d for d in dets if d["class"] in PERSON_CLASSES]
    items = [d for d in dets if d["class"] in ALL_PPE_CLASSES]
    return persons, items


def annotate(img, report, items):
    """Gambar bbox APD (abu) + person berwarna by-kepatuhan. Return RGB."""
    img = img.copy()
    for it in items:
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(img, f"{it['class']} {it.get('conf', 0):.2f}",
                    (x1, max(0, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    for r in report:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        if r["violations"]:
            col, tag = (0, 0, 255), "VIOLATION: " + ",".join(r["violations"])
        elif r["unverified"]:
            col, tag = (0, 200, 255), "UNVERIFIED"
        else:
            col, tag = (0, 200, 0), "COMPLIANT"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def metric_cards(cards):
    html = '<div class="metrics">'
    for n, label, cls in cards:
        html += f'<div class="metric {cls}"><div class="n">{n}</div><div class="l">{label}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def pill(cls, txt):
    return f'<span class="pill {cls}">{txt}</span>'


# ─────────────────────────────── Sidebar ─────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-sub" style="font-weight:700;color:#e6edf3;'
                'font-size:15px;margin-bottom:2px;">⚙️ Pengaturan</div>',
                unsafe_allow_html=True)
    st.caption(f"Model aktif: `{os.path.basename(WEIGHTS)}`")

    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("IoU (NMS)", 0.30, 0.90, 0.45, 0.05)

    st.divider()
    st.markdown('<div class="card-h" style="margin-bottom:8px;">📊 Performa Model (test set)</div>',
                unsafe_allow_html=True)
    ev = load_eval()
    if ev:
        st.markdown(
            f'<div class="side-metric"><span>mAP@0.5</span><b>{ev["mAP@0.5"]:.3f}</b></div>'
            f'<div class="side-metric"><span>mAP@0.5:0.95</span><b>{ev["mAP@0.5:0.95"]:.3f}</b></div>'
            f'<div class="side-metric"><span>Precision</span><b>{ev["precision"]:.3f}</b></div>'
            f'<div class="side-metric"><span>Recall</span><b>{ev["recall"]:.3f}</b></div>',
            unsafe_allow_html=True)
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="card-h" style="margin-bottom:8px;">Per-class AP@0.5</div>',
                    unsafe_allow_html=True)
        for cls_name, ap in ev["per_class_AP@0.5"].items():
            st.markdown(
                f'<div class="bar-wrap"><div class="bar-lbl"><span>{cls_name}</span>'
                f'<span>{ap:.2f}</span></div><div class="bar-bg">'
                f'<div class="bar-fg" style="width:{ap*100:.0f}%"></div></div></div>',
                unsafe_allow_html=True)
    else:
        st.info("eval_report.json tidak ditemukan.")
    st.divider()
    st.caption("INaAI 2026 · YOLOv8s · ByteTrack")


# ─────────────────────────────── Header ──────────────────────────────────
st.markdown(f"""
<div class="app-head">
  <div class="app-logo">⛑️</div>
  <div>
    <div class="app-title">Human Safety Monitoring System</div>
    <div class="app-sub">Deteksi APD · People Counting · Person Tracking — Computer Vision</div>
  </div>
  <div class="live"><span class="dot"></span>model online · {os.path.basename(WEIGHTS)}</div>
</div>
""", unsafe_allow_html=True)

detector = get_detector()
detector.conf, detector.iou = conf, iou

# ─────────────────────────── Layout 2 kolom ──────────────────────────────
left, right = st.columns([5, 7], gap="large")

with left:
    st.markdown('<div class="card-h">① Input & Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode analisis", [
        "🦺 PPE Compliance (gambar)",
        "🎯 Deteksi person (gambar)",
        "🔢 Hitung orang (gambar)",
        "🧪 Robustness Test (gambar)",
        "🎬 Tracking (video)",
    ], label_visibility="collapsed")

    is_video = mode.startswith("🎬")
    upload = st.file_uploader(
        "Unggah file",
        type=["mp4", "mov", "avi"] if is_video else ["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if upload:
        if is_video:
            st.video(upload)
        else:
            st.image(upload, use_column_width=True)
   
    run = st.button("Analisis ▸", disabled=upload is None)


def read_image(file):
    arr = np.frombuffer(file.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


with right:
    st.markdown('<div class="card-h">② Hasil Analisis</div>', unsafe_allow_html=True)

    if not (run and upload):
        st.markdown(
            '<div class="card" style="text-align:center;padding:60px 20px;color:#8b9bb4;">'
            '<div style="font-size:46px;opacity:.45;">📊</div>'
            '<p style="margin-top:10px;">Unggah file di kiri lalu klik <b>Analisis</b>.</p></div>',
            unsafe_allow_html=True)

    # ── Video tracking ──
    elif is_video:
        with st.spinner("Memproses video (ByteTrack)… bisa beberapa detik."):
            suffix = os.path.splitext(upload.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(upload.getvalue())
            tmp.close()
            t0 = time.time()
            try:
                summary = track_summary(tmp.name, weights=TRACK_WEIGHTS,
                                        classes=[PERSON_IDX])
            finally:
                os.unlink(tmp.name)
            dt = time.time() - t0
        metric_cards([
            (summary["unique_ids"], "Orang Unik", "acc"),
            (summary["count_persons"], "People Count", "ok"),
            (summary["num_frames"], "Frame Diproses", "info"),
            (f"{dt:.1f}s", "Waktu Proses", ""),
        ])
        rows = "".join(
            f"<tr><td>ID {t['track_id']}</td><td>{t['class']}</td>"
            f"<td>{t['frames']} frame</td></tr>" for t in summary["tracks"])
        rows = rows or '<tr><td colspan="3" style="color:#8b9bb4;">Tidak ada track.</td></tr>'
        st.markdown(
            '<div class="card"><table class="tbl"><thead><tr><th>Track</th>'
            f'<th>Kelas</th><th>Kemunculan</th></tr></thead><tbody>{rows}</tbody>'
            '</table><div class="meta"><span>🎯 ByteTrack</span>'
            '<span>filter: Person</span></div></div>',
            unsafe_allow_html=True)

    # ── Image modes ──
    else:
        img = read_image(upload)
        if img is None:
            st.error("File gambar tidak valid / rusak.")
        else:
            t0 = time.time()
            dets = detector.detect(img)
            dt = (time.time() - t0) * 1000
            persons, items = split_dets(dets)

            if mode.startswith("🔢"):  # Count
                metric_cards([
                    (count_image(persons), "Orang Terdeteksi", "acc"),
                    (len(dets), "Total Deteksi", "info"),
                    (f"{dt:.0f} ms", "Latency", ""),
                    ("Person", "Filter Kelas", ""),
                ])
                st.image(annotate(img, [], []), use_column_width=True,
                         caption="Deteksi (tanpa anotasi kepatuhan)")

            elif mode.startswith("🎯"):  # Detect
                by_cls = {}
                for d in dets:
                    by_cls[d["class"]] = by_cls.get(d["class"], 0) + 1
                metric_cards([
                    (len(persons), "Person", "acc"),
                    (len(dets), "Total bbox", "info"),
                    (len(by_cls), "Kelas Unik", ""),
                    (f"{dt:.0f} ms", "Latency", ""),
                ])
                report = assess_ppe(persons, items)
                st.image(annotate(img, report, items), use_column_width=True)
                chips = " ".join(pill("unk", f"{k}: {v}") for k, v in by_cls.items())
                st.markdown(f'<div class="card">{chips}</div>', unsafe_allow_html=True)

            elif mode.startswith("🧪"):  # Robustness Test
                def case_metrics(im):
                    d = detector.detect(im)
                    ps, its = split_dets(d)
                    rep = assess_ppe(ps, its)
                    s = summarize(rep)
                    mc = (float(np.mean([p["conf"] for p in ps])) if ps else 0.0)
                    return annotate(im, rep, its), {
                        "person": s["num_persons"], "ppe": len(its),
                        "viol": s["num_violations"], "conf": round(mc, 3)}

                base_ann, base_m = case_metrics(img)
                low_ann, low_m = case_metrics(make_low_light(img))
                far_ann, far_m = case_metrics(make_far_small(img))
                variants = [
                    ("Baseline", base_ann, base_m),
                    ("Low-light 0.18×", low_ann, low_m),
                    ("Far/small 0.4×", far_ann, far_m),
                ]
                st.caption("Uji ketahanan: gambar yang sama diberi gangguan, "
                           "lalu dibandingkan terhadap baseline.")
                cols = st.columns(3)
                for col, (title, ann, m) in zip(cols, variants):
                    with col:
                        st.markdown(f'<div class="card-h" style="margin-bottom:6px">'
                                    f'{title}</div>', unsafe_allow_html=True)
                        st.image(ann, use_column_width=True)
                        d_person = m["person"] - base_m["person"]
                        d_conf = m["conf"] - base_m["conf"]
                        delta = "" if title == "Baseline" else (
                            f'<div class="meta" style="margin-top:6px">'
                            f'<span>Δperson {d_person:+d}</span>'
                            f'<span>Δconf {d_conf:+.3f}</span></div>')
                        st.markdown(
                            f'<div class="meta"><span>👤 {m["person"]}</span>'
                            f'<span>🦺 {m["ppe"]}</span>'
                            f'<span>⛔ {m["viol"]}</span>'
                            f'<span>conf {m["conf"]:.3f}</span></div>{delta}',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div class="card" style="margin-top:14px"><b>Catatan:</b> '
                    'saat PPE tak terdeteksi karena gangguan, status menjadi '
                    '<span class="pill unk">unknown/unverified</span>, '
                    '<b>bukan</b> pelanggaran palsu — model memvonis pelanggaran '
                    'hanya dari bukti positif kelas <code>NO-*</code>.</div>',
                    unsafe_allow_html=True)

            else:  # PPE Compliance
                report = assess_ppe(persons, items)
                summ = summarize(report)
                metric_cards([
                    (summ["num_persons"], "Total Orang", "acc"),
                    (summ["num_compliant"], "Patuh", "ok"),
                    (summ["num_violations"], "Pelanggaran", "bad"),
                    (summ["highest_severity"].upper(), "Severity Tertinggi", "info"),
                ])
                st.image(annotate(img, report, items), use_column_width=True)

                def st_pill(v):
                    return pill("ok" if v == "ok" else "bad" if v == "violation" else "unk", v)

                rows = ""
                for i, p in enumerate(report):
                    status = (pill("ok", "compliant") if p["compliant"]
                              else pill("bad", "violation"))
                    rows += (
                        f"<tr><td>#{i+1}</td><td>{st_pill(p['status']['helmet'])}</td>"
                        f"<td>{st_pill(p['status']['vest'])}</td>"
                        f"<td>{st_pill(p['status']['mask'])}</td>"
                        f"<td>{status}</td>"
                        f"<td>{pill(p['severity'], p['severity'])}</td></tr>")
                rows = rows or '<tr><td colspan="6" style="color:#8b9bb4;">Tidak ada orang terdeteksi.</td></tr>'
                st.markdown(
                    '<div class="card"><table class="tbl"><thead><tr><th>#</th>'
                    '<th>Helmet</th><th>Vest</th><th>Mask</th><th>Status</th>'
                    f'<th>Severity</th></tr></thead><tbody>{rows}</tbody></table>'
                    f'<div class="meta"><span>⏱ {dt:.0f} ms</span>'
                    '<span>🧠 ' + os.path.basename(WEIGHTS) + '</span>'
                    '<span>wajib: helmet + vest</span></div></div>',
                    unsafe_allow_html=True)
