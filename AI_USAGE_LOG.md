# AI Usage Log — Human Safety Monitoring System

> WAJIB submission. Catat tool AI, pola penggunaan, prompt penting, dan verifikasi.
> Isi seiring pengerjaan — jangan tulis di akhir saja.

## Tools yang digunakan
| Tool | Versi | Untuk apa |
|---|---|---|
| Claude (Claude Code) | Opus 4.8 | Scaffold repo, skeleton kode, roadmap, review |
| Ultralytics YOLOv8 | 8.3.0 | Training, inference, eval `val()` |
| ByteTrack | (built-in Ultralytics) | Tracking |
| | | |

## Pola penggunaan
- **Scaffolding & boilerplate:** struktur repo, Dockerfile, skeleton FastAPI/eval/train di-generate oleh AI lalu diverifikasi & disesuaikan manual.
- **Penjelasan teknis:** argumen mAP-vs-accuracy, cara kerja ByteTrack untuk Live Defense.
- (tambah sesuai pemakaian)

## History prompt penting
1. "Buatkan roadmap & tahapan teknis Fase 2 terstruktur dan terukur" → menghasilkan ROADMAP-FASE2.md.
2. "Generate scaffold T0 (struktur folder, requirements, Dockerfile, skeleton)" → repo ini.
3. (lanjutkan...)

## Verifikasi
- Semua kode hasil AI dijalankan & diuji sebelum dipakai (lihat smoke_test.py, eval.py).
- Metrik di report adalah hasil training sendiri (bukan angka dummy).
- (Jika pakai data sintetis: dokumentasikan verifikasi ≥30%.)
