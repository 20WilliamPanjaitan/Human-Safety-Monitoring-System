"""Smoke test semua endpoint — Tahap 6 / Must Have 5.

Lokal:  python smoke_test.py
Publik: python smoke_test.py https://YOUR_APP.onrender.com

Membuktikan tiap modul yang di-claim punya endpoint hidup (status 200) + error
handling benar (file rusak -> 400, bukan 500).
"""
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

passed = True


def check(name, ok, extra=""):
    global passed
    passed = passed and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")


def post_file(ep, path, field="file"):
    with open(path, "rb") as fh:
        return requests.post(f"{BASE}/{ep}", files={field: fh}, timeout=120)


def main():
    # health
    try:
        check("health", requests.get(f"{BASE}/health", timeout=30).json().get("status") == "ok")
    except Exception as e:
        check(f"health ({e})", False)

    # endpoint gambar
    for ep, f in [("detect", "samples/crowd.jpg"),
                  ("ppe", "samples/worker.jpg"),
                  ("count", "samples/crowd.jpg"),
                  ("annotate", "samples/worker.jpg")]:
        try:
            r = post_file(ep, f)
            lat = r.json().get("latency_ms") if "json" in r.headers.get("content-type", "") else ""
            check(ep, r.status_code == 200, f"{r.status_code}, {lat}ms" if lat else str(r.status_code))
        except FileNotFoundError:
            check(f"{ep} (sample '{f}' belum ada)", False)
        except Exception as e:
            check(f"{ep} ({e})", False)

    # endpoint video (tracking)
    try:
        r = post_file("track", "samples/clip.mp4")
        j = r.json() if r.status_code == 200 else {}
        check("track", r.status_code == 200,
              f"{j.get('num_frames')} frames, {j.get('unique_ids')} ID")
    except FileNotFoundError:
        check("track (sample 'samples/clip.mp4' belum ada)", False)
    except Exception as e:
        check(f"track ({e})", False)

    # error handling: file rusak -> 400 (BUKAN 500)
    try:
        r = requests.post(f"{BASE}/detect",
                          files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
                          timeout=30)
        check("error-case (file rusak -> 400)", r.status_code == 400, str(r.status_code))
    except Exception as e:
        check(f"error-case ({e})", False)

    print("\nALL PASS" if passed else "\nADA YANG GAGAL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
