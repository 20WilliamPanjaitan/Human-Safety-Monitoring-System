"""Smoke test semua endpoint — Tahap 6 / Must Have 5.

Lokal:  python smoke_test.py
Publik: python smoke_test.py https://YOUR_APP.onrender.com
"""
import sys

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

passed = True


def check(name, ok):
    global passed
    passed = passed and ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")


def main():
    try:
        check("health", requests.get(f"{BASE}/health", timeout=30).json().get("status") == "ok")
    except Exception as e:
        check(f"health ({e})", False)

    for ep, f in [("detect", "samples/crowd.jpg"),
                  ("ppe", "samples/worker.jpg"),
                  ("count", "samples/crowd.jpg")]:
        try:
            with open(f, "rb") as fh:
                r = requests.post(f"{BASE}/{ep}", files={"file": fh}, timeout=60)
            check(ep, r.status_code == 200)
        except FileNotFoundError:
            check(f"{ep} (sample '{f}' belum ada)", False)
        except Exception as e:
            check(f"{ep} ({e})", False)

    print("\nALL PASS" if passed else "\nADA YANG GAGAL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
