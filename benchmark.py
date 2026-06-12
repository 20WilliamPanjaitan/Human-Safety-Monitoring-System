"""Benchmark latency endpoint — Tahap 6 / Must Have 5.

Ukur p50/p95 untuk /detect & /ppe (target < 500 ms p50, CPU). Request pertama
(warm-up) dibuang dari statistik.

  python benchmark.py                       # lokal, n=30
  python benchmark.py https://app... --n 20
"""
import argparse
import time

import numpy as np
import requests


def bench(base, ep, path, n):
    lat = []
    for i in range(n + 1):  # +1 warm-up
        t = time.time()
        with open(path, "rb") as fh:
            r = requests.post(f"{base}/{ep}", files={"file": fh}, timeout=120)
        r.raise_for_status()
        if i > 0:                       # buang warm-up
            lat.append((time.time() - t) * 1000)
    return np.percentile(lat, 50), np.percentile(lat, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    print(f"Benchmark {args.base}  (n={args.n}, warm-up dibuang)\n")
    print(f"{'endpoint':<10}{'p50 (ms)':>12}{'p95 (ms)':>12}{'target':>10}")
    for ep, path in [("detect", "samples/crowd.jpg"), ("ppe", "samples/worker.jpg")]:
        p50, p95 = bench(args.base, ep, path, args.n)
        ok = "OK" if p50 < 500 else "SLOW"
        print(f"{ep:<10}{p50:>12.0f}{p95:>12.0f}{'<500 ' + ok:>10}")


if __name__ == "__main__":
    main()
