#!/usr/bin/env python3
"""E4 센서 이상 주입 시험(2026-10-08, 실험 지침서 14.1·14.6 '센서 오류를 장소 모호성으로 혼동한 빈도', PILOT).
원본 a1_3_1 폴더는 건드리지 않고, 새 폴더에 samples·far_points 는 심볼릭 링크, odom_link.csv 만 튐을 넣은 사본을 만든다.
튐 = FAST-LIO 가 순간적으로 자세를 잘못 잡아 그 뒤 궤적 전체가 옮겨지는 계단형 불연속:
  t ≥ t_j 인 모든 자세에 p' = R(dθ)(p − p_j) + p_j + d, yaw' = yaw + dθ (d 는 그 순간 진행 방향 기준).
    python3 scripts/e4_fault_inject.py --src /data/kci_work/a131_full_v0_4 --dst /data/kci_work/e4_fault_inject_20261008/4F
"""
import argparse, csv, json, math, os
from pathlib import Path
import numpy as np

PLAN = [(60, 0.5, 0), (100, 1.0, 0), (140, 2.0, 0), (180, 3.0, 0), (220, 0, 10), (260, 0, 20), (300, 0, 45), (340, 2.0, 20)]  # (상대초, 이동 m, 회전 °)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=False)
    for p in src.iterdir():
        if p.name != "odom_link.csv":
            os.symlink(p.resolve(), dst / p.name)
    rows = list(csv.DictReader(open(src / "odom_link.csv"))); T = np.array([float(r["t_sec"]) for r in rows])
    X = np.array([float(r["fastlio_x"]) for r in rows]); Y = np.array([float(r["fastlio_y"]) for r in rows]); W = np.array([float(r["fastlio_yaw_rad"]) for r in rows])
    log = []
    for rel, dm, dd in PLAN:
        j = int(np.searchsorted(T, T[0] + rel)); th = math.radians(dd)
        k = max(j - 5, 0); hd = math.atan2(Y[j] - Y[k], X[j] - X[k]) if j > k else W[j]   # 그 순간 진행 방향
        c, s = math.cos(th), math.sin(th); px, py = X[j], Y[j]
        dx, dy = X[j:] - px, Y[j:] - py
        X[j:] = px + c * dx - s * dy + dm * math.cos(hd); Y[j:] = py + s * dx + c * dy + dm * math.sin(hd); W[j:] = W[j:] + th
        log.append(dict(rel_sec=rel, t_sec=float(T[j]), sample_id=rows[j]["sample_id"], jump_m=dm, jump_deg=dd))
    with open(dst / "odom_link.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r, x, y, yw in zip(rows, X, Y, W):
            r = dict(r); r["fastlio_x"] = f"{x:.4f}"; r["fastlio_y"] = f"{y:.4f}"; r["fastlio_yaw_rad"] = f"{math.atan2(math.sin(yw), math.cos(yw)):.5f}"
            w.writerow(r)
    json.dump(dict(src=str(src), plan=log, note="원본 폴더 불변. odom_link.csv 만 사본에 튐 주입"), open(dst / "INJECTION.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False))


if __name__ == "__main__":
    main()
