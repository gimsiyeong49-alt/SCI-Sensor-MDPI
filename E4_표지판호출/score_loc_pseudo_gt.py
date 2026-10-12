#!/usr/bin/env python3
"""채점용 정답(scripts/make_pseudo_gt.py 의 gt.csv·summary.json)으로 위치추정 1초 기록을 채점한다(2026-10-05).
오차(m) = 지도 픽셀 거리 / 정답 축척(px / 오돔 m).

    python3 scripts/score_loc_pseudo_gt.py <정답 폴더> 이름=위치추정폴더 [이름=폴더 ...]
"""
import csv
import json
import os
import sys

import numpy as np


def main():
    G = sys.argv[1]
    # 정답 폴더에 warp_clean/gt_warp.csv(계단 빗금 제거 + 공간 보정, 2026-10-05)가 있으면 그것을 쓴다
    gfile = f"{G}/warp_clean/gt_warp.csv" if os.path.exists(f"{G}/warp_clean/gt_warp.csv") else f"{G}/gt.csv"
    s = json.load(open(f"{G}/summary.json"))
    ppm = s["ppm"]
    g = list(csv.DictReader(open(gfile)))
    print(f"[정답 {gfile.split('/')[-2] + '/' + gfile.split('/')[-1]}]")
    gt = np.array([float(r["relative_sec"]) for r in g])
    gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
    for arg in sys.argv[2:]:
        name, D = arg.split("=", 1)
        tk = {}
        for r in csv.DictReader(open(f"{D}/topk_series.csv")):
            tk.setdefault(float(r["relative_sec"]), []).append(r)
        lv = {float(r["relative_sec"]): r["confidence_level"] for r in csv.DictReader(open(f"{D}/timeline.csv"))}
        e1, ek, amb = [], [], []
        for t, c in tk.items():
            i = int(np.argmin(abs(gt - t)))
            if abs(gt[i] - t) > 0.5:
                continue
            c = sorted(c, key=lambda r: -float(r["score"]))
            d = [np.hypot(float(r["x_px"]) - gxy[i, 0], float(r["y_px"]) - gxy[i, 1]) / ppm for r in c]
            e1.append(d[0]); ek.append(min(d)); amb.append(lv.get(t) == "AMBIGUOUS")
        e1, ek, amb = map(np.array, (e1, ek, amb))
        conf = ~amb
        print(f"{name}: {len(e1)}초 | 1위 중앙 {np.median(e1):.1f} m, 3 m 안 {np.mean(e1 < 3) * 100:.0f}% | 상위12 안 {np.mean(ek < 3) * 100:.0f}% | "
              f"애매 {amb.mean() * 100:.0f}% | 확신 {conf.sum()}초 중 맞음 {np.mean(e1[conf] < 3) * 100 if conf.any() else float('nan'):.0f}% "
              f"(확신하며 틀림 = 전체의 {np.mean(conf & (e1 >= 3)) * 100:.0f}%) | 애매할 때 상위12 안 {np.mean(ek[amb] < 3) * 100 if amb.any() else float('nan'):.0f}%")


if __name__ == "__main__":
    main()
