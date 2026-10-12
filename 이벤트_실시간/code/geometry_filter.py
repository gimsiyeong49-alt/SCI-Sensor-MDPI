#!/usr/bin/env python3
"""후보에서 바닥·천장 쪽을 걷어낸다 (1단계 보강, 기하 단서).

이 녹화에는 깊이 영상이 없다(/camera/color/image_raw만 저장). 그래서 영상만으로
구할 수 있는 기하를 쓴다. 복도의 긴 선들이 모이는 소실점의 높이가 카메라 높이의
수평선이다. 카메라가 로봇 위 약 0.4 m에 있고 벽 부착물은 그보다 높이 붙으므로,
수평선보다 아래에 있는 후보는 바닥·반사광·걸레받이 쪽이다.

깊이나 FAST-LIO 상대 이동이 들어오면 여기에 벽면 거리·높이를 더해 더 좁힐 수 있다.
지금은 없으므로 쓰지 않는다. 소실점을 못 구하면 아무것도 지우지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def vanishing_point(bgr, min_lines=8):
    """복도 선들이 모이는 점 (x, y). 직진할 때 정지 물체가 퍼져 나오는 중심이기도 하다."""
    pts = _intersections(bgr, min_lines)
    if pts is None:
        return None
    return (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))


def _intersections(bgr, min_lines=8):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 70)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=12)
    if lines is None:
        return None
    left, right = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        sl = (y2 - y1) / (x2 - x1)
        if not (0.12 < abs(sl) < 3.0):
            continue
        (left if sl > 0 else right).append((x1, y1, sl))
    if len(left) < 2 or len(right) < 2 or len(left) + len(right) < min_lines:
        return None
    out = []
    for x1, y1, s1 in left:
        for x2, y2, s2 in right:
            if abs(s1 - s2) < 1e-3:
                continue
            x = ((y2 - s2 * x2) - (y1 - s1 * x1)) / (s1 - s2)
            y = y1 + s1 * (x - x1)
            if 0 <= x <= bgr.shape[1] and 0 <= y <= bgr.shape[0]:
                out.append((x, y))
    return out if len(out) >= 4 else None


def horizon_y(bgr, min_lines=8):
    """복도 선들의 교점에서 수평선 높이를 구한다. 못 구하면 None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 복도는 벽·바닥 대비가 낮아 문턱값을 낮게 잡아야 선이 잡힌다(40/120에서는 장당 2~6개뿐).
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 70)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=40, maxLineGap=12)
    if lines is None:
        return None
    left, right = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        s = (y2 - y1) / (x2 - x1)
        if not (0.12 < abs(s) < 3.0):      # 수평·수직선은 소실점을 안 만든다
            continue
        (left if s > 0 else right).append((x1, y1, s))   # 화면 왼쪽 벽은 아래로 기운다
    if len(left) < 2 or len(right) < 2 or len(left) + len(right) < min_lines:
        return None
    ys = []
    for x1, y1, s1 in left:
        for x2, y2, s2 in right:
            if abs(s1 - s2) < 1e-3:
                continue
            x = ((y2 - s2 * x2) - (y1 - s1 * x1)) / (s1 - s2)
            y = y1 + s1 * (x - x1)
            if 0 <= x <= bgr.shape[1] and 0 <= y <= bgr.shape[0]:
                ys.append(y)
    return float(np.median(ys)) if len(ys) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="proposals.jsonl 이 있는 폴더")
    ap.add_argument("--dataset", type=Path, default=Path("/data2/kci_datasets/4F_RGB_FULL_V0_1"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=15.0, help="수평선 아래로 이만큼까지는 남긴다")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "proposals.jsonl", "w") as f:
        for line in open(args.run_dir / "proposals.jsonl"):
            d = json.loads(line)
            bgr = cv2.imread(str(args.dataset / d["frame"]))
            hy = horizon_y(bgr)
            recs = d.get("records") or [{"box": b, "support": 0, "src": "frame"} for b in d["boxes"]]
            if hy is None:
                kept = recs
            else:
                kept = [r for r in recs if r["box"][3] <= hy + args.margin]
            d.update(horizon_y=hy, boxes=[r["box"] for r in kept], records=kept,
                     n_before=len(recs), n_after=len(kept))
            f.write(json.dumps(d) + "\n")
            print(f"{d['t']:6.1f}s 수평선 {('%.0f' % hy) if hy else '못구함':>6}  "
                  f"{len(recs)} → {len(kept)}", flush=True)


if __name__ == "__main__":
    main()
