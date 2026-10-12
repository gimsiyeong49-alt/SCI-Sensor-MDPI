#!/usr/bin/env python3
"""리얼센스 RGB에서 '벽에 붙은 작은 판'이 될 만한 조각을 빠짐없이 뽑는다 (1단계: 후보 생성).

여기서는 판정하지 않는다. 놓치지 않는 것(재현율)이 목적이고, 걸러 내는 일은 뒷단
(Qwen 조각 판정, 여러 장 누적, 고화질 재탐색)이 맡는다.

크기 기준은 정답이 아니라 물리 값에서 온다. 번호판 실물이 대략 10x25 cm이고
복도에서 2~8 m 거리에 있으므로, fx=606.8 화면에서 가로 7~30 px 정도가 된다.
여유를 둬 가로 4~50 px, 세로 6~90 px 를 받는다.

단서 세 가지를 쓴다. 한 가지 색이나 한 가지 모양에 기대지 않는다.
  1) MSER: 주변과 밝기가 다른 덩어리. 밝은 판, 어두운 판 둘 다 보려고 원본과 반전 모두 본다.
  2) 형태학 top-hat / black-hat: 벽면보다 밝거나 어두운 작은 조각.
  3) 채도 대비: 벽(베이지)과 다른 색을 띤 조각. 파랑만 찾지 않도록 색상은 안 가린다.
겹치는 후보는 합친다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

MIN_W, MAX_W = 4, 50
MIN_H, MAX_H = 6, 90
MAX_AREA_FRAC = 0.02     # 화면의 2%보다 큰 덩어리는 판이 아니다
ASPECT = (0.25, 6.0)     # 세로/가로


def _boxes_from_mask(mask, min_fill=0.25):
    out = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H):
            continue
        if not (ASPECT[0] <= h / w <= ASPECT[1]):
            continue
        if area < min_fill * w * h:
            continue
        out.append((x, y, x + w, y + h))
    return out


def propose(bgr, scales=(1, 2), debug=None):
    """원본과 2배 확대에서 각각 뽑는다. 가로 6~12 px짜리는 원본 해상도에서 덩어리로 안 잡힌다."""
    boxes = []
    for s in scales:
        img = bgr if s == 1 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        boxes += [(b[0] / s, b[1] / s, b[2] / s, b[3] / s) for b in _propose_one(img)]
    boxes = merge(boxes)
    boxes = boxes + group(boxes)   # 쪼개진 파편을 묶은 것도 후보로 같이 낸다
    boxes = merge(boxes, thr=0.85)
    if debug is not None:
        debug.update(n_raw=len(boxes))
    return boxes


def _propose_one(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    boxes = []

    # 1) MSER (밝은 쪽 / 어두운 쪽)
    mser = cv2.MSER_create(delta=4, min_area=MIN_W * MIN_H, max_area=int(MAX_AREA_FRAC * w * h))
    for img in (gray, 255 - gray):
        for pts in mser.detectRegions(img)[0]:
            x, y, bw, bh = cv2.boundingRect(pts.reshape(-1, 1, 2))
            if MIN_W <= bw <= MAX_W and MIN_H <= bh <= MAX_H and ASPECT[0] <= bh / bw <= ASPECT[1]:
                boxes.append((x, y, x + bw, y + bh))

    # 2) top-hat / black-hat: 벽면 밝기에서 튀어나온 작은 조각
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    for op in (cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT):
        r = cv2.morphologyEx(gray, op, k)
        thr = max(8, int(np.percentile(r, 99.0)))
        boxes += _boxes_from_mask(r >= thr)

    # 3) 채도 대비: 주변보다 색이 진한 조각 (색상은 가리지 않는다)
    sat = hsv[:, :, 1].astype(np.int16)
    local = cv2.blur(sat.astype(np.float32), (41, 41))
    boxes += _boxes_from_mask((sat - local) >= 18)

    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    if not ix or not iy:
        return 0.0
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def merge(boxes, thr=0.3):
    out = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        hit = next((i for i, o in enumerate(out) if iou(o, b) > thr), None)
        if hit is None:
            out.append(list(b))
        else:  # 합치기
            o = out[hit]
            out[hit] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
    return [tuple(int(v) for v in b) for b in out]


def group(boxes, pad=3):
    """가까이 붙은 조각을 하나로 묶는다. 번호판 하나가 글자·테두리로 쪼개져 나오기 때문이다."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            b = boxes[j]
            if (a[0] - pad <= b[2] and b[0] - pad <= a[2]
                    and a[1] - pad <= b[3] and b[1] - pad <= a[3]):
                parent[find(i)] = find(j)
    g = {}
    for i, b in enumerate(boxes):
        r = find(i)
        o = g.get(r)
        g[r] = b if o is None else (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
    out = [b for b in g.values()
           if MIN_W <= b[2] - b[0] <= MAX_W and MIN_H <= b[3] - b[1] <= MAX_H]
    return [tuple(int(v) for v in b) for b in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-csv", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=Path("/data2/kci_datasets/4F_RGB_FULL_V0_1"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.out / "proposals.jsonl", "w") as f:
        for row in csv.reader(open(args.frames_csv)):
            t, rel = float(row[0]), row[1]
            bgr = cv2.imread(str(args.dataset / rel))
            boxes = propose(bgr)
            f.write(json.dumps({"t": t, "frame": rel, "boxes": boxes}) + "\n")
            print(f"{t:6.1f}s 후보 {len(boxes)}", flush=True)


if __name__ == "__main__":
    main()
