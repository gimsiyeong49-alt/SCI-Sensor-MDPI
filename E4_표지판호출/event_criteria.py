#!/usr/bin/env python3
"""RGB에서 어떤 근거가 있을 때 '관측 시작 이벤트'를 낼지 조건을 재 본다.

새 점수식이 아니다. 29_방향대판독에서 확인한 '영상에서 보이는 근거'를 조건으로 쓴다.
  두드러짐 : 후보 자리가 같은 장면·같은 크기의 아무 벽 조각보다 두드러진다.
             밝기 평균 차이(평균 대비)와 테두리 경계 세기(경계 대비) 중 큰 쪽.
             테두리는 어둡고 안은 밝은 판은 평균만으로는 벽과 비슷하게 나오기 때문이다.
             기준은 정답이 아니라 아무 벽 조각의 분포(상위 10%)에서 잡는다.
  지속성   : 과거 1.2초 동안 같은 자리에서 관측된 횟수(support)
  크기     : 가로 px
정답은 채점에만 쓴다. 이 두 구간은 이미 여러 번 본 개발 구간이다.
"""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

R = Path("reports/_00_SIGN_PIPELINE_20260920")
D = Path("/data2/kci_datasets/4F_RGB_FULL_V0_1")
SEGS = [
    ("4F_148_178", R / "6_기하점수/proposals.jsonl", R / "1_정답/gt_marks_148_178_1s_objid.jsonl"),
    ("4F_228_258", R / "17_평가구간_228_258/prior/proposals.jsonl", R / "17_평가구간_228_258/gt_marks_228_258_1s_objid.jsonl"),
]


def ov(p, g):
    ix = max(0, min(p[2], g[2]) - max(p[0], g[0]))
    iy = max(0, min(p[3], g[3]) - max(p[1], g[1]))
    return ix * iy > 0 and ix * iy >= 0.3 * min((p[2] - p[0]) * (p[3] - p[1]), (g[2] - g[0]) * (g[3] - g[1]))


def salience(gray, grad, box, m=8):
    """평균 대비와 경계 대비 중 큰 쪽. 둘 다 둘레 벽의 흔들림으로 나눈다."""
    h, w = gray.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w - 1, x1), min(h - 1, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return 0.0
    a0, b0, a1, b1 = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m + 1), min(h, y1 + m + 1)
    ring = np.ones((b1 - b0, a1 - a0), bool)
    ring[y0 - b0:y1 - b0 + 1, x0 - a0:x1 - a0 + 1] = False
    rg, rgr = gray[b0:b1, a0:a1][ring].astype(np.float32), grad[b0:b1, a0:a1][ring]
    if rg.size < 10:
        return 0.0
    inner = gray[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    mean_c = abs(inner.mean() - rg.mean()) / (rg.std() + 3.0)
    border = np.concatenate([grad[y0, x0:x1 + 1], grad[y1, x0:x1 + 1], grad[y0:y1 + 1, x0], grad[y0:y1 + 1, x1]])
    edge_c = border.mean() / (rgr.mean() + 3.0)
    return float(max(mean_c, edge_c / 2.0))


def main():
    rng = np.random.default_rng(0)
    frames = []
    for name, pp, gp in SEGS:
        runs = {float(json.loads(l)["t"]): json.loads(l) for l in open(pp)}
        for l in open(gp):
            d = json.loads(l)
            t = float(d["t"])
            bgr = cv2.imread(str(D / runs[t]["frame"]))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            grad = np.hypot(gx, gy)
            recs = runs[t]["records"]
            sal = [salience(gray, grad, r["box"]) for r in recs]
            base = []                       # 같은 장면 아무 벽 조각의 두드러짐
            for r in rng.choice(len(recs), min(60, len(recs)), replace=False):
                b = recs[r]["box"]
                w_, h_ = b[2] - b[0], b[3] - b[1]
                x, y = rng.uniform(10, 630 - w_), rng.uniform(10, 320 - h_)
                base.append(salience(gray, grad, [x, y, x + w_, y + h_]))
            plaques = [(g, f"{name}#{o}") for (c, *g), o in zip(d["items"], d["object_id"]) if c == "plaque"]
            signs = [g for c, *g in d["items"] if c in ("plaque", "sign")]
            frames.append(dict(name=name, t=t, recs=recs, sal=np.array(sal), base=np.array(base),
                               plaques=plaques, signs=signs))
    thr = float(np.percentile(np.concatenate([f["base"] for f in frames]), 90))
    all_objs = {o for f in frames for _, o in f["plaques"]}
    print(f"두드러짐 기준(아무 벽 조각 상위 10%) = {thr:.2f}. 물리 번호판 {len(all_objs)}개, 장면 {len(frames)}장\n")
    print(f"{'관측≥':>5} {'가로≥':>5} | {'이벤트 걸린 번호판':>16} | {'장당 이벤트':>9} {'그중 표지':>8}")
    results = []
    for smin, wmin in itertools.product((0, 1, 2, 3), (4, 6, 8, 10)):
        hit_obj, ev, ev_ok = set(), 0, 0
        for f in frames:
            for r, s in zip(f["recs"], f["sal"]):
                b = r["box"]
                if s < thr or r.get("support", 0) < smin or b[2] - b[0] < wmin:
                    continue
                ev += 1
                ok = any(ov(b, g) for g in f["signs"])
                ev_ok += ok
                for g, o in f["plaques"]:
                    if ov(b, g):
                        hit_obj.add(o)
        results.append((smin, wmin, len(hit_obj), ev / len(frames), ev_ok / len(frames)))
        print(f"{smin:5d} {wmin:5d} | {len(hit_obj):6d}/{len(all_objs):<9d} | {ev / len(frames):9.1f} {ev_ok / len(frames):8.1f}")
    out = R / "30_관측이벤트"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"threshold": thr, "grid": results}, open(out / "grid.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
