#!/usr/bin/env python3
"""E2 과제 65개(13개 층 녹화 × 출발점 5개) 고르기(2026-10-09, 실험 설계용). 알고리즘에는 들어가지 않는다.
출발 시각 t_s 의 자리를 지침서 출발 유형으로 나눈다(채점용 정답 경로·의미지도·위치추정 장면 표시를 쓴다. 설계용이라 GT 사용 가능, 평가 파일과 분리).
  S4 계단·엘리베이터 앞: 정답 위치가 계단·엘리베이터 칸에서 4 m 안
  S2 꺾이는 곳: 출발 뒤 10초 동안 진행 방향이 60° 넘게 바뀜
  S3 갈림길: 위치추정 장면이 junction
  S5 막다른 곳: 장면이 dead-end
  S1 긴 직선 복도: 장면이 corridor 이고 a1_3_1 벽 방향 다양성 < 0.15
조건: 출발 뒤 60초 이상 데이터가 남아야 함(E3 60초까지). 드문 유형부터(S4→S2→S3→S5→S1) 고르고, 이미 고른 출발점과 시간 간격이
가장 큰 후보를 고른다. 그 유형이 없으면 남는 칸은 간격이 가장 큰 시각으로 채우고 실제 유형을 붙인다(missing 표시).
    python3 scripts/e2_select_starts.py
결과: _01_E2_초기위치/_작업자료/tasks/E2_tasks.csv, tasks_gt_EVAL_ONLY.csv(평가 전용), 출발점지도_<층>.png
"""
import csv, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import sys
sys.path.insert(0, str(Path(__file__).parent))
from obs_feedback import poly_mask  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); OUT = Path("reports/_01_E2_초기위치/_작업자료/tasks"); D = Path("/data/kci_work/maps_perfloor_20261001")
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
BLD = {**{f"{i}F": ("kci", i) for i in range(1, 10)}, "G1F": ("gong", 1), "G2F": ("gong", 2), "S1F": ("sw", 1), "S6F": ("sw", 6)}
ORDER = ["S4", "S2", "S3", "S5", "S1"]
NAME = {"S1": "긴 직선 복도", "S2": "꺾이는 곳", "S3": "갈림길", "S4": "계단·엘리베이터 앞", "S5": "막다른 곳"}
NEED_AFTER = 60.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks, gts = [], []
    for F in FL:
        S = json.load(open(R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]
        g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gt = np.array([float(x["relative_sec"]) for x in g]); gxy = np.array([[float(x["x_px"]), float(x["y_px"])] for x in g])
        span = gt[-1]
        im = cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE); H, W = im.shape
        sem = json.load(open(D / F / f"{F}_semantic_map.json"))
        fac = poly_mask((H, W), [s.get("polygon_px") for s in sem["spaces"] if s.get("type") in ("stair", "elevator")])
        DTf = distance_transform_edt(~fac) / ppm if fac.any() else np.full((H, W), 1e9)
        scene = {round(float(x["relative_sec"])): x["scene"] for x in csv.DictReader(open(R / "35_위치추정_축척추정" / F / "loc/timeline.csv"))}
        gc = list(csv.DictReader(open(R / "59_E4_호출정책/paper/csv" / f"geom_cells_{F}.csv")))
        gct = np.array([float(x["relative_sec"]) for x in gc]); gcd = np.array([float(x["normal_diversity"]) for x in gc])
        at = lambda t: gxy[int(np.argmin(abs(gt - t)))]
        def heading(t0, t1):
            p0, p1 = at(t0), at(t1); return math.atan2(p1[1] - p0[1], p1[0] - p0[0]) if np.hypot(*(p1 - p0)) / ppm > 0.5 else None
        cand = {}
        for t in np.arange(1.0, span - NEED_AFTER + 1e-6, 1.0):
            p = at(t); x, y = int(round(p[0])), int(round(p[1]))
            types = []
            if 0 <= x < W and 0 <= y < H and DTf[y, x] <= 4.0:
                types.append("S4")
            h0, h1 = heading(t, t + 3), heading(t + 7, t + 10)
            if h0 is not None and h1 is not None and abs(math.degrees(math.atan2(math.sin(h1 - h0), math.cos(h1 - h0)))) > 60:
                types.append("S2")
            sc = scene.get(int(round(max(t, 20.0))))
            if sc == "junction":
                types.append("S3")
            if sc == "dead-end":
                types.append("S5")
            m = (gct > t - 3) & (gct <= t)
            if sc == "corridor" and m.any() and np.median(gcd[m]) < 0.15:
                types.append("S1")
            cand[float(t)] = types
        ts_all = np.array(sorted(cand)); picked = []
        def far(ts):
            return max(ts, key=lambda t: min([abs(t - p) for p, _ in picked] or [1e9]))
        for ty in ORDER:
            c = [t for t in ts_all if ty in cand[t] and all(abs(t - p) >= 20 for p, _ in picked)]
            if c:
                picked.append((far(c), ty))
        miss = [ty for ty in ORDER if ty not in [p[1] for p in picked]]
        while len(picked) < 5:
            t = far([t for t in ts_all if all(abs(t - p) >= 10 for p, _ in picked)])
            picked.append((t, (cand[t][0] if cand[t] else "S0") + "*"))
        picked.sort()
        bld, m = BLD[F]
        for k, (t, ty) in enumerate(picked, 1):
            tid = f"{F}_T{k}"
            tasks.append(dict(task_id=tid, floor=F, building=bld, start_sec=t, start_type=ty.rstrip("*"), type_filled=int(ty.endswith("*")),
                              available_sec=round(span - t, 1), split="PILOT"))
            p = at(t); h = heading(t, t + 3)
            gts.append(dict(task_id=tid, floor=F, true_map=m, start_sec=t, x_px=round(p[0], 1), y_px=round(p[1], 1), ppm_gt=round(ppm, 3),
                            heading_px_rad="" if h is None else round(h, 4), gt_source="36_채점용정답(LiDAR 정합, 독립 GT 아님)"))
        # 그림
        img = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); img = (255 - (255 - img) * 0.5).astype(np.uint8)
        pts = gxy.astype(int); cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (170, 170, 170), 6)
        col = {"S1": (214, 120, 42), "S2": (52, 104, 235), "S3": (122, 175, 27), "S4": (0, 161, 237), "S5": (164, 123, 232), "S0": (90, 90, 90)}
        for k, (t, ty) in enumerate(picked, 1):
            p = at(t).astype(int); c = col.get(ty.rstrip("*"), (90, 90, 90))
            cv2.circle(img, tuple(p), 26, c, -1); cv2.circle(img, tuple(p), 26, (255, 255, 255), 4)
            cv2.putText(img, f"T{k} {ty}", (p[0] + 30, p[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
        ys, xs = np.where(im < 200); y0, y1, x0, x1 = max(0, ys.min() - 60), min(H, ys.max() + 60), max(0, xs.min() - 60), min(W, xs.max() + 60)
        crop = img[y0:y1, x0:x1]; s = 1600 / crop.shape[1]; crop = cv2.resize(crop, None, fx=s, fy=s)
        bar = np.full((60, crop.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{F}: " + ", ".join(f"T{k} {t:.0f}s {ty}" for k, (t, ty) in enumerate(picked, 1)) + (f"  (no {','.join(miss)})" if miss else ""),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.imwrite(str(OUT / f"출발점지도_{F}.png"), np.vstack([bar, crop]))
        print(F, f"{span:.0f}s", [(int(t), ty) for t, ty in picked], "없는 유형:", miss)
    for name, rows in (("E2_tasks.csv", tasks), ("tasks_gt_EVAL_ONLY.csv", gts)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print("과제", len(tasks), "유형", Counter(t["start_type"] for t in tasks), "채운 칸", sum(t["type_filled"] for t in tasks))


if __name__ == "__main__":
    main()
