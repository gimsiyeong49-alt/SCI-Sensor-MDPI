#!/usr/bin/env python3
"""지도 빈 곳 점검(2026-10-12): 13개 층 녹화본 전체의 실제 경로(채점용 정답)를 위치추정 지도에 얹어, 경로가 벽(문 마스크 뺀 것)을 지나는 자리를 찾는다.
점검용이다. 정답 경로로 지도를 직접 고치지 않는다(고치는 것은 모든 층에 같은 지도 생성 규칙으로).
  벽 = base_map 검정 → wall_clean.clean_wall(위치추정과 같은 가짜 벽 제거) − 문 마스크(3 px 키움, metric_fit 과 같음)
  자리 = 경로 위 연속 관통 구간. 길이 0.3 m 이상만(벽 관통 판정 기준). 같은 자리를 여러 번 지나면 1 m 안끼리 묶는다.
  분류: 계단 안(계단 마스크 0.3 m 안) / 계단 옆(2.5 m 안) / 그 밖(문 마스크 빠짐 후보)
출력: reports/_03_E3_걸으며좁히기/_작업자료/지도빈곳점검_20261012/{자리목록.csv, 요약.md, <층>.png}
    python3 scripts/map_hole_audit.py [--maps <floor_id/maps 폴더>]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, distance_transform_edt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402,F401  (글꼴)

ap = argparse.ArgumentParser()
ap.add_argument("--maps", default="/data/kci_work/collab_laptop_20261012/onboard_robot_20261001/floor_id/maps")
ap.add_argument("--out", default="reports/_03_E3_걸으며좁히기/_작업자료/지도빈곳점검_20261012")
ap.add_argument("--gt", default="warp", choices=["warp", "raw"], help="warp = 채점에 쓰는 공간 보정판(warp_clean/gt_warp.csv), raw = 보정 전 gt.csv")
a = ap.parse_args()
sys.path.insert(0, str(Path(a.maps).parent / "localizer"))
from wall_clean import clean_wall  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); GT = R / "36_채점용정답"; LOC = R / "35_위치추정_축척추정"
MAP = {f"{i}F": ("kci", f"{i}F") for i in range(1, 10)} | {"G1F": ("gong", "1F"), "G2F": ("gong", "2F"), "S1F": ("sw", "1F"), "S6F": ("sw", "6F")}
BN = {"kci": "정보관", "gong": "공학관", "sw": "소융대"}
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
rows, summ = [], []
for F, (b, mf) in MAP.items():
    M = Path(a.maps)
    base = np.array(Image.open(M / f"{b}_maps_perfloor/{mf}/maps/base_map.png").convert("L")) < 128
    wall, _ = clean_wall(base)
    door = binary_dilation(np.array(Image.open(M / f"{b}_maps_perfloor/{mf}/maps/door_mask.png").convert("L")) > 127, iterations=3)
    stair = np.array(Image.open(M / f"{b}_semmaps/{mf}/sem/4F_stair_mask.png").convert("L")) > 127
    dst = distance_transform_edt(~stair)
    ppm = float(list(csv.DictReader(open(LOC / F / "loc/timeline.csv")))[-1]["ppm_est"])
    gp = GT / F / ("warp_clean/gt_warp.csv" if a.gt == "warp" else "gt.csv")
    g = np.array([[float(r["x_px"]), float(r["y_px"])] for r in csv.DictReader(open(gp))])
    seg = np.diff(g, axis=0); n = np.maximum(np.ceil(np.hypot(*seg.T)).astype(int), 1)
    p = np.vstack([g[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))])
    H, W = wall.shape
    x, y = np.clip(np.rint(p[:, 0]).astype(int), 0, W - 1), np.clip(np.rint(p[:, 1]).astype(int), 0, H - 1)
    hit = wall[y, x] & ~door[y, x]
    idx = np.where(hit)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1) if len(idx) else []
    spots = []
    for r in runs:
        L = len(r) / ppm
        if L < 0.3:
            continue
        cx, cy = float(np.median(x[r])), float(np.median(y[r]))
        for s in spots:                           # 같은 자리를 다시 지나면 묶는다(1 m 안)
            if np.hypot(s["x"] - cx, s["y"] - cy) / ppm < 1.0:
                s["n"] += 1; s["len"] = max(s["len"], L); break
        else:
            spots.append(dict(x=cx, y=cy, n=1, len=L, dstair=float(dst[int(cy), int(cx)] / ppm)))
    for s in spots:
        s["kind"] = "계단 안" if s["dstair"] <= 0.3 else ("계단 옆" if s["dstair"] <= 2.5 else "그 밖(문 빠짐 후보)")
        rows.append(dict(층=F, 건물=BN[b], x_px=round(s["x"]), y_px=round(s["y"]), 관통길이_m=round(s["len"], 2), 지난횟수=s["n"],
                         계단까지_m=round(s["dstair"], 2), 분류=s["kind"]))
    k = {c: sum(s["kind"] == c for s in spots) for c in ("계단 안", "계단 옆", "그 밖(문 빠짐 후보)")}
    summ.append((F, BN[b], len(g) / 5 / 60, len(spots), k))
    fig, ax = plt.subplots(figsize=(16, 16 * H / W + 1.2))
    img = np.full((H, W, 3), 255, np.uint8); img[wall] = (130, 130, 130); img[stair] = (215, 195, 240); img[door] = (110, 175, 245)
    ax.imshow(img); ax.plot(g[:, 0], g[:, 1], color="#e74c3c", lw=0.6, alpha=0.5)
    col = {"계단 안": "#8e44ad", "계단 옆": "#e67e22", "그 밖(문 빠짐 후보)": "#c0392b"}
    for i, s in enumerate(spots):
        ax.add_patch(plt.Circle((s["x"], s["y"]), 1.2 * ppm, fill=False, ec=col[s["kind"]], lw=2.5))
        ax.text(s["x"], s["y"] - 1.5 * ppm, str(i + 1), color=col[s["kind"]], fontsize=11, ha="center", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{BN[b]} {F.lstrip('GS')} · 실제 경로가 벽을 0.3 m 넘게 지난 자리 {len(spots)}곳 (보라 = 계단 안, 주황 = 계단 옆, 빨강 = 그 밖)  "
                 f"· 하늘색 = 문 마스크, 연보라 = 계단 마스크", loc="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / f"{F}.png", dpi=60, bbox_inches="tight"); plt.close(fig)
    print(F, len(spots), k, flush=True)
with open(OUT / "자리목록.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
L = ["# 지도 빈 곳 점검 (2026-10-12)", "", "13개 층 녹화본 전체의 실제 경로(채점용 정답, " + ("공간 보정판" if a.gt == "warp" else "보정 전") + ")를 위치추정 지도에 얹어, 벽(가짜 벽 제거 뒤, 문 마스크 뺀 것)을 0.3 m 넘게 지난 자리를 셌다. "
     "점검용이며 이 결과로 지도를 직접 고치지 않는다. 채점용 정답 자세 자체의 오차(수십 cm)도 섞여 있다.", "",
     "| 층 | 녹화(분) | 자리 | 계단 안 | 계단 옆 | 그 밖(문 빠짐 후보) |", "|---|---|---|---|---|---|"]
L += [f"| {bn} {F} | {mins:.0f} | {n} | {k['계단 안']} | {k['계단 옆']} | {k['그 밖(문 빠짐 후보)']} |" for F, bn, mins, n, k in summ]
(OUT / "요약.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
