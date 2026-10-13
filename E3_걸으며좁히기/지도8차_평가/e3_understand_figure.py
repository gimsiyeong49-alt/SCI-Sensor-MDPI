#!/usr/bin/env python3
"""E3 합치기 결과 이해용 그림(2026-10-12): 60초 때 65개 출발점 표(층 × 장소 종류, 색 = 1위 정답/후보 안/없음, 숫자 = 정답 순위) + 실제 예시 3곳 지도.
    python3 scripts/e3_understand_figure.py [--T 60]
출력: reports/_03_E3_걸으며좁히기/3_결과표와_그림/이해용_1위와_후보안정답_<T>초.png
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=60); ap.add_argument("--dir", default=""); ap.add_argument("--out", default=""); a = ap.parse_args(); T = a.T
W = Path(sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else Path("reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색")
EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(EV.G / "tasks/E2_tasks.csv"))}
GTT = {r["task_id"]: r for r in csv.DictReader(open(EV.G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
S = {r["trial_id"]: r for r in csv.DictReader(open(W / "연결판_채점.csv")) if int(r["T"]) == T}
C = {}
for r in csv.DictReader(open(W / "연결판_남은후보.csv")):
    if int(r["T"]) == T:
        C.setdefault(r["trial_id"], []).append(r)
GREEN, AMBER, GRAYC, INK, INK2 = "#2E8B57", "#E8A33D", "#C9CED6", "#1F2328", "#57606A"
FLOORS = [f"{i}F" for i in range(1, 10)] + ["G1F", "G2F", "S1F", "S6F"]
FNAME = {f: (f"정보관 {f[:-1]}층" if f[0].isdigit() else ("공학관 " if f[0] == "G" else "소융대 ") + f[1:-1] + "층") for f in FLOORS}
TYPES = ["S1", "S2", "S3", "S4", "S5"]
SHORT = {"S1": "직선복도", "S2": "꺾임", "S3": "갈림길", "S4": "계단·엘베", "S5": "막다른곳"}
cat = {}
for k, r in S.items():
    cat[k] = "g" if r["top1_ok"] == "1" else ("a" if r["true_rank"] else "x")   # 후보 안 = 남은 후보 전체(README 와 같은 기준)
fig = plt.figure(figsize=(22, 14)); gs = fig.add_gridspec(3, 3, width_ratios=[1.05, 1, 1], hspace=0.55, wspace=0.12, top=0.86)
ax = fig.add_subplot(gs[:, 0])
for i, F in enumerate(FLOORS):
    for j in range(5):
        k = f"{F}_T{j + 1}"; ty = EV.TASK[k]["start_type"]
        c = cat.get(k, "x"); col = {"g": GREEN, "a": AMBER, "x": GRAYC}[c]
        ax.add_patch(plt.Rectangle((j, -i), 0.94, 0.9, color=col))
        rk = S[k]["true_rank"] if k in S else ""
        txt = "1위" if c == "g" else (f"{rk}위" if c == "a" else "없음")
        ax.text(j + 0.47, -i + 0.55, txt, ha="center", va="center", fontsize=12, fontweight="bold", color="white" if c == "g" else INK)
        ax.text(j + 0.47, -i + 0.18, SHORT[ty], ha="center", va="center", fontsize=8, color="white" if c == "g" else INK2)
    ax.text(-0.08, -i + 0.45, FNAME[F], ha="right", va="center", fontsize=12, color=INK)
for j in range(5):
    ax.text(j + 0.47, 1.15, f"출발점 {j + 1}", ha="center", va="bottom", fontsize=11.5, color=INK)
ax.set_xlim(-2.2, 5.1); ax.set_ylim(-len(FLOORS) + 0.7, 2.2); ax.axis("off")
ng, na, nx = (sum(v == c for v in cat.values()) for c in "gax")
ax.set_title(f"{T}초 걸었을 때 65개 출발점(칸 하나 = 출발점 하나)\n초록 1위가 정답 {ng}곳({ng/65*100:.0f}%)\n주황 정답이 후보 안에 있음(숫자 = 정답 순위) {na}곳\n회색 정답이 후보에 없음 {nx}곳  →  후보 안에 정답 {ng+na}곳({(ng+na)/65*100:.0f}%)",
             loc="left", fontsize=13.5, fontweight="bold", color=INK)
# 예시 3곳: 초록(정보관), 주황(1위가 다른 층, 정답 순위 2~10), 회색
def pick(c, extra=lambda k: True):
    for k in sorted(cat, key=lambda k: (FLOORS.index(EV.TASK[k]["floor"]), k)):
        if cat[k] == c and extra(k):
            return k
def top1(k):
    return min(C[k], key=lambda r: int(r["rank"]))
ex = [("g", pick("g", lambda k: EV.TASK[k]["building"] == "kci")),
      ("a", pick("a", lambda k: S[k]["true_rank"] and 2 <= int(S[k]["true_rank"]) <= 10 and int(top1(k)["floor_id"]) != int(GTT[k]["true_map"]))),
      ("x", pick("x", lambda k: EV.TASK[k]["building"] == "kci"))]
def crop(b, m, pts, pad=260):
    im = cv2.imread(str(EV.MAPS[b] / f"{m}F" / "maps" / "base_map.png"), cv2.IMREAD_GRAYSCALE)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = int(max(min(xs) - pad, 0)), int(min(max(xs) + pad, im.shape[1])); y0, y1 = int(max(min(ys) - pad, 0)), int(min(max(ys) + pad, im.shape[0]))
    return im[y0:y1, x0:x1], (x0, x1, y1, y0)
for row, (c, k) in enumerate(ex):
    t = EV.TASK[k]; b = t["building"]; m_true = int(GTT[k]["true_map"]); ts = float(t["start_sec"])
    _, ppm, _, gtime, gxy = EV.gt_of(k); i0, i1 = np.argmin(abs(gtime - ts)), np.argmin(abs(gtime - ts - T)); path = gxy[i0:i1 + 1]; truth = gxy[i1]
    t1 = top1(k); m1 = int(t1["floor_id"]); p1 = (float(t1["x_px"]), float(t1["y_px"]))
    head = {"g": "1위 후보가 바로 정답", "a": f"정답은 후보 {S[k]['true_rank']}위, 1위는 다른 층", "x": "정답이 후보에 없음"}[c]
    col = {"g": GREEN, "a": AMBER, "x": INK2}[c]
    axl = fig.add_subplot(gs[row, 1]); axr = fig.add_subplot(gs[row, 2])
    im, ext = crop(b, m_true, [truth, *path[::10]]); axl.imshow(im, cmap="gray", extent=ext)
    axl.plot(path[:, 0], path[:, 1], "-", color="#3B82F6", lw=2, label="실제로 걸은 길"); axl.plot(*truth, "*", color="#D62728", ms=22, label="정답 위치")
    cand_true = [r for r in C[k] if int(r["floor_id"]) == m_true]
    if cand_true:
        best = min(cand_true, key=lambda r: int(r["rank"])); bx, by = float(best["x_px"]), float(best["y_px"]); dm = np.hypot(bx - truth[0], by - truth[1]) / ppm
        axl.plot(bx, by, "o", mfc="none", mec=col, mew=3, ms=20)
        lab = f"정답 층 최고 후보 {best['rank']}위 (정답에서 {dm:.1f} m" + (", 3 m·30° 밖이라 오답)" if c == "x" else ")")
        axl.text(bx + 30, by - 40, lab, color=INK, fontsize=10.5, fontweight="bold", bbox=dict(fc="white", ec=col, alpha=0.9))
    axl.set_title(f"{EV.BNAME[b]} {t['floor'].lstrip('GS').replace('F','층')} 출발점{k.split('_T')[1]}({EV.TYPE[t['start_type']]}) · 정답 층 지도", loc="left", fontsize=11.5)
    axl.set_xlim(ext[0], ext[1]); axl.set_ylim(ext[2], ext[3]); axl.set_xticks([]); axl.set_yticks([])
    if row == 0:
        axl.legend(loc="lower left", fontsize=9.5)
    im, ext = crop(b, m1, [p1]); axr.imshow(im, cmap="gray", extent=ext)
    axr.plot(*p1, "o", color=col if c != "x" else INK2, ms=14); axr.text(p1[0] + 30, p1[1] - 40, "1위 후보", fontsize=11, fontweight="bold", color=INK, bbox=dict(fc="white", ec=INK2, alpha=0.9))
    if m1 == m_true:
        axr.plot(*truth, "*", color="#D62728", ms=22)
    fl1 = (f"{m1}층" if b == "kci" else (("G" if b == "gong" else "S") + f"{m1}F"))
    axr.set_title(f"1위 후보가 고른 지도: {fl1}{' (정답 층)' if m1 == m_true else ' (틀린 층)'}", loc="left", fontsize=11.5, color=INK)
    axr.set_xlim(ext[0], ext[1]); axr.set_ylim(ext[2], ext[3]); axr.set_xticks([]); axr.set_yticks([])
    fig.text(axl.get_position().x0, axl.get_position().y1 + 0.035, f"예시 {row + 1} · {head}", fontsize=14.5, fontweight="bold", color=col)
fig.suptitle(f"E3 합치기 결과 읽는 법 ({T}초): '1위가 정답'과 '정답이 후보 안에 있음'은 다르다 (정답 = 층이 맞고 위치 3 m·방향 30° 안)", x=0.01, y=0.985, ha="left", fontsize=17, fontweight="bold", color=INK)
out = Path(a.out) if a.out else Path(f"reports/_03_E3_걸으며좁히기/3_결과표와_그림/이해용_1위와_후보안정답_{T}초.png"); fig.savefig(out, dpi=80, bbox_inches="tight"); print(out, ex)
