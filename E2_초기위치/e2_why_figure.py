#!/usr/bin/env python3
"""E2 '왜 올랐나' 그림(2026-10-10): 거리 비교(광선 거리 비교, 협업자 twoway)가 비슷한 다른 층 자리를 어떻게 걸러내나.
같은 출발점에서 벽 겹침 점수 1위(엉뚱한 자리)와 실제 위치를 나란히: 지도 위 후보 자리, 36방향(10°)마다
지도상 첫 벽까지 선(회색)과 실제 라이다가 그 방향으로 본 거리 점(색 = 두 거리 차이, 초록 작음·빨강 큼). 아래에 방향별 거리 비교 막대 없이 숫자.
    python3 scripts/e2_why_figure.py [--task 3F_T1] [--src T0_nospace]
결과: reports/_01_E2_초기위치/1_시각자료/3_왜올랐나_<층>_출발점<n>_<종류>.png
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402
import e2_twoway_test as TT  # noqa: E402
from e2_dense_score import distinct  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="3F_T1"); ap.add_argument("--src", default="T0_nospace"); a = ap.parse_args()
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(EV.G / "tasks/E2_tasks.csv"))}; EV.GT = {r["task_id"]: r for r in csv.DictReader(open(EV.G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    k = a.task; t = EV.TASK[k]; F, b = t["floor"], t["building"]; m_true = int(EV.GT[k]["true_map"])
    d = dict(np.load(Path("/data2/e2_dense_20261009") / a.src / f"{k}.npz")); meta = json.loads(str(d["meta"])); acc = distinct(d, meta, 1000)
    p_gt, ppm_gt, hd_gt, _, _ = EV.gt_of(k); wrap = lambda x: (x + 180) % 360 - 180
    ti = next(i for i, c in enumerate(acc) if c["m"] == m_true and np.hypot(*(c["px"] - p_gt)) / ppm_gt <= 3 and abs(wrap(c["hd"] - hd_gt)) <= 30)
    pts, yaw = EV.obs(k); import twoway as TW
    import e2_dense as ED
    L0 = ED._ST["cache"][(F, 1)].L; pts_full, _ = L0._scan(L0.T[0] + float(t["start_sec"])); pts_full = np.asarray(pts_full, float)
    fig, ax = plt.subplots(1, 2, figsize=(19, 8.4))
    for a_, (lab, c, colr) in zip(ax, ((f"벽 겹침 점수 1위: {acc[0]['m']}층 (엉뚱한 자리)", acc[0], EV.ORANGE), (f"실제 위치: {m_true}층 (벽 겹침 점수 {ti + 1}위)", acc[ti], EV.GREEN))):
        m = c["m"]; wall = TT.walls(b, m, F); im = EV.mapimg(b, m); a_.imshow(im, cmap="gray", vmin=0, vmax=255)
        hd = math.radians(c["hd"]) + yaw; d_obs = TW.sector_obs(pts_full, hd); d_map = TW.sector_map(wall, c["px"][0], c["px"][1], c["s"])
        ang = (np.arange(36) + 0.5) / 36 * 2 * np.pi; cx, cy = c["px"]; s = c["s"]
        diffs = []
        for i, th in enumerate(ang):
            if np.isfinite(d_map[i]):
                a_.plot([cx, cx + d_map[i] * s * np.cos(th)], [cy, cy - d_map[i] * s * np.sin(th)], color="#b9c0ca", lw=1.2, zorder=2)
            if np.isfinite(d_obs[i]):
                e = abs(d_obs[i] - d_map[i]) if np.isfinite(d_map[i]) else 4.0; diffs.append(min(e, 4.0))
                cc = EV.GREEN if e <= 0.5 else (EV.YELLOW if e <= 1.5 else EV.RED)
                a_.scatter(cx + d_obs[i] * s * np.cos(th), cy - d_obs[i] * s * np.sin(th), s=46, color=cc, edgecolor=EV.SURF, linewidth=0.6, zorder=4)
        a_.scatter(cx, cy, marker="*" if m == m_true else "o", s=380 if m == m_true else 220, color=EV.RED if m == m_true else colr, edgecolor=EV.SURF, zorder=5)
        r_ = 19 * s; a_.set_xlim(cx - r_, cx + r_); a_.set_ylim(cy + r_ * 0.8, cy - r_ * 0.8); a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_color(colr); sp.set_linewidth(3)
        a_.set_title(lab, loc="left", fontsize=14, fontweight="bold", color=colr)
        a_.text(0.02, 0.03, f"36방향 벽 거리 차이 중앙 {np.median(diffs):.1f} m", transform=a_.transAxes, fontsize=14, fontweight="bold", color=EV.INK,
                bbox=dict(facecolor=EV.SURF, edgecolor=colr, pad=5))
    fig.suptitle(f"왜 올랐나: 거리 비교 — {EV.fl(F)} 출발점{k.split('_T')[1]} ({EV.TYPE[t['start_type']]})", x=0.01, ha="left", fontsize=17, fontweight="bold", y=1.0)
    fig.text(0.01, 0.0, "회색 선 = 그 자리에서 지도상 첫 벽까지(10°마다 36방향), 점 = 실제 라이다가 그 방향으로 본 벽 거리 (초록 0.5 m 안 · 노랑 1.5 m 안 · 빨강 그 이상).\n"
             "벽 겹침 점수는 점이 '어느 벽에든' 붙으면 높아서 비슷한 복도가 있는 다른 층도 1위가 된다. 거리 비교는 '그 방향 그 거리에 벽이 있어야 한다'를 봐서 엉뚱한 자리를 걸러낸다.",
             fontsize=12, color=EV.INK2, va="top", linespacing=1.6)
    out = EV.PKG / "1_시각자료"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"3_왜올랐나_{EV.ko_task(k)}.png", dpi=130, bbox_inches="tight"); plt.close(fig); print("저장", ti + 1)


if __name__ == "__main__":
    main()
