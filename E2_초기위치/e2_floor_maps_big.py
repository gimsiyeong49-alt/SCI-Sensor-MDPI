#!/usr/bin/env python3
"""E2 전층 결과 지도를 크게(2026-10-13). 13개 층을 한 장에 넣은 4_전층결과지도는 층 하나가 너무 작아 확대해도 안 보여서
  ① 층마다 한 장(가로 16인치, 출발점 동그라미 옆에 종류·순위), ② 전체 한 장은 PNG 220 dpi + PDF(확대해도 지도가 안 깨짐)로 다시 그린다.
그림 내용(색 = Top-100 안·1000위 안·밖, 원 안 숫자 = 출발점 종류, 회색 선 = 그 층 주행 경로)은 e2_visual_ko.v3 와 같다.
    python3 scripts/e2_floor_maps_big.py --src _nospace_지도v8_정밀검증 --method "+ 정밀 검증 (8차 지도)" --name 정밀검증_지도v8
    python3 scripts/e2_floor_maps_big.py --src _nospace --method "벽 겹침" --name 벽겹침
출력: reports/_01_E2_초기위치/1_시각자료_크게/<name>/ (층별 PNG 13장, 전체 PNG·PDF)
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

TYPE_SHORT = {"S1": "긴 직선 복도", "S2": "꺾이는 곳", "S3": "갈림길", "S4": "계단·엘베 앞", "S5": "막다른 곳"}
col_of = lambda v: EV.GREEN if (v and v <= 100) else (EV.YELLOW if v else EV.RED)
rk_of = lambda r: int(r["true_rank"]) if str(r["true_rank"]).isdigit() else None


def draw_floor(a, F, RES, big):
    tk = sorted([t for t in EV.TASK.values() if t["floor"] == F], key=lambda t: t["start_type"]); b = tk[0]["building"]; m = int(EV.GT[tk[0]["task_id"]]["true_map"])
    im = EV.mapimg(b, m); a.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    _, _, _, _, gxy = EV.gt_of(tk[0]["task_id"]); a.plot(gxy[:, 0], gxy[:, 1], color="#c3c8d0", lw=2.5 if big else 2, zorder=1)
    placed = []                                        # 이미 이름표를 단 점(가까우면 이름표를 위아래로 비켜 놓음)
    near_px = 0.06 * im.shape[1]
    for t in sorted(tk, key=lambda t: EV.gt_of(t["task_id"])[0][0]):
        p = EV.gt_of(t["task_id"])[0]; v = rk_of(RES[t["task_id"]])
        a.scatter(*p, s=420 if big else 300, color=col_of(v), edgecolor=EV.SURF, linewidth=2, zorder=4, alpha=0.92)
        a.text(*p, t["start_type"][1], ha="center", va="center", fontsize=11 if big else 9, fontweight="bold", color=EV.SURF, zorder=5)
        if big:                                        # 동그라미 옆에 종류·순위, 가까운 점끼리는 위아래로 번갈아 비킴 + 가리키는 선
            k = sum(1 for q in placed if np.hypot(*(np.asarray(q) - p)) < near_px); placed.append(p)
            dy = (22 + 30 * (k // 2)) * (1 if k % 2 == 0 else -1)
            a.annotate(f"{TYPE_SHORT[t['start_type']]} · {str(v) + '위' if v else '1000위 밖'}", p, xytext=(16, dy), textcoords="offset points", fontsize=12, fontweight="bold",
                       color=EV.INK, va="center", bbox=dict(boxstyle="round,pad=0.25", facecolor=EV.SURF, edgecolor=col_of(v), linewidth=1.6),
                       arrowprops=dict(arrowstyle="-", color=col_of(v), lw=1.4, shrinkA=0, shrinkB=6), zorder=6)
    ys, xs = np.where(im < 200); pad = 60; a.set_xlim(xs.min() - pad, xs.max() + pad); a.set_ylim(ys.max() + pad, ys.min() - pad); a.axis("off")
    nin = sum(1 for t in tk if rk_of(RES[t["task_id"]]) and rk_of(RES[t["task_id"]]) <= 100)
    return nin, (ys.max() - ys.min() + 2 * pad) / (xs.max() - xs.min() + 2 * pad)


def legend(a, fs):
    h = [Line2D([], [], marker="o", ls="", color=c, ms=fs + 2) for c in (EV.GREEN, EV.YELLOW, EV.RED)]
    a.legend(h, ["정답 위치가 Top-100 안", "100위 밖 · 1000위 안", "1000위 밖"], loc="upper left", frameon=False, fontsize=fs, ncol=3, bbox_to_anchor=(0, 0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--method", required=True); ap.add_argument("--name", required=True); a = ap.parse_args()
    G = EV.G; EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    RES = {r["trial_id"]: r for r in csv.DictReader(open(G / f"E2_dense_T0{a.src}/E2_top100_per_trial.csv"))}
    out = Path("reports/_01_E2_초기위치/1_시각자료_크게") / a.name; out.mkdir(parents=True, exist_ok=True)
    FL = list(dict.fromkeys(t["floor"] for t in EV.TASK.values())); tot = sum(1 for r in RES.values() if rk_of(r) and rk_of(r) <= 100)
    for i, F in enumerate(FL, 1):                                     # ① 층마다 한 장
        fig, ax = plt.subplots(figsize=(16, 9)); nin, asp = draw_floor(ax, F, RES, True)
        fig.set_size_inches(16, max(4.5, 16 * asp) + 1.4)
        ax.set_title(f"{EV.fl(F)} — 정답 위치가 Top-100 안 {nin}/5  (A 라이다 · 축척 탐색 · {a.method})", loc="left", fontsize=16, fontweight="bold")
        legend(ax, 13); fig.text(0.99, 0.01, "원 안 숫자 = 출발점 종류(1 긴 직선 복도 · 2 꺾이는 곳 · 3 갈림길 · 4 계단·엘베 앞 · 5 막다른 곳), 회색 선 = 그 층 주행 경로", ha="right", fontsize=11, color=EV.INK2)
        fig.savefig(out / f"{i:02d}_{EV.fl(F).replace(' ', '')}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(7, 2, figsize=(17, 25)); ax = ax.ravel()   # ② 전체 한 장(PDF 는 확대 가능)
    for i, F in enumerate(FL):
        nin, _ = draw_floor(ax[i], F, RES, False)
        parts = [f"{t['start_type']} {str(rk_of(RES[t['task_id']])) + '위' if rk_of(RES[t['task_id']]) else '1000위 밖'}" for t in sorted([t for t in EV.TASK.values() if t["floor"] == F], key=lambda t: t["start_type"])]
        ax[i].set_title(f"{EV.fl(F)} — Top-100 안 {nin}/5", loc="left", fontsize=13, fontweight="bold")
        ax[i].text(0.0, -0.02, "  ·  ".join(parts), transform=ax[i].transAxes, fontsize=10.5, color=EV.INK2, va="top")
    ax[13].axis("off"); legend(ax[13], 13); ax[13].get_legend().set_bbox_to_anchor((0, 0.9))
    ax[13].text(0.5, 0.35, "원 안 숫자 = 출발점 종류\n1 긴 직선 복도 · 2 꺾이는 곳 · 3 갈림길\n4 계단·엘리베이터 앞 · 5 막다른 곳\n회색 선 = 그 층 주행 경로", ha="center", va="center", fontsize=12, color=EV.INK2, transform=ax[13].transAxes, linespacing=1.7)
    fig.suptitle(f"13개 층 65개 출발점: 제자리 한 번 관측으로 정답 위치가 몇 위였나 (A 라이다 · 축척 탐색 · {a.method}, Top-100 안 {tot}/65)", x=0.01, ha="left", fontsize=19, fontweight="bold", y=1.0)
    fig.tight_layout(h_pad=2.5); fig.savefig(out / "00_전체.png", dpi=220, bbox_inches="tight"); fig.savefig(out / "00_전체.pdf", bbox_inches="tight"); plt.close(fig)
    print(out, tot)


if __name__ == "__main__":
    main()
