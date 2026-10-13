#!/usr/bin/env python3
"""E3 논문용 그림 두 장(2026-10-13, 지도 8차). 65칸 순위 표(E3_읽는법) 대신 쓴다.
  E3_4_RecallK_시점별: 정답이 상위 K 안에 있는 출발점 비율. E2(0초)와 E3 15·30·45·60초를 한 축에 겹침
  E3_5_실패원인: 시점마다 65곳을 1위 정답 / 정답 후보 있음·1위가 다른 층 / 정답 후보 있음·1위가 같은 층 다른 자리 / 정답 후보 없음으로 나눔
입력: e2e3_eval_v8.py 가 만든 E3_출발점별_시점별.csv(정답 순위), 합치기 남은 후보(1위 층), E2 출발점별(1위 층). 정답은 채점에만.
    python3 scripts/e3_paper_figs.py
출력: reports/_11_E2E3_지도v8_평가_20261012/그래프/ (PNG 200 dpi + PDF), 숫자는 E3/E3_실패원인.csv
"""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402,F401  (글꼴)

P = Path("reports/_11_E2E3_지도v8_평가_20261012"); O = P / "그래프"
CM = Path("reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색_지도v8/연결판_남은후보.csv")
E2T = Path("reports/_01_E2_초기위치/_작업자료/E2_dense_T0_nospace_지도v8_정밀검증/E2_top100_per_trial.csv")
INK, INK2, GRID = "#1F2328", "#57606A", "#E6E8EB"
C_TOP1, C_FLOOR, C_SAME, C_NONE = "#1B7F5B", "#D9822B", "#F2C48D", "#B8BFC9"
SEQ = ["#9BB7DE", "#6C97CF", "#3F75BC", "#1F4E8C"]           # 15·30·45·60초: 한 색 옅음 → 짙음
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#8C959F", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
TS = (15, 30, 45, 60)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(O / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(O / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


rk = lambda v: int(v) if str(v).strip().isdigit() else None
R = list(csv.DictReader(open(P / "E3/E3_출발점별_시점별.csv"))); N = len(R)
rank = {0: {r["출발점"]: rk(r["E2_정답순위"]) for r in R}}
for T in TS:
    rank[T] = {r["출발점"]: rk(r[f"{T}초_정답순위"]) for r in R}
true_floor = {}
for r in csv.DictReader(open("reports/_01_E2_초기위치/_작업자료/tasks/tasks_gt_EVAL_ONLY.csv")):
    true_floor[r["task_id"]] = int(r["true_map"])
top1 = {0: {r["trial_id"]: int(r["top1_floor"]) for r in csv.DictReader(open(E2T))}}
for r in csv.DictReader(open(CM)):
    if r["rank"] == "1":
        top1.setdefault(int(r["T"]), {})[r["trial_id"]] = int(r["floor_id"])
rank[0] = {k: (v if v is not None and v <= 100 else None) for k, v in rank[0].items()}

# 1. Recall@K 시점별
K = [1, 2, 3, 5, 10, 20, 50, 100]
fig, ax = plt.subplots(figsize=(7.8, 4.8))
curves = [(0, "E2 정지 (0초)", C_NONE, "--")] + [(T, f"E3 {T}초", c, "-") for T, c in zip(TS, SEQ)]
ends = {}
for T, lab, c, ls in curves:
    y = [100 * sum(1 for v in rank[T].values() if v is not None and v <= k) / N for k in K]
    ax.plot(K, y, ls, color=c, lw=2.6 if T else 2.2, marker="o", ms=6 if T else 5, label=lab, zorder=3 if T else 2)
    ends.setdefault(round(y[-1]), []).append(lab.split(" (")[0].replace("E3 ", "").replace("초", ""))
for v, labs in ends.items():                                   # 끝값이 같은 시점은 이름표 하나로(겹침 방지)
    txt = ("E2 정지" if labs == ["E2 정지"] else "·".join(labs) + "초") + f"  {v}%"
    ax.annotate(txt, (K[-1], v), xytext=(8, 0), textcoords="offset points", va="center", color=INK2 if labs == ["E2 정지"] else INK, fontsize=10.5)
ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels([str(k) for k in K]); ax.minorticks_off()
ax.set_xlim(0.85, 160); ax.set_ylim(0, 100); ax.set_xlabel("상위 K개 후보"); ax.set_ylabel("정답이 상위 K 안에 있는 출발점 (%)")
clean(ax); ax.legend(frameon=False, loc="upper left", fontsize=10.5)
ax.set_title("E2·E3 Recall@K: 걸을수록 정답이 위로 올라온다 (n = 65)", loc="left", fontsize=13.5, fontweight="bold", color=INK)
save(fig, "E3_4_RecallK_시점별")

# 2. 실패 원인 누적 막대
cats = [("1위가 정답", C_TOP1), ("정답 후보 있음 · 1위가 다른 층", C_FLOOR), ("정답 후보 있음 · 1위가 같은 층 다른 자리", C_SAME), ("정답 후보 없음", C_NONE)]
rows = []
for T in (0,) + TS:
    c = [0, 0, 0, 0]
    for k, v in rank[T].items():
        if v == 1:
            c[0] += 1
        elif v is None:
            c[3] += 1
        elif top1[T][k] != true_floor[k]:
            c[1] += 1
        else:
            c[2] += 1
    rows.append(dict(시점=("0초(E2)" if T == 0 else f"{T}초"), **{n: v for (n, _), v in zip(cats, c)}))
with open(P / "E3/E3_실패원인.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
fig, ax = plt.subplots(figsize=(8.6, 4.6))
x = np.arange(len(rows)); bottom = np.zeros(len(rows))
for n_, col in cats:
    v = np.array([r[n_] for r in rows], float)
    ax.bar(x, v / N * 100, 0.62, bottom=bottom, color=col, edgecolor="white", linewidth=2, label=n_, zorder=3)
    for xi, vi, bi in zip(x, v, bottom):
        if vi >= 3:
            ax.text(xi, bi + vi / N * 50, f"{int(vi)}", ha="center", va="center", fontsize=10.5, color="white" if col in (C_TOP1, C_FLOOR) else INK, fontweight="bold")
    bottom += v / N * 100
ax.set_xticks(x); ax.set_xticklabels([r["시점"] for r in rows]); ax.set_ylim(0, 100); ax.set_ylabel("출발점 비율 (%)  ·  막대 안 숫자 = 곳 수")
clean(ax); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10.5)
ax.set_title("E3 1위가 틀린 이유: 정답을 찾고도 층을 잘못 고른 경우가 가장 많다 (n = 65)", loc="left", fontsize=13, fontweight="bold", color=INK)
save(fig, "E3_5_실패원인")
for r in rows:
    print(r)
