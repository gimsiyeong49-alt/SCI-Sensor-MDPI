#!/usr/bin/env python3
"""E2·E3 발표·논문용 그래프(2026-10-12, 지도 8차). e2e3_eval_v8.py 가 만든 CSV 만 읽는다. PNG 와 PDF 로 저장.
    python3 scripts/e2e3_figs_clean.py
출력: reports/_11_E2E3_지도v8_평가_20261012/그래프/
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402,F401  (글꼴)

P = Path("reports/_11_E2E3_지도v8_평가_20261012"); O = P / "그래프"; O.mkdir(exist_ok=True)
C_TOP1, C_TOP10, C_IN, C_NODE, C_SEED, C_GRAY = "#1B7F5B", "#D9822B", "#2F6DB5", "#2F6DB5", "#9BB7DE", "#B8BFC9"
INK, INK2, GRID = "#1F2328", "#57606A", "#E6E8EB"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#8C959F", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def clean(ax, ygrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(O / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(O / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def label_end(ax, x, y, txt, c, dy=0):
    ax.annotate(txt, (x, y), xytext=(8, dy), textcoords="offset points", va="center", color=c, fontsize=12, fontweight="bold")


R2 = list(csv.DictReader(open(P / "E2/E2_출발점별.csv"))); N = len(R2)
TK = list(csv.DictReader(open(P / "E2/E2_TopK.csv"))); SENS = list(csv.DictReader(open(P / "E2/E2_임계값민감도.csv")))
TAB = list(csv.DictReader(open(P / "E3/E3_시점별_집단별.csv"))); SURV = list(csv.DictReader(open(P / "E3/E3_정답생존_복구.csv")))

# 1. E2 Top-K
K = [int(r["K"]) for r in TK]; hit = [int(r["정답포함"]) for r in TK]; fh = [int(r["정답층포함"]) for r in TK]
fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(K, [v / N * 100 for v in fh], "-o", color=C_GRAY, lw=2.2, ms=7)
ax.plot(K, [v / N * 100 for v in hit], "-o", color=C_IN, lw=2.8, ms=8)
for k, v in zip(K, hit):
    ax.annotate(f"{v / N * 100:.0f}%", (k, v / N * 100), xytext=(0, -17), textcoords="offset points", ha="center", color=C_IN, fontsize=10.5)
label_end(ax, K[-1], hit[-1] / N * 100, "정답 후보 포함\n(층 + 3 m + 30°)", C_IN)
label_end(ax, K[-1], fh[-1] / N * 100, "정답 층 후보 포함", INK2)
ax.set_xscale("log"); ax.set_xticks(K); ax.set_xticklabels(K); ax.minorticks_off(); ax.set_ylim(0, 105); ax.set_xlim(0.8, 130)
ax.set_xlabel("상위 K개 후보"); ax.set_ylabel("출발 구간 비율 (%)"); clean(ax)
ax.set_title("E2 초기 후보: 상위 K개 안에 정답이 있는 비율 (n = 65)", loc="left", fontsize=13.5, fontweight="bold", color=INK)
save(fig, "E2_1_TopK")

# 2. E2 놓친 이유 (65 = 1위 + Top-100 안 + 정밀 검증 후 밀림 + 첫 단계 1000위 밖)
n1 = sum(1 for r in R2 if r["정답순위"] == "1"); nin = sum(1 for r in R2 if r["정답순위"] not in ("", "1"))
npush = sum(1 for r in R2 if not r["정답순위"] and r["벽겹침단계_정답순위"] != "1000밖"); nout = sum(1 for r in R2 if r["벽겹침단계_정답순위"] == "1000밖" and not r["정답순위"])
parts = [(n1, "1위가 정답", C_TOP1), (nin, "2~100위 안", "#7DBF9E"), (npush, "1000위 안이었으나\n정밀 검증 후 100위 밖", C_TOP10), (nout, "첫 단계부터\n1000위 밖", C_GRAY)]
fig, ax = plt.subplots(figsize=(10, 2.6)); x = 0
for v, lab, c in parts:
    ax.barh(0, v, left=x, color=c, height=0.55, edgecolor="white", lw=2)
    ax.text(x + v / 2, 0, f"{v}", ha="center", va="center", color="white" if c in (C_TOP1, C_TOP10) else INK, fontsize=14, fontweight="bold")
    ax.text(x + v / 2, -0.52, lab, ha="center", va="top", color=INK, fontsize=10.5); x += v
ax.annotate("", xy=(0, 0.45), xytext=(n1 + nin, 0.45), arrowprops=dict(arrowstyle="<->", color=C_IN, lw=1.5))
ax.text((n1 + nin) / 2, 0.55, f"상위 100 안 {n1 + nin}곳 ({(n1 + nin) / N * 100:.0f}%)", ha="center", va="bottom", color=C_IN, fontsize=11.5, fontweight="bold")
ax.annotate("", xy=(n1 + nin, 0.45), xytext=(N, 0.45), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5))
ax.text((n1 + nin + N) / 2, 0.55, f"놓침 {npush + nout}곳", ha="center", va="bottom", color=INK2, fontsize=11.5, fontweight="bold")
ax.set_xlim(0, N); ax.set_ylim(-1.3, 1.0); ax.axis("off")
ax.set_title("E2 65개 출발 구간: 정답이 어디에 있었나", loc="left", fontsize=13.5, fontweight="bold", color=INK)
save(fig, "E2_2_놓친이유")

# 3. E2 위치 허용 민감도 (방향 허용은 10~30° 모두 같음)
rr = [1, 2, 3]
t100 = [int(next(s["Top100"] for s in SENS if float(s["위치_m"]) == r and float(s["방향_도"]) == 30)) for r in rr]
t1 = [int(next(s["Top1"] for s in SENS if float(s["위치_m"]) == r and float(s["방향_도"]) == 30)) for r in rr]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.plot(rr, [v / N * 100 for v in t100], "-o", color=C_IN, lw=2.8, ms=8); ax.plot(rr, [v / N * 100 for v in t1], "-o", color=C_TOP1, lw=2.8, ms=8)
for r, v in zip(rr, t100):
    ax.annotate(f"{v}곳", (r, v / N * 100), xytext=(0, 9), textcoords="offset points", ha="center", color=C_IN, fontsize=11)
label_end(ax, 3, t100[-1] / N * 100, "상위 100 안", C_IN); label_end(ax, 3, t1[-1] / N * 100, "1위", C_TOP1)
ax.set_xticks(rr); ax.set_xticklabels(["1 m", "2 m", "3 m"]); ax.set_xlim(0.8, 3.7); ax.set_ylim(0, 60); ax.set_xlabel("정답으로 인정하는 위치 오차"); ax.set_ylabel("출발 구간 비율 (%)"); clean(ax)
ax.set_title("E2 정답 기준 민감도 (방향 10·20·30° 는 결과 동일)", loc="left", fontsize=13, fontweight="bold", color=INK)
save(fig, "E2_3_위치기준민감도")

# 4. E3 전체 시간 곡선
T = ["0", "15", "30", "45", "60"]; X = [int(t) for t in T]
all_ = [r for r in TAB if r["집단"] == "전체"]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
for key, c, lab in (("후보안", C_IN, "후보 안에 정답"), ("상위10", C_TOP10, "상위 10 안"), ("일위정답", C_TOP1, "1위가 정답")):
    v = [int(r[key]) / 65 * 100 for r in all_]; ax.plot(X, v, "-o", color=c, lw=2.8, ms=8)
    for x_, r in zip(X, all_):
        ax.annotate(f"{int(r[key]) / 65 * 100:.0f}", (x_, int(r[key]) / 65 * 100), xytext=(0, -17 if key == "상위10" else 9), textcoords="offset points", ha="center", color=c, fontsize=10)
    label_end(ax, 60, v[-1], lab, c, dy={"후보안": 6, "상위10": -6, "일위정답": 0}[key])
ax.axvspan(-3, 3, color="#F2F4F7", zorder=0); ax.text(0, 101, "E2", ha="center", color=INK2, fontsize=10)
ax.set_xticks(X); ax.set_xticklabels([f"{x}초" for x in X]); ax.set_xlim(-3, 75); ax.set_ylim(0, 108); ax.set_xlabel("출발 후 걸은 시간"); ax.set_ylabel("출발 구간 비율 (%)"); clean(ax)
ax.set_title("E3 걸으며 후보 갱신: 시간에 따른 정답 포함과 1위 정답 (n = 65)", loc="left", fontsize=13.5, fontweight="bold", color=INK)
save(fig, "E3_1_시간별성능")

# 5. E3 A 정답 유지 / B 정답 복구
A = [r for r in TAB if r["집단"].startswith("A")]; B = [r for r in TAB if r["집단"].startswith("B")]; nA = int(A[0]["분모"]); nB = int(B[0]["분모"])
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
ax = axs[0]
ax.plot(X, [int(r["후보안"]) / nA * 100 for r in A], "-o", color=C_IN, lw=2.8, ms=8); ax.plot(X, [int(r["일위정답"]) / nA * 100 for r in A], "-o", color=C_TOP1, lw=2.8, ms=8)
for x_, r in zip(X, A):
    ax.annotate(f"{r['후보안']}", (x_, int(r["후보안"]) / nA * 100), xytext=(0, 9), textcoords="offset points", ha="center", color=C_IN, fontsize=10.5)
    ax.annotate(f"{r['일위정답']}", (x_, int(r["일위정답"]) / nA * 100), xytext=(0, -17), textcoords="offset points", ha="center", color=C_TOP1, fontsize=10.5)
label_end(ax, 60, int(A[-1]["후보안"]) / nA * 100, "정답 유지", C_IN); label_end(ax, 60, int(A[-1]["일위정답"]) / nA * 100, "1위가 정답", C_TOP1)
ax.text(0.5, 92, "0초는 정의상 100%\n(E2 에서 정답을 찾은 구간)", fontsize=9.5, color=INK2, va="top")
ax.set_title(f"A. E2 에서 정답을 찾은 {nA}곳: 정답을 지키나", loc="left", fontsize=13, fontweight="bold", color=INK)
ax = axs[1]
node = [0] + [int(s["B_그중_노드재탐색에서"]) for s in SURV]; rec = [0] + [int(s["B_정답복구"]) for s in SURV]
ax.bar(X, [n_ / nB * 100 for n_ in node], width=6, color=C_NODE, label="노드 재탐색이 새로 찾은 후보")
ax.bar(X, [(r_ - n_) / nB * 100 for r_, n_ in zip(rec, node)], bottom=[n_ / nB * 100 for n_ in node], width=6, color=C_SEED, label="E2 후보가 다듬어져 정답이 됨")
ax.plot(X, [int(r["일위정답"]) / nB * 100 for r in B], "-o", color=C_TOP1, lw=2.8, ms=8, zorder=5)
for x_, r_ in zip(X, rec):
    ax.text(x_, r_ / nB * 100 + 2, f"{r_}", ha="center", color=INK, fontsize=10.5, fontweight="bold")
label_end(ax, 60, int(B[-1]["일위정답"]) / nB * 100, "1위가 정답", C_TOP1)
ax.text(1.5, 92, "0초는 정의상 0%\n(E2 에서 정답을 놓친 구간)", fontsize=9.5, color=INK2, va="top")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.78), fontsize=10)
ax.set_title(f"B. E2 에서 정답을 놓친 {nB}곳: 정답을 되찾나", loc="left", fontsize=13, fontweight="bold", color=INK)
for a in axs:
    a.set_xticks(X); a.set_xticklabels([f"{x}초" for x in X]); a.set_xlim(-5, 75); a.set_ylim(0, 108); a.set_xlabel("출발 후 걸은 시간"); clean(a)
axs[0].set_ylabel("구간 비율 (%)  ·  숫자 = 곳 수")
save(fig, "E3_2_정답유지_복구")

# 6. E3 남은 후보 수
med = [int(r["후보수_중앙"]) for r in all_]
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.plot(X, med, "-o", color=INK2, lw=2.6, ms=8)
for x_, v in zip(X, med):
    ax.annotate(f"{v}", (x_, v), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=11)
ax.set_xticks(X); ax.set_xticklabels([f"{x}초" for x in X]); ax.set_ylim(0, 115); ax.set_xlabel("출발 후 걸은 시간"); ax.set_ylabel("남은 후보 수 (중앙값)"); clean(ax)
ax.set_title("E3 남은 후보 수 (65곳 중앙값)", loc="left", fontsize=13, fontweight="bold", color=INK)
save(fig, "E3_3_남은후보수")
print(sorted(p.name for p in O.glob("*.png")))
