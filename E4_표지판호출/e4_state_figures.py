#!/usr/bin/env python3
"""E4 결과물(새 계획서 6.7): 출발점마다 후보 점수·순위 변화와 매 시점 위치 상태 그림(2026-10-12).
  위: 층마다 그 층 최고 후보의 merit(낮을수록 잘 맞음). 정답 층 = 초록 굵은 선, 나머지 층 = 회색, 점선 = 1위 merit + Δ(0.2, 이 안이면 애매 경쟁)
  가운데: 정답 층의 순위(층 순위, 1 = 1위)와 정답 후보 순위(Top-100 중)
  아래: 매 계산 상태 띠(e4_state_eval.py 와 같은 판단) + 그 시점 1위가 맞았나(●맞음 ×틀림)
출력: reports/_02_E4_표지판호출/2_본결과_상태판단/출발점별그림/<건물>/<층>/출발점n_종류.png
    python3 scripts/e4_state_figures.py
"""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402  (글꼴·색)
import e4_state_eval as SE  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUTD = Path("reports/_02_E4_표지판호출/2_본결과_상태판단/출발점별그림" + {"_merged": "_합치기", "_v8": "_지도v8"}.get(SE.UG.SFX, ""))
SCOL = {"위치 확정": EV.GREEN, "층 간 모호": EV.ORANGE, "층 내 모호": EV.YELLOW, "관측 부족": EV.GRAY, "센서 이상": EV.RED}
D = 0.2


def main():
    segs, recs, _ = SE.load()
    st = SE.classify(recs, D, 10.0, "multi")
    by = {}
    for r, s in zip(recs, st):
        by.setdefault(r["k"], []).append((r, s))
    for k, L in by.items():
        sg = segs[k]; F = sg["floor"]; b = sg["building"]; m_true = L[0][0]["m_true"]
        T = np.array([r["T"] for r, _ in L]); floors = sorted({c["m"] for r, _ in L for c in r["P"]})
        best = {m: [min([c["merit"] for c in r["P"] if c["m"] == m], default=np.nan) for r, _ in L] for m in floors}
        top = np.array([r["top"]["merit"] for r, _ in L])
        frank, crank = [], []
        for r, _ in L:
            bm = sorted({c["m"]: c["merit"] for c in r["P"][::-1]}.items(), key=lambda kv: kv[1])
            frank.append(next((i + 1 for i, (m, _) in enumerate(bm) if m == m_true), np.nan))
        fig = plt.figure(figsize=(13, 8.2)); gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.4, 0.6], hspace=0.35)
        a = fig.add_subplot(gs[0])
        for m in floors:
            if m != m_true:
                a.plot(T, best[m], color=EV.GRAY, lw=1.1, alpha=0.8)
        a.plot(T, best.get(m_true, [np.nan] * len(T)), color=EV.GREEN, lw=3, label=f"정답 층({m_true}층) 최고 후보")
        a.plot(T, top, color=EV.INK, lw=0, marker="o", ms=4, label="1위")
        a.plot(T, top + D, color=EV.INK2, lw=1, ls="--", label=f"1위 + Δ({D}) — 이 안이면 애매 경쟁")
        a.plot([], [], color=EV.GRAY, lw=1.1, label="다른 층 최고 후보")
        a.set_ylabel("merit (낮을수록 잘 맞음)"); a.set_xlim(2, 61); a.legend(loc="upper right", fontsize=9.5, ncol=2, frameon=False)
        a.set_title("① 층마다 가장 잘 맞는 후보의 점수", loc="left", fontsize=12.5, fontweight="bold")
        a2 = fig.add_subplot(gs[1]); a2.plot(T, frank, color=EV.GREEN, marker="o", ms=4, lw=2)
        a2.set_ylim(len(floors) + 0.5, 0.5); a2.set_yticks(range(1, len(floors) + 1)); a2.set_xlim(2, 61); a2.set_ylabel("정답 층 순위")
        a2.set_title("② 정답 층이 몇 번째로 잘 맞나 (1 = 1위)", loc="left", fontsize=12.5, fontweight="bold")
        a3 = fig.add_subplot(gs[2])
        for (r, s), t in zip(L, T):
            a3.add_patch(plt.Rectangle((t - 1.5, 0), 3, 1, color=SCOL[s], lw=0))
            a3.text(t, 0.5, "●" if r["ok"] else "×", ha="center", va="center", fontsize=9, color=EV.INK)
        a3.set_xlim(2, 61); a3.set_ylim(0, 1); a3.set_yticks([]); a3.set_xlabel("출발 뒤 시간(초)")
        for s, c in SCOL.items():
            a3.add_patch(plt.Rectangle((0, 0), 0, 0, color=c, label=s))
        a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.9), ncol=5, fontsize=9.5, frameon=False)
        a3.set_title("③ 매 계산 위치 상태 (● 그 시점 1위 맞음, × 틀림)", loc="left", fontsize=12.5, fontweight="bold")
        n_ok = sum(r["ok"] for r, _ in L); n_conf_wrong = sum((s == "위치 확정") and not r["ok"] for r, s in L)
        fig.suptitle(f"E4 위치 상태 — {EV.BNAME[b]} {F.lstrip('GS').replace('F', '층')} 출발점{k.split('_T')[1]} ({EV.TYPE.get(sg_type(k), '')}) · "
                     f"1위 맞음 {n_ok}/{len(L)} · 확정인데 틀림 {n_conf_wrong}", x=0.01, ha="left", fontsize=14.5, fontweight="bold", y=0.985)
        od = OUTD / EV.BNAME[b] / F.lstrip("GS").replace("F", "층"); od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / f"출발점{k.split('_T')[1]}_{EV.TYPE_FILE.get(sg_type(k), '')}.png", dpi=90, bbox_inches="tight"); plt.close(fig)
    print("그림", len(by), "장 →", OUTD)


_TYPES = None


def sg_type(k):
    global _TYPES
    if _TYPES is None:
        import csv
        _TYPES = {t["task_id"]: t["start_type"] for t in csv.DictReader(open(SE.G / "tasks/E2_tasks.csv"))}
    return _TYPES[k]


if __name__ == "__main__":
    main()
