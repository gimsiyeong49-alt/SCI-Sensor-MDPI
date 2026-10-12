#!/usr/bin/env python3
"""E4 오검출 종류 집계(2026-10-08, PILOT). 25_판정/오검출종류_추정.json(사용자가 짐작을 확인함)으로
paper/csv/E4_fp_types.csv, paper/tables/E4_fp_types.md, paper/figures/E4_fig_fp_types.{png,pdf} 를 만든다.
    python3 scripts/e4_fp_types.py
"""
import csv, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import plt, save, BLUE, INK, SURF  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); J = R / "25_판정"; OUT = R / "59_E4_호출정책" / os.environ.get("E4_PAPER", "paper")
FPLIST = os.environ.get("E4_FPLIST", "")   # 2026-10-09: 본 정책이 바뀌면 [{event_id, floor, t, fp_type, source}] json
G = json.load(open(J / "오검출종류_추정.json")); L = json.load(open(J / "오검출종류_목록.json"))
if FPLIST:
    L = [dict(no=i, event_id=x["event_id"], floor=x["floor"], t=x["t"]) for i, x in enumerate(json.load(open(FPLIST)), 1)]
    G = dict(categories=G["categories"], guess={str(i): x["fp_type"] for i, x in enumerate(json.load(open(FPLIST)), 1)})
EN = {"A": "paper notice on door/wall", "B": "exit sign light", "C": "bulletin board / poster", "D": "box / device (alarm, panel, monitor)",
      "E": "other sign or wall lettering (not a room sign)", "F": "plain wall / door frame / window"}
rows = [dict(no=x["no"], event_id=x["event_id"], floor=x["floor"], relative_sec=x["t"], fp_type=G["guess"][str(x["no"])],
             fp_type_ko=G["categories"][G["guess"][str(x["no"])]], fp_type_en=EN[G["guess"][str(x["no"])]]) for x in L]
with open(OUT / "csv/E4_fp_types.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
C = Counter(r["fp_type"] for r in rows); n = len(rows); ks = sorted(C, key=lambda k: -C[k])
md = [(f"C 배포판 멈춤 중 사람 판정 '표지판 아님' {n}개(판정한 4개 층). 종류: 10-08 판과 같은 멈춤은 사용자 확인 종류, 새 14개는 Claude 매김(확인 필요, 2026-10-09)." if FPLIST else
       f"P_AMBIGUITY 멈춤 중 사람 판정 '표지판 아님' {n}개(판정한 4개 층). 종류는 Claude 가 먼저 매기고 사용자가 전부 맞다고 확인(2026-10-08)."), "",
      "| 종류 | 개수 | 비율 |", "|---|---|---|"] + [f"| {G['categories'][k]} | {C[k]} | {100 * C[k] / n:.0f}% |" for k in ks]
(OUT / "tables/E4_fp_types.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
fig, a = plt.subplots(figsize=(6, 2.6)); y = np.arange(len(ks))[::-1]
a.barh(y, [C[k] for k in ks], height=0.6, color=BLUE, ec=SURF, lw=2)
for yi, k in zip(y, ks):
    a.text(C[k] + 0.15, yi, f"{C[k]} ({100 * C[k] / n:.0f}%)", va="center", fontsize=8, color=INK)
a.set_yticks(y); a.set_yticklabels([EN[k] for k in ks]); a.set_xlim(0, max(C.values()) * 1.3); a.grid(axis="y", visible=False)
a.set_xlabel("false stops"); a.set_title(f"E4 false sign stops by type (n={n}, PILOT" + (", deployed C" if FPLIST else ", human-confirmed") + ")", fontsize=9.5, color=INK)
fig.tight_layout(); save(fig, "E4_fig_fp_types")
