#!/usr/bin/env python3
"""E4 층 모름(합치기 후보) 출발점마다 멈춘 장면 사진표(2026-10-12). 사람 판정을 테두리 색으로 보여 준다.
  칸 = 멈춘 순간 RealSense 화면 + 초록 상자(멈춰서 보려던 곳) + 상자 주변 확대(make_label_sheet.tile 과 같은 그림)
  테두리: 초록 = 사람이 표지판이라고 함, 빨강 = 표지판 아님, 회색 = 판정 없음
  제목: 판정 번호 · 출발 뒤 초 · 멈춘 방식(다 멈춤 / 애매할 때 / 갈릴 때)
출력: reports/_02_E4_표지판호출/1_본결과_층모름/멈춤사진_합치기/<건물>/<층>/출발점n_종류.png + 요약 그림
    python3 scripts/e4_unknown_stop_sheets.py
"""
import csv, json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402  (글꼴·색)
import make_label_sheet as LS  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); RUN = R / "70_층모름E4"; J = R / "25_판정"
OUT = Path("reports/_02_E4_표지판호출/1_본결과_층모름/멈춤사진_합치기")
POL = {"다 멈춤": RUN / "P_ALL", "애매할 때": RUN / "P_AMBIGUITY_gate_merged_d0.2"}
INF = RUN / "P_INFORMATIVE_merged_d0.2"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    M = json.load(open(J / "판정목록.json")); D = json.load(open(J / "판정결과.json")); X = json.load(open(J / "층모름합치기_판정기록.json"))
    TT = {t["task_id"]: t for t in csv.DictReader(open(EV.G / "tasks/E2_tasks.csv"))}
    segs = {s["task_id"]: s for s in csv.DictReader(open("/data2/e4_unknown_20261011/segments.csv"))}
    sc = {(r["trial_id"], r["policy"]): r for r in csv.DictReader(open(RUN / "채점_merged" / "출발점별_정책별.csv"))}
    tot = {"다 멈춤": [0, 0], "애매할 때": [0, 0], "갈릴 때": [0, 0]}
    for k, sg in segs.items():
        F = sg["floor"]; lo = float(sg["seg_lo"]); d = D.get(F, {})
        yes = set(d.get("plaque", [])) | set(d.get("sign", [])) | set(X["맞음"].get(F, []))
        judged = set(d.get("judged", range(1, d.get("judged_max", 0) + 1))) | set(X["시트"].get(F, []))
        inf = {round(x["t"], 3) for x in json.load(open(INF / f"{k}.json")) if x["informative"] and x["kind"] == "번호판"} if (INF / f"{k}.json").exists() else set()
        ev = {}
        for name, base in POL.items():
            for e in json.load(open(base / k / "run/events_live_replay.json")):
                key = (round(float(e["t"]), 1), tuple(int(v) for v in e["picks"][0][:4]))
                r = ev.setdefault(key, dict(t=float(e["t"]), box=[float(v) for v in e["picks"][0][:4]], pols=[]))
                r["pols"].append(name)
                if name == "애매할 때" and round(float(e["t"]), 3) in inf:
                    r["pols"].append("갈릴 때")
        items = sorted(ev.values(), key=lambda r: r["t"])
        for r in items:
            m = [it for it in M[F] if abs(it["t"] - r["t"]) <= 0.6 and iou(it["box"], r["box"]) > 0.3]
            r["id"] = m[0]["id"] if m else None
            r["lab"] = None if not m or m[0]["id"] not in judged else (m[0]["id"] in yes)
            for p in r["pols"]:
                tot[p][0] += 1; tot[p][1] += bool(r["lab"])
        t = TT[k]; n = max(len(items), 1); cols = 4; rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.9 * rows + 1.2), squeeze=False)
        for a in axs.ravel():
            a.axis("off")
        for a, r in zip(axs.ravel(), items):
            img = LS.tile(F, dict(t=r["t"], box=r["box"], id=r["id"] if r["id"] is not None else "-"))
            a.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); a.axis("on"); a.set_xticks([]); a.set_yticks([])
            col = EV.GREEN if r["lab"] else (EV.RED if r["lab"] is False else EV.GRAY)
            for sp in a.spines.values():
                sp.set_color(col); sp.set_linewidth(5)
            pol = " · ".join(p for p in ("다 멈춤", "애매할 때", "갈릴 때") if p in r["pols"])
            a.set_title(f"{'표지판 맞음' if r['lab'] else ('아님' if r['lab'] is False else '판정 없음')} · 출발 뒤 {r['t'] - lo:.0f}초\n멈춘 방식: {pol}", fontsize=10.5, color=col, loc="left", fontweight="bold")
        s_all = sc.get((k, "P_ALL"), {}); s_inf = sc.get((k, "P_INFORMATIVE"), {})
        fig.suptitle(f"E4 층 모름 · {EV.BNAME[t['building']]} {F.lstrip('GS').replace('F', '층')} 출발점{k.split('_T')[1]} ({EV.TYPE[t['start_type']]}) · 60초 동안 멈춘 장면 {len(items)}곳\n"
                     f"진짜 번호판 앞 멈춤 {'있음' if s_inf.get('chance') == '1' else '없음'}(갈릴 때만 멈춰도) · 번호 읽으면 60초 위치 {'맞음' if s_inf.get('oracle_ok_60') == '1' else '틀림'} · "
                     f"걷기만 했을 때 {'맞음' if s_all.get('top1_ok_60') == '1' else '틀림'}  (테두리 초록 = 사람 판정 표지판, 빨강 = 아님)",
                     x=0.01, ha="left", fontsize=13, fontweight="bold")
        od = OUT / EV.BNAME[t["building"]] / F.lstrip("GS").replace("F", "층"); od.mkdir(parents=True, exist_ok=True)
        fig.savefig(od / f"출발점{k.split('_T')[1]}_{EV.TYPE_FILE[t['start_type']]}.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    print("멈춤 방식별 (멈춤 수, 표지판 맞음):", tot)


if __name__ == "__main__":
    main()
