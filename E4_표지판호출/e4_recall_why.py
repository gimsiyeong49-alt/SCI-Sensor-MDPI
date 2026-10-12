#!/usr/bin/env python3
"""E4 호실 단위 재현율 분석(2026-10-08, PILOT). e4_recall.py 결과(호실마다)와 멈춤을 놓고 층마다:
  멈춤 수, 그중 호실 벽에 찍힌 멈춤, 같은 호실 재멈춤, 판정 층이면 틀린 멈춤
  놓친 호실을 이유별로: (가) 확실한 동안 지나감(규칙상 안 멈춤) (나) 헷갈렸지만 ±3초 안에 다른 멈춤이 있었음(한 번에 하나)
                       (다) 헷갈렸고 근처에 멈춤도 없었음(후보를 못 찾았거나 점수 미달)
    python3 scripts/e4_recall_why.py
"""
import csv, json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e4_recall as E  # noqa: E402
from obs_feedback import A  # noqa: E402

OUT = E.OUT; MAIN = E.MAIN; NAME = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}


def main():
    Mj = json.load(open(E.J / "판정목록.json")); Dj = json.load(open(E.J / "판정결과.json"))
    RR = list(csv.DictReader(open(OUT / os.environ.get("E4_ROOMS", "csv/E4_recall_rooms.csv"))))
    T = list(csv.DictReader(open(OUT / "csv/E4_trigger_decisions.csv")))
    rows = []
    for F in E.FL:
        rooms, _ = E.floor_rooms(F); byid = {r["id"]: r for r in rooms}
        S = json.load(open(E.R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]
        g = list(csv.DictReader(open(E.R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gt = np.array([float(x["relative_sec"]) for x in g]); gxy = np.array([[float(x["x_px"]), float(x["y_px"])] for x in g])
        Mg = A(math.radians(S["theta_deg"])) * ppm
        tl = list(csv.DictReader(open(E.L / F / "loc/timeline.csv"))); tt = np.array([float(x["relative_sec"]) for x in tl])
        od = np.array([[float(x["odom_x_m"]), float(x["odom_y_m"])] for x in tl])
        judged = F in Dj
        if judged:
            Pset = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", [])); jd = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1)))
        E_ = json.load(open(E.P / MAIN / F / "run/events_live_replay.json"))
        lab, hit = [], []
        for e in E_:
            l = "?"
            if judged:
                m = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and E.iou(it["box"], e["picks"][0][:4]) > 0.3]
                l = ("TP" if m[0]["id"] in Pset else "FP") if (m and m[0]["id"] in jd) else "?"
            i = int(np.argmin(abs(gt - e["t"]))); oo = od[int(np.argmin(abs(tt - e["t"])))]
            mp = gxy[i] + Mg @ (np.array(e["xyz"][:2]) - oo) if e.get("xyz") else None
            best = None
            if mp is not None:
                for r in rooms:
                    dd = np.min(np.linalg.norm(r["edge"] - mp, axis=1)) / ppm
                    if dd <= E.HIT_M and (best is None or dd < best[0]):
                        best = (dd, r["id"])
            lab.append(l); hit.append(best[1] if best else None)
        cnt = Counter(h for h, l in zip(hit, lab) if h and l != "FP")
        repeat = sum(v - 1 for v in cnt.values())
        st = {float(x["relative_sec"]): x["uncertainty_class"] for x in T if x["trial_id"] == f"PILOT_{F}"}; sk = np.array(sorted(st))
        stop_t = np.array([e["t"] for e in E_])
        why = Counter()
        rr = [r for r in RR if r["trial_id"] == f"PILOT_{F}"]
        for r in rr:
            if r[f"seen_{MAIN}"] == "1":
                continue
            t0 = float(r["closest_pass_sec"]); k = sk[np.argmin(abs(sk - t0))] if len(sk) else None
            if k is None or abs(k - t0) > 1.0:
                why["위치추정 시작 전"] += 1
            elif st[k] == "GEOMETRIC_CONFIDENT":
                why["확실해서 안 멈춤"] += 1
            elif len(stop_t) and np.min(abs(stop_t - t0)) <= 3.0:
                why["다른 곳에 멈춰 있었음"] += 1
            else:
                why["헷갈렸는데 못 찾음"] += 1
        rows.append(dict(F=F, judged=judged, passed=len(rr), seen=sum(r[f"seen_{MAIN}"] == "1" for r in rr), stops=len(E_),
                         tp=lab.count("TP"), fp=lab.count("FP"), on_room=sum(1 for h in hit if h), repeat=repeat, **why))
    keys = ["확실해서 안 멈춤", "다른 곳에 멈춰 있었음", "헷갈렸는데 못 찾음", "위치추정 시작 전"]
    L = [f"{E.MAIN_NAME}. 판정 층(●)은 사람이 틀렸다고 한 멈춤을 뺐다. 놓친 이유는 그 호실을 가장 가까이 지난 순간의 상태로 나눴다. PILOT.", "",
         "| 층 | 지나간 호실 | 본 호실 | 멈춤 (호실 벽에 찍힘 / 틀림) | 같은 호실 재멈춤 | 놓침: 확실해서 안 멈춤 | 놓침: 다른 곳에 멈춰 있었음 | 놓침: 헷갈렸는데 못 찾음 | 놓침: 시작 전 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {NAME.get(r['F'], r['F'].replace('F', '층'))}{' ●' if r['judged'] else ''} | {r['passed']} | {r['seen']} | {r['stops']} ({r['on_room']}" +
                 (f" / {r['fp']}" if r['judged'] else "") + f") | {r['repeat']} | " + " | ".join(str(r.get(k, 0)) for k in keys) + " |")
    tot = {k: sum(r.get(k, 0) for r in rows) for k in keys}; miss = sum(tot.values())
    L += ["", "놓친 호실 합계 " + str(miss) + "개: " + ", ".join(f"{k} {v} ({100 * v / max(miss, 1):.0f}%)" for k, v in tot.items())]
    (OUT / "tables/E4_recall_why.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    json.dump(rows, open(OUT / "csv/E4_recall_why.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
