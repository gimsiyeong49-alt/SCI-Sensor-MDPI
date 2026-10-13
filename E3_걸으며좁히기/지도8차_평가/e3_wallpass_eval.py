#!/usr/bin/env python3
"""E3 반대편 통과 판정 전후 비교(2026-10-13). 입력 = scripts/e3_wallpass_post.py 가 판정을 붙인 합치기 결과.
  · 합치기 순위(E3 본 결과): 층마다 X ≤ 0.3(전부 넘으면 가장 덜 뚫은 것) → merit 순, 같은 자세 하나로. 판정 후 = 통과 1번 이상인 후보를 뒤로
  · 로봇식 층 순위: 층마다 merit 최고 → twoway 순. 판정 후 = 최고 후보가 통과인 층을 뒤로(모든 층이 통과면 그대로, realtime/loc_worker.py 와 같음)
  · 안전성: 위치까지 맞는 정답 후보 중 통과로 걸린 수
정답은 채점에만 쓴다.
    python3 scripts/e3_wallpass_eval.py
출력: reports/_11_E2E3_지도v8_평가_20261012/E3/반대편통과_전후.csv, 반대편통과_바뀐곳.csv
"""
import csv, glob, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

D = Path("/data2/e3_merged_v8_wp_20261013"); G = Path("reports/_01_E2_초기위치/_작업자료")
OUT = Path("reports/_11_E2E3_지도v8_평가_20261012/E3"); TS = ("15", "30", "45", "60")
wrap = lambda a: (a + 180) % 360 - 180


def feas_of(rows):
    C = [r for r in rows if r["ok"]]
    return ([c for c in C if c["X"] <= 0.3] or [c for c in C if c["X"] <= min(x["X"] for x in C) + 1e-9]) if C else []


def picked(L):
    out = []
    for c in L:
        if any(c["m"] == p["m"] and math.hypot(c["x"] - p["x"], c["y"] - p["y"]) / math.sqrt(c["s"] * p["s"]) < 1.0 and abs(wrap(c["hd"] - p["hd"])) < 10
               and abs(math.log(c["s"] / p["s"])) < math.log(1.1) for p in out):
            continue
        out.append(c)
    return out


def main():
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    S = {T: dict(a1=0, b1=0, a10=0, b10=0, ain=0, bin=0, fa=0, fb=0, pa=0, pb=0) for T in TS}
    safe = dict(정답후보=0, 정답후보_통과=0, 순위쓰는정답후보=0, 순위쓰는정답후보_통과=0, 전체행=0, 전체행_통과=0, 순위쓰는행=0, 순위쓰는행_통과=0)
    ch = []
    for k in sorted(EV.TASK):
        _, ppm, hdg, gt, gx = EV.gt_of(k); mt = int(GT[k]["true_map"]); ts = float(EV.TASK[k]["start_sec"])
        J = [json.load(open(f)) for f in sorted(glob.glob(str(D / k / "*.json")))]
        for T in TS:
            pg = gx[int(np.argmin(abs(gt - (ts + float(T)))))]
            okp = lambda c: c["m"] == mt and math.hypot(c["x"] - pg[0], c["y"] - pg[1]) / ppm <= 3 and abs(wrap(c["hd"] - hdg)) <= 30
            feas, best = [], {}
            for j in J:
                rows = j["out"][T]["rows"]; F = feas_of(rows); feas += F
                for r in rows:
                    safe["전체행"] += 1; safe["전체행_통과"] += r["wp"] > 0
                    if okp(r):
                        safe["정답후보"] += 1; safe["정답후보_통과"] += r["wp"] > 0
                for r in F:
                    safe["순위쓰는행"] += 1; safe["순위쓰는행_통과"] += r["wp"] > 0
                    if okp(r):
                        safe["순위쓰는정답후보"] += 1; safe["순위쓰는정답후보_통과"] += r["wp"] > 0
                if F:
                    b = min(F, key=lambda c: c["merit"]); best[b["m"]] = b
            A = picked(sorted(feas, key=lambda c: c["merit"])); B = picked(sorted(feas, key=lambda c: (c["wp"] > 0, c["merit"])))
            ra = next((i + 1 for i, c in enumerate(A) if okp(c)), None); rb = next((i + 1 for i, c in enumerate(B) if okp(c)), None)
            allp = all(c["wp"] > 0 for c in best.values())
            FA = sorted(best.values(), key=lambda c: -c["twoway"]); FB = sorted(best.values(), key=lambda c: (c["wp"] > 0 and not allp, -c["twoway"]))
            R = S[T]
            R["a1"] += ra == 1; R["b1"] += rb == 1; R["a10"] += bool(ra and ra <= 10); R["b10"] += bool(rb and rb <= 10)
            R["ain"] += ra is not None; R["bin"] += rb is not None
            R["fa"] += FA[0]["m"] == mt; R["fb"] += FB[0]["m"] == mt; R["pa"] += okp(FA[0]); R["pb"] += okp(FB[0])
            if ra != rb:
                ch.append(dict(출발점=k, 시점초=T, 구분="합치기 순위", 전=f"정답 {ra}위" if ra else "정답 없음", 후=f"정답 {rb}위" if rb else "정답 없음", 정답층=mt))
            if FA[0]["m"] != FB[0]["m"]:
                ch.append(dict(출발점=k, 시점초=T, 구분="로봇식 층 순위", 전=f"1위 {FA[0]['m']}층", 후=f"1위 {FB[0]['m']}층", 정답층=mt))
    tab = [dict(시점초=T, 합치기_1위정답_전=R["a1"], 합치기_1위정답_후=R["b1"], 합치기_상위10_전=R["a10"], 합치기_상위10_후=R["b10"],
                합치기_후보안정답_전=R["ain"], 합치기_후보안정답_후=R["bin"], 로봇식_층1위_전=R["fa"], 로봇식_층1위_후=R["fb"],
                로봇식_1위위치_전=R["pa"], 로봇식_1위위치_후=R["pb"]) for T, R in S.items()]
    with open(OUT / "반대편통과_전후.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)
    with open(OUT / "반대편통과_바뀐곳.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["출발점", "시점초", "구분", "전", "후", "정답층"]); w.writeheader(); w.writerows(ch)
    for t in tab:
        print(t)
    print(safe)
    for c in ch:
        print(c)


if __name__ == "__main__":
    main()
