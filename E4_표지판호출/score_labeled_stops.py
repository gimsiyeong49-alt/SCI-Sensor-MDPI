#!/usr/bin/env python3
"""사람 판정표(25_판정)로 실행 결과의 표지판 멈춤을 채점한다(2026-10-04).
판정표에 없는 멈춤(같은 시각 ±0.6초·상자 겹침 0.3 이상인 판정 항목이 없음)은 '미판정'으로 따로 센다.
4층은 정답 구간(148~178, 228~258초) 번호판 중심이 멈춤 상자 안이면 맞힘으로 센다.
    python3 scripts/score_labeled_stops.py 결과폴더 V0 V0B
"""
import json, sys
sys.path.insert(0, "scripts")
import sign_segsets as S
J = "reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/25_판정/"
M = json.load(open(J + "판정목록.json")); D = json.load(open(J + "판정결과.json"))
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
gt = {}
for sg in S.get("4F"):
    for l in open(sg["gt"]):
        d = json.loads(l); gt[int(d["t"])] = d
inside = lambda g, b: b[0] <= (g[0] + g[2]) / 2 <= b[2] and b[1] <= (g[1] + g[3]) / 2 <= b[3]
out_dir, runs = sys.argv[1], sys.argv[2:]
unl = {}
for v in runs:
    tot = [0, 0, 0, 0]
    for F in [f for f in D if not f.startswith("_")] + ["4F"]:
        try:
            E = json.load(open(f"{out_dir}/{F}_{v}/events_live_replay.json"))
        except FileNotFoundError:
            continue
        if F == "4F":
            found = set()
            for e in E:
                d = gt.get(int(round(e["t"])))
                if d:
                    found |= {o for (c, *g), o in zip(d["items"], d["object_id"]) if c == "plaque" and inside(g, e["picks"][0])}
            print(f"  {v} 4F: 멈춤 {len(E)}, 정답 구간 번호판 {len(found)}/38"); continue
        P = set(D[F]["plaque"]) | set(D[F]["sign"]); n = p = u = 0
        for e in E:
            b = e["picks"][0][:4]; n += 1
            m = [it for it in M[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], b) > 0.3]
            judged = set(D[F].get("judged", range(1, D[F].get("judged_max", 10 ** 9) + 1)))
            if not m or m[0]["id"] not in judged:              # 목록에는 있어도 아직 판정 안 받은 항목은 미판정
                u += 1; unl.setdefault(F, []).append(dict(t=e["t"], box=b, z=e["z"][0], run=v, id=m[0]["id"] if m else None)); continue
            p += m[0]["id"] in P
        tot[0] += n; tot[1] += p; tot[2] += u
        print(f"  {v} {F}: 멈춤 {n}, 번호판·표지 {p}, 미판정 {u}")
    print(f"{v} 판정 층 합: 멈춤 {tot[0]}, 번호판·표지 {tot[1]} ({tot[1] / max(tot[0] - tot[2], 1) * 100:.0f}% of 판정된 멈춤), 미판정 {tot[2]}")
json.dump(unl, open(f"{out_dir}/미판정_멈춤.json", "w"), ensure_ascii=False, indent=1)
