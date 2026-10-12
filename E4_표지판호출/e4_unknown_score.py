#!/usr/bin/env python3
"""층 모름 E4 채점(2026-10-11, PILOT). 65개 출발점 60초 구간 실행(70_층모름E4/<정책>/<출발점>/run)을 정책별로 비교한다.
정답(채점용 정답 36_, 사람 판정 25_)은 여기서만 읽는다.
  정책: P_ALL(애매 기록 없이), P_AMBIGUITY(Δ=0.2 애매 기록), P_INFORMATIVE(P_AMBIGUITY 멈춤 중 e4_unknown_informative.py 가 가를 수 있다고 한 것)
        Δ 0.1·0.4 는 P_ALL 멈춤을 그 애매 기록으로 거른 근사(다시 돌리지 않음, 거른 대신 다른 곳에 멈췄을 가능성 반영 안 함)
  출발점마다: 번호판 멈춤 수, 계단 관측 수
  번호판 멈춤 중 호실 앞: 멈춘 3D 점을 정답 자세로 정답 층 지도에 찍어 호실 벽 1.5 m 안(e4_recall.py 와 같은 식). 판정 층(2·7층, 공학관 1층,
        소융대 1층)은 사람이 표지판이라고 한 멈춤만 셈(판정 목록과 시각 0.6초·상자 IoU 0.3 으로 맞춤, 안 맞으면 '미판정')
  번호 읽을 기회 = 60초 안에 호실 앞 번호판 멈춤이 한 번이라도 있었나, 첫 멈춤까지 걸린 시간
  번호를 읽었다면(E6 미리 보기, 상한): 첫 호실 앞 멈춤의 번호를 맞게 읽어 층이 정해졌다고 치고, 60초 때 위치 = 연결판의 정답 층 최고 후보
        (정답 층·3 m·30°). 기회가 없으면 연결판 1위 그대로.
    python3 scripts/e4_unknown_score.py
"""
import csv, json, math, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).parent))
from obs_feedback import A, poly_mask  # noqa: E402

import e4_unknown_gate as _UG  # noqa: E402
SFX = _UG.SFX
R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); RUN = R / "70_층모름E4"; OUT = RUN / f"채점{SFX}"
D = Path("/data/kci_work/maps_perfloor_20261001"); J = R / "25_판정"; L = R / "35_위치추정_축척추정"
W = Path("/data2/e4_unknown_20261011"); SEEDED = _UG.SEEDED
G = Path("reports/_01_E2_초기위치/_작업자료")
PASS_M, HIT_M = 3.0, 1.5
wrap = lambda a: (a + 180) % 360 - 180


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def floor_rooms(F):
    im = cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE); H, Wd = im.shape
    sem = json.load(open(D / F / f"{F}_semantic_map.json"))
    rooms = [s for s in sem["spaces"] if s.get("type") == "room" and s.get("polygon_px")]
    corr = poly_mask((H, Wd), [c.get("polygon_px") for c in sem.get("corridors", [])])
    DTc = distance_transform_edt(~corr) if corr.any() else np.zeros((H, Wd))
    out = []
    for r in rooms:
        m = poly_mask((H, Wd), [r["polygon_px"]]); e = m & cv2.dilate((~m).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.where(e); out.append(dict(id=r["id"], num=r.get("room_number") or r["id"], edge=np.c_[xs, ys].astype(float), DTc=DTc[ys, xs]))
    return out


def seeded_at60(k, m_true, ts, ppm, hd_gt, gt, gx):
    """연결판(3초판) 60초: 1위 맞음, 정답 층 최고 후보 맞음 — e3_seeded.score 와 같은 거름."""
    import e4_unknown_gate as UG
    J_ = [json.load(open(p)) for p in (SEEDED / k).glob("*.json")]
    P = UG.hypotheses(J_, 60)
    pg = gx[int(np.argmin(abs(gt - (ts + 60))))]
    ok = lambda c: c["m"] == m_true and math.hypot(c["x"] - pg[0], c["y"] - pg[1]) / ppm <= 3.0 and abs(wrap(c["hd"] - hd_gt)) <= 30
    tf = next((c for c in P if c["m"] == m_true), None)
    return bool(P and ok(P[0])), bool(tf and ok(tf))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Mj = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json"))
    extra = J / "층모름합치기_판정기록.json"                          # 2026-10-12 층 모름 출발점 구간 멈춤 사람 판정(13개 층) — 기존 판정에 더한다
    if extra.exists():
        X = json.load(open(extra))
        for F, ids in X["시트"].items():
            d = Dj.setdefault(F, {})
            old = set(d.get("judged", range(1, d.get("judged_max", 0) + 1)))
            d["judged"] = sorted(old | set(ids)); d["sign"] = sorted(set(d.get("sign", [])) | set(X["맞음"][F]))
    segs = {s["task_id"]: s for s in csv.DictReader(open(W / "segments.csv"))}
    GTt = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    pols = {"P_ALL": RUN / "P_ALL", "P_AMBIGUITY": RUN / f"P_AMBIGUITY_gate{SFX}_d0.2"}   # P_ALL 은 애매 기록을 안 써서 후보 판과 무관
    INF = RUN / f"P_INFORMATIVE{SFX}_d0.2"                                   # e4_unknown_informative.py 결과(<출발점>.json)
    gates = {d: W / f"gate{SFX}_d{d}" for d in ("0.1", "0.2", "0.4")}
    rooms_c, rows = {}, []
    for k, sg in segs.items():
        F = sg["floor"]; ts = float(sg["start_sec"]); m_true = int(GTt[k]["true_map"])
        if F not in rooms_c:
            rooms_c[F] = floor_rooms(F)
        S = json.load(open(R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]; hd_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
        g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gt = np.array([float(x["relative_sec"]) for x in g]); gx = np.array([[float(x["x_px"]), float(x["y_px"])] for x in g])
        Mg = A(math.radians(S["theta_deg"])) * ppm if S["hand"] > 0 else None
        tl = list(csv.DictReader(open(L / F / "loc/timeline.csv"))); tt = np.array([float(x["relative_sec"]) for x in tl])
        od = np.array([[float(x["odom_x_m"]), float(x["odom_y_m"])] for x in tl])
        lo, hi = float(sg["seg_lo"]), float(sg["seg_hi"])
        sel = (gt >= lo) & (gt <= hi); path = gx[sel]
        passed = {}
        for r in rooms_c[F]:
            cw = r["edge"][r["DTc"] / ppm <= 1.0]
            if len(cw) and len(path):
                cw = cw[::max(1, len(cw) // 400)]
                if np.min(np.linalg.norm(path[::3, None] - cw[None], axis=2)) / ppm <= PASS_M:
                    passed[r["id"]] = r
        judged = F in Dj
        Pset = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", [])) if judged else set()
        jd = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1))) if judged else set()

        def room_of(e):
            if not e.get("xyz") or Mg is None:
                return None
            i = int(np.argmin(abs(gt - e["t"]))); oo = od[int(np.argmin(abs(tt - e["t"])))]
            mp = gx[i] + Mg @ (np.array(e["xyz"][:2]) - oo); best = None
            for rid, r in passed.items():
                dd = np.min(np.linalg.norm(r["edge"] - mp, axis=1)) / ppm
                if dd <= HIT_M and (best is None or dd < best[0]):
                    best = (dd, rid)
            return best[1] if best else None

        def label(e):                                                 # 판정 층: 'tp'/'fp'/'미판정', 아니면 None
            if not judged:
                return None
            m = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], e["picks"][0][:4]) > 0.3]
            if not m or m[0]["id"] not in jd:
                return "미판정"
            return "tp" if m[0]["id"] in Pset else "fp"

        top1_ok, tf_ok = seeded_at60(k, m_true, ts, ppm, hd_gt, gt, gx)
        ev_all = {}
        for name, base in pols.items():
            run = base / k / "run"
            if not (run / "events_live_replay.json").exists():
                continue
            ev_all[name] = (json.load(open(run / "events_live_replay.json")), [e for e in json.load(open(run / "facility_events.json")) if e.get("label") == "계단"])
        if (INF / f"{k}.json").exists() and "P_AMBIGUITY" in ev_all:
            keep = {(round(x["t"], 3), x["kind"]) for x in json.load(open(INF / f"{k}.json")) if x["informative"]}
            E, Fe = ev_all["P_AMBIGUITY"]
            ev_all["P_INFORMATIVE"] = ([e for e in E if (round(float(e["t"]), 3), "번호판") in keep], [e for e in Fe if (round(float(e["t"]), 3), "계단") in keep])
        if "P_ALL" in ev_all:                                         # Δ 민감도: P_ALL 멈춤을 애매 기록으로 거른 근사
            for d, gd in gates.items():
                gr = {float(x["t"]): bool(x["amb"] and not x["sensor_bad"]) for x in json.load(open(gd / f"{k}.json"))}
                E, Fe = ev_all["P_ALL"]
                ev_all[f"근사_Δ{d}"] = ([e for e in E if gr.get(float(e["t"]), False)], [e for e in Fe if gr.get(float(e["t"]), False)])
        for name, (E, Fe) in ev_all.items():
            labs = [label(e) for e in E]
            good = []
            for e, lb in zip(E, labs):
                rid = room_of(e)
                if rid is not None and (lb is None or lb == "tp"):
                    good.append((float(e["t"]), rid))
            t_first = min((t for t, _ in good), default=None)
            rows.append(dict(trial_id=k, floor=F, building=sg["building"], policy=name, judged=int(judged), plate_stops=len(E), stair_obs=len(Fe),
                             judged_tp=sum(lb == "tp" for lb in labs), judged_fp=sum(lb == "fp" for lb in labs), unjudged=sum(lb == "미판정" for lb in labs),
                             rooms_passed=len(passed), rooms_seen=len({r for _, r in good}), chance=int(bool(good)),
                             first_good_sec=round(t_first - lo, 1) if t_first is not None else "",
                             top1_ok_60=int(top1_ok), truefloor_ok_60=int(tf_ok), oracle_ok_60=int(tf_ok if good else top1_ok)))
    with open(OUT / "출발점별_정책별.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    order = ["P_ALL", "P_AMBIGUITY", "P_INFORMATIVE", "근사_Δ0.1", "근사_Δ0.2", "근사_Δ0.4"]
    lines = ["| 정책 | 출발점 | 번호판 멈춤(출발점당) | 계단 | 판정 층 정밀도 | 미판정 | 본 호실 / 지나간 호실 | 번호 읽을 기회 | 첫 기회까지(중앙) | 번호를 읽었다면 60초 위치 맞음 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in order:
        rr = [r for r in rows if r["policy"] == p]
        if not rr:
            continue
        n = len(rr); tp = sum(r["judged_tp"] for r in rr); fp = sum(r["judged_fp"] for r in rr)
        fg = [r["first_good_sec"] for r in rr if r["first_good_sec"] != ""]
        lines.append(f"| {p} | {n} | {sum(r['plate_stops'] for r in rr)} ({sum(r['plate_stops'] for r in rr) / n:.1f}) | {sum(r['stair_obs'] for r in rr)} | "
                     f"{100 * tp / max(tp + fp, 1):.0f}% ({tp}/{tp + fp}) | {sum(r['unjudged'] for r in rr)} | {sum(r['rooms_seen'] for r in rr)}/{sum(r['rooms_passed'] for r in rr)} | "
                     f"{sum(r['chance'] for r in rr)}/{n} | {np.median(fg) if fg else float('nan'):.0f}초 | {sum(r['oracle_ok_60'] for r in rr)}/{n} |")
    rr = [r for r in rows if r["policy"] == "P_ALL"]
    lines.append(f"\n기준: 이벤트 없이 연결판 60초 1위 맞음 {sum(r['top1_ok_60'] for r in rr)}/{len(rr)}, 층을 알 때(정답 층 최고 후보) {sum(r['truefloor_ok_60'] for r in rr)}/{len(rr)}")
    txt = "\n".join(lines); print(txt); (OUT / "정책비교.md").write_text(txt + "\n")
    states(segs)


def states(segs, delta="0.2", div_th=0.10):
    """매 장면 상태 4종(지침서 14.1): 센서 이상 → SENSOR_UNRELIABLE, 애매 + 최근 3초 벽 방향 다양성 중앙 < 0.1 → GEOMETRIC_INSUFFICIENT,
    애매 → GEOMETRIC_AMBIGUITY, 그 밖 → GEOMETRIC_CONFIDENT. 벽 방향 다양성은 층 아는 판(e4_paper.py)과 같은 표(geom_cells_<층>.csv)."""
    GC = R / "59_E4_호출정책" / "paper_C" / "csv"; cache = {}; rows = []
    for k, sg in segs.items():
        F = sg["floor"]; off = float(sg["off"])
        if F not in cache:
            gc = list(csv.DictReader(open(GC / f"geom_cells_{F}.csv")))
            cache[F] = (np.array([float(r["relative_sec"]) for r in gc]), np.array([float(r["normal_diversity"]) if r["normal_diversity"] else np.nan for r in gc]))
        gt_t, gdv = cache[F]
        for x in json.load(open(W / f"gate{SFX}_d{delta}" / f"{k}.json")):
            t = float(x["t"]) - off                                   # 장면 시각 → 오돔 첫 샘플 기준
            m = (gt_t > t - 3.0) & (gt_t <= t + 1e-6) & np.isfinite(gdv); div = float(np.median(gdv[m])) if m.any() else None
            c = ("SENSOR_UNRELIABLE" if x["sensor_bad"] else ("GEOMETRIC_INSUFFICIENT" if x["amb"] and div is not None and div < div_th
                 else ("GEOMETRIC_AMBIGUITY" if x["amb"] else "GEOMETRIC_CONFIDENT")))
            rows.append(dict(trial_id=k, scene_sec=x["t"], state=c, amb_src=x.get("amb_src", ""), n_floors=len(x.get("floors", [])), normal_diversity="" if div is None else round(div, 4)))
    with open(OUT / f"매초상태4종_d{delta}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows); import collections
    cnt = collections.Counter(r["state"] for r in rows); src = collections.Counter(r["amb_src"] for r in rows if r["state"] != "GEOMETRIC_CONFIDENT")
    nf = [r["n_floors"] for r in rows if r["amb_src"] == "score"]
    txt = (f"\n매초 상태 4종(Δ{delta}, 65곳 {n}초): " + ", ".join(f"{c} {cnt[c]}초({100 * cnt[c] / n:.1f}%)" for c in
           ("GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT", "SENSOR_UNRELIABLE")) +
           f"\n애매 이유: 출발 직후 {src['start']}초, 점수 경쟁 {src['score']}초, 1위 이동 뒤 {src['switch']}초 | 점수 경쟁일 때 Δ 안 가설의 층 수 중앙 {np.median(nf):.0f}")
    print(txt); open(OUT / "정책비교.md", "a").write(txt + "\n")


if __name__ == "__main__":
    main()
