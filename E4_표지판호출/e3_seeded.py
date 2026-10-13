#!/usr/bin/env python3
"""E3 연결판(2026-10-11): 정지 때 만든 E2 Top-100(A 라이다·축척 탐색·+ 정밀 검증)을 이어받아, 걷는 동안 노드(floor_id)와 같은 후보 처리로 다시 매긴다.
노드 metric_fit_v2.fit_window_v2 에서 '후보 만들기'(축척마다 localize → 상위 3)만 E2 Top-100 으로 바꾸고, 나머지는 같은 식이다:
  · 창: 출발부터 지금까지(W = T), window_scans(dt 3초), 프레임마다 400점 표본(같은 씨앗), 구름 기준 = 지금 프레임
  · 미세 정합: least_squares(soft_l1, f_scale 0.2, 60회), 위치 ±1.5칸·방향 ±8°·축척 0.8~1.25배, 벽 속 보정 거리장(_maps F)
  · 채점: part = 비용(가까운 70%) + 0.5·지나온 길 막힌 칸 + 0.1·w_nogo·금지 + 1.5·T2 + 3.0·T3 + 1.0·min(X, 0.3), merit = part + 0.6·T4
    벽 관통 X(0.2초 조밀 경로) 0.3 m 넘으면 그 시점 순위에서 뺌(전부 넘으면 가장 적게 뚫은 것만), 같은 자세(1 m·10°·1.1배)는 하나로
  · 층 순위(노드식): 층마다 merit 가장 좋은 후보 → 지금 스캔으로 twoway → 내림차순
노드와 다른 점: 후보가 E2 Top-100(층마다 여러 개)이고 시점마다 자세를 이어받는다(층마다 1개로 줄이지 않음).
축척(v2, 2026-10-11): E2 정지 한 장 축척은 불안정해(정답 근처 후보도 중앙 21% 오차) 30% 틀린 축척으로 따라가면 길이 늘어나 벽 관통으로 빠졌다.
  그래서 시점마다 후보 하나에 초기값 두 개를 맞춰 보고 나은 쪽을 남긴다 — ① 같은 창에서 노드(floor_id)가 그 층에 대해 찾은 축척으로
  출발 자리부터 옮긴 자세(방향은 앞 시점 값), ② 앞 시점 자세를 그 축척 그대로 옮긴 것. 노드 축척은 같은 창 데이터만 쓴 값이라 인과적(정답 안 씀).
센서 각도·거울은 E2 와 같은 228°·1 로 고정(노드는 창마다 다시 추정). T4 는 노드처럼 상위 12개가 아니라 모든 후보에 계산.
협업자 코드는 고치지 않고 부른다. 정답은 --score 에만 쓴다.
저장: /data2/e3_seeded_v3_20261011/<과제>/<층지도>.json (v3: 축척 노드값 재시작 + 시작 프레임 + 층별 벽 관통 판정, 교차 검증 반영)
    python3 scripts/e3_seeded.py --tasks 4F_T1,S1F_T4 --workers 8
    python3 scripts/e3_seeded.py --score
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

NODE = Path(os.environ.get("E3_NODE_DIR", "/data2/e3_node_20261011"))
OUT = Path("/data2/e3_seeded_v3_20261011"); G = Path("reports/_01_E2_초기위치/_작업자료"); TS = (15, 30, 45, 60)
SEED = Path(os.environ.get("E3_SEED", str(G / "E2_dense_T0_nospace_정밀검증" / "E2_top100_candidates.csv")))
RES = Path("reports/_03_E3_걸으며좁히기/_작업자료/연결판")
BETA_DEG = 228.0
# 재탐색 합치기(2026-10-12, E3_ADD_NODE=1): 노드(floor_id)가 그 시점 창에서 새로 찾은 층별 후보를 연결판 후보에 더하고 이후 계속 이어받는다.
# E2 후보만 이어받으면 E2 에서 빠진 정답을 되찾을 수 없어서(새 계획서 4.5 '후보 누락 시 재탐색'). 모든 층 지도를 계산한다.
ADD_NODE = os.environ.get("E3_ADD_NODE") == "1"
NODE_EVERY = int(os.environ.get("E3_NODE_EVERY", "15"))   # 재탐색 간격(초). 노드 결과 중 이 배수 시점만 넣는다(15 = 15·30·45·60초)
NMAP = {"kci": 9, "gong": 7, "sw": 8}
if ADD_NODE:
    OUT = Path("/data2/e3_merged_20261012"); RES = Path("reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색")
OUT = Path(os.environ.get("E3_OUT", str(OUT))); RES = Path(os.environ.get("E3_RES", str(RES)))   # 2026-10-12: 지도 수정판 재실행용


def seeds():
    S = {}
    for r in csv.DictReader(open(SEED)):
        S.setdefault(r["trial_id"], []).append(dict(sid=int(r["rank"]), m=int(r["floor_id"]), x=float(r["x_px"]), y=float(r["y_px"]),
                                                   hd=float(r["heading_deg"]), s=float(r["ppm"])))
    return S


def job(arg):
    tid, F, building, t_s, m, cand, times, out_dir = arg
    f = Path(out_dir) / tid / f"{m}.json"
    if f.exists():
        return tid, m, "있음"
    import e2_dense as ED
    sys.path.insert(0, os.environ.get("KCI_FLOORID_PKG", "/data/kci_work/floor_id_pkg/floor_id"))   # 2026-10-12: 지도 수정판 패키지
    import floor_id as FI
    cfg = FI.load_cfg(); FI.setup_env(cfg, building)
    import rt_localizer as RT, run_floor_id as R, twoway as TW, metric_fit_negev as MN
    MN._door.clear()
    from rt_localizer import _A, _R
    from metric_fit_v2 import _maps, _sample, _crossing_m, W_FREE, W_OUT, W_MISS, W_CROSS, FAR_M, OUT_M, CROSS_MAX_M, TRAJ_DT
    from metric_fit_negev import unexplained_walls
    from node_local import GRID_STEP
    from scipy.optimize import least_squares
    t0 = time.time(); DT = float(cfg["consts"]["dt_s"])
    Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
    L = Z.L; Z.BETA, Z.MIRROR = math.radians(BETA_DEG), 1; M = _maps(Z, m); CV = float(max(L.H, L.W)); wall = getattr(L, "wall_real", L.wall)
    # 가설 상태: 지도 위치(지금 시점 기준), 방향 φ(rad, 지도 좌표), log 축척 — 시작은 E2 정밀 검증 결과(출발 시각)
    st = {c["sid"]: dict(x=np.array([c["x"], c["y"], math.radians(c["hd"] - BETA_DEG), math.log(c["s"])]), xy_ref=None, p0=np.array([c["x"], c["y"]])) for c in cand}
    node = json.load(open(NODE / tid / f"{m}.json"))["rows"]                # 같은 창 노드 정합(그 층) — 축척만 가져옴
    out = {}
    i0 = int(np.argmin(abs(L.T - (L.T[0] + t_s)))); xy_start = np.array([L.X[i0], L.Y[i0]], float)   # E2 정밀 검증 자세의 프레임(가장 가까운 스캔)과 같게
    for c in st.values():
        c["xy_ref"] = xy_start
    for T in times:
        t_now = L.T[0] + t_s + T; start = L.T[0] + t_s
        buf = Z.window_scans(t_now, float(T), DT); frames = np.array([fr["t"] for fr in buf]); anchor = np.asarray(buf[-1]["xy"], float)
        cloud, trajectory = [], []
        for fr in buf:
            pts, yaw = fr["pts"], fr["yaw"]
            if len(pts) > 400:
                pts = pts[np.random.RandomState(int(fr["sample_seed_t"]) % 97).choice(len(pts), 400, replace=False)]
            delta = np.asarray(fr["xy"]) - anchor
            cloud.append(delta + (pts * [1, Z.MIRROR]) @ _R(yaw + Z.BETA).T); trajectory.append(delta)
        cloud = np.vstack(cloud); trajectory = np.asarray(trajectory)
        if len(cloud) < 50:                                       # 노드와 같은 검사
            out[str(T)] = dict(n_frames=len(buf), rows=[], err="too few points"); continue
        sel = np.where((L.T >= start) & (L.T <= frames[-1]))[0]; keep, last = [], -1e9
        for i in sel:
            if L.T[i] - last >= TRAJ_DT:
                keep.append(i); last = L.T[i]
        dense = np.c_[np.asarray(L.X)[keep], np.asarray(L.Y)[keep]] - anchor
        if len(dense) < 2:
            dense = trajectory
        sample = cloud[np.linspace(0, len(cloud) - 1, min(1000, len(cloud))).astype(int)]
        project = lambda x, P: P @ _A(x[2]).T * np.exp(x[3]) + x[:2]
        dist = lambda x, P: _sample(M["F"], project(x, P), CV) / np.exp(x[3])
        pts_now, yaw_now = L._scan(t_now); pts_now = np.asarray(pts_now, float)
        rows = []
        tn = [int(k_) for k_ in node if int(k_) <= T and "ppm" in node[k_]]                   # 그 시점까지 나온 가장 최근 노드 축척(인과)
        s_node = float(node[str(max(tn))]["ppm"]) if tn else None
        def fit(x0):
            sh = 1.5 * GRID_STEP
            lo = x0 + [-sh, -sh, -np.deg2rad(8), np.log(.8)]; hi = x0 + [sh, sh, np.deg2rad(8), np.log(1.25)]
            return least_squares(lambda x: np.minimum(dist(x, sample), 5.), x0, bounds=(lo, hi), loss="soft_l1", f_scale=.2,
                                 diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60).x
        def part_of(x):
            px, py = np.rint(x[:2]).astype(int); s = float(np.exp(x[3])); ok = bool(0 <= px < L.W and 0 <= py < L.H and Z.ok_px[py, px])
            err = dist(x, cloud); cost = float(np.mean(np.sort(err)[:int(.7 * len(err))]))
            track = project(x, trajectory); ix, iy = np.rint(track).astype(int).T
            inside = (ix >= 0) & (ix < L.W) & (iy >= 0) & (iy < L.H); trav = np.zeros(len(ix), bool); trav[inside] = Z.ok_px[iy[inside], ix[inside]]
            blocked = float(1 - trav.mean()); nogo = float(Z.pen_px[py, px]) if ok else 1.0
            cpx = project(x, cloud); T2 = float(np.mean(err > FAR_M)); T3 = float(np.mean(_sample(M["out_d"], cpx, CV) / s > OUT_M))
            X = float(_crossing_m(M["wall_tr"], project(x, dense), s))
            part = cost + .5 * blocked + .1 * Z.w_nogo * nogo + W_FREE * T2 + W_OUT * T3 + W_CROSS * min(X, CROSS_MAX_M)
            return dict(ok=ok, cost=cost, blocked=blocked, nogo=nogo, T2=T2, T3=T3, X=X, part=float(part), cpx=cpx, track=track)
        if ADD_NODE and T % NODE_EVERY == 0 and str(T) in node and "pos" in node[str(T)]:          # 노드가 이 창에서 찾은 후보를 새 가설로(그 층 하나)
            nr = node[str(T)]; ph = math.radians(float(nr["phi"]) + float(nr["beta_deg"]) - BETA_DEG); s_ = float(nr["ppm"]); p_ = np.array(nr["pos"], float)
            st[f"N{T}"] = dict(x=np.array([p_[0], p_[1], ph, math.log(s_)]), xy_ref=anchor.copy(), p0=p_ - s_ * (_A(ph) @ (anchor - xy_start)))
        for sid, c in st.items():
            x_prev = c["x"]; tries = []
            off = math.exp(x_prev[3]) * (_A(x_prev[2]) @ (anchor - c["xy_ref"]))                    # ② 앞 시점 자세를 걸은 만큼 옮김
            tries.append(np.r_[x_prev[:2] + off, x_prev[2], x_prev[3]])
            if s_node:                                                                                # ① 노드 축척으로 출발 자리부터 옮김
                tries.append(np.r_[c["p0"] + s_node * (_A(x_prev[2]) @ (anchor - xy_start)), x_prev[2], math.log(s_node)])
            best = None
            for x0 in tries:
                xf = fit(x0); q = part_of(xf); key = (not q["ok"], q["X"] > CROSS_MAX_M, q["part"])
                if best is None or key < best[0]:
                    best = (key, xf, q)
            _, x, q = best
            c["x"], c["xy_ref"] = x.copy(), anchor
            px, py = np.rint(x[:2]).astype(int); s = float(np.exp(x[3]))
            ok, cost, blocked, nogo, T2, T3, X, part, cpx, track = (q[k_] for k_ in ("ok", "cost", "blocked", "nogo", "T2", "T3", "X", "part", "cpx", "track"))
            t4 = float(unexplained_walls(M["wall"], M["door"], cpx, track, s)[0])
            hd_abs = x[2] + yaw_now + Z.BETA
            tw = float(TW.twoway_score(wall, x[0], x[1], s, TW.sector_obs(pts_now * [1, Z.MIRROR], hd_abs))[0])
            rows.append(dict(sid=sid, m=m, x=float(x[0]), y=float(x[1]), hd=float((math.degrees(x[2]) + BETA_DEG) % 360), s=s, ok=ok,
                             cost=cost, blocked=blocked, nogo=nogo, T2=T2, T3=T3, X=X, T4=t4, part=float(part), merit=float(part + W_MISS * t4), twoway=tw))
        out[str(T)] = dict(n_frames=len(buf), rows=rows)
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(task=tid, map=m, t_s=t_s, out=out, sec=round(time.time() - t0, 1))))
    return tid, m, round(time.time() - t0, 1)


def run(tasks, workers, times=TS, out_dir=None):
    out_dir = str(out_dir or OUT)
    T_ = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; S = seeds(); jobs = []
    for k in tasks:
        by = {}
        for c in S.get(k, []):
            by.setdefault(c["m"], []).append(c)
        if ADD_NODE:
            for m_ in range(1, NMAP[T_[k]["building"]] + 1):
                by.setdefault(m_, [])
        for m, cs in sorted(by.items()):
            jobs.append((k, T_[k]["floor"], T_[k]["building"], float(T_[k]["start_sec"]), m, cs, list(times), out_dir))
    jobs.sort(key=lambda j: (j[2], j[1], j[0], j[4]))
    print("과제", len(tasks), "작업", len(jobs), flush=True); t0 = time.time()
    from multiprocessing import Pool
    with Pool(workers, maxtasksperchild=4) as pool:
        for tid, m, s in pool.imap_unordered(job, jobs):
            print(f"  {tid} {m}층 {s}s  ({time.time() - t0:.0f}s)", flush=True)


def score():
    import e2_visual_ko as EV
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    S = seeds(); wrap = lambda a: (a + 180) % 360 - 180; rows = []; cands_out = []
    for k, t in EV.TASK.items():
        need = set(range(1, NMAP[t["building"]] + 1)) if ADD_NODE else {c["m"] for c in S.get(k, [])}
        d = OUT / k
        if not need or not d.exists() or {int(f.stem) for f in d.glob("*.json")} != need:
            continue
        J = [json.load(open(f)) for f in d.glob("*.json")]; m_true = int(EV.GT[k]["true_map"]); ts = float(t["start_sec"]); _, ppm, hd_gt, gt, gx = EV.gt_of(k)
        for T in TS:
            feas = []
            for j in J:                                                       # 층마다: 0.3 m 안만, 전부 넘으면 그 층에서 가장 덜 뚫은 것(노드와 같음)
                C = [r for r in j["out"][str(T)]["rows"] if r["ok"]]
                if C:
                    feas += [c for c in C if c["X"] <= 0.3] or [c for c in C if c["X"] <= min(x["X"] for x in C) + 1e-9]
            feas.sort(key=lambda c: c["merit"]); picked = []
            for c in feas:                                                   # 같은 자세(1 m·10°·1.1배)는 하나로 — 노드와 같은 기준
                if any(c["m"] == p["m"] and math.hypot(c["x"] - p["x"], c["y"] - p["y"]) / math.sqrt(c["s"] * p["s"]) < 1.0 and abs(wrap(c["hd"] - p["hd"])) < 10
                       and abs(math.log(c["s"] / p["s"])) < math.log(1.1) for p in picked):
                    continue
                picked.append(c)
            for i, c in enumerate(picked, 1):
                cands_out.append(dict(trial_id=k, T=T, rank=i, floor_id=c["m"], x_px=round(c["x"], 1), y_px=round(c["y"], 1), heading_deg=round(c["hd"], 1),
                                      ppm=round(c["s"], 3), merit=round(c["merit"], 4), twoway=round(c["twoway"], 4), sid=c["sid"]))
            pg = gx[int(np.argmin(abs(gt - (ts + T))))]
            okp = lambda c: c["m"] == m_true and math.hypot(c["x"] - pg[0], c["y"] - pg[1]) / ppm <= 3.0 and abs(wrap(c["hd"] - hd_gt)) <= 30
            tr = next((i + 1 for i, c in enumerate(picked) if okp(c)), None)
            best = {}
            for c in picked:                                                  # 층마다 merit 1등 → twoway 로 층 순위(노드식)
                best.setdefault(c["m"], c)
            order = sorted(best.values(), key=lambda c: -c["twoway"]); fr = next((i + 1 for i, c in enumerate(order) if c["m"] == m_true), None)
            tf = best.get(m_true)
            rows.append(dict(trial_id=k, floor=t["floor"], building=t["building"], T=T, n_left=len(picked), true_rank=tr if tr else "",
                             top1_ok=int(bool(tr == 1)), top10_ok=int(bool(tr and tr <= 10)), top100_ok=int(bool(tr and tr <= 100)),
                             floor_rank_node=fr if fr else "", node_top1_ok=int(bool(order) and okp(order[0])), truefloor_ok=int(bool(tf) and okp(tf)),
                             truefloor_pos_err_m=round(math.hypot(tf["x"] - pg[0], tf["y"] - pg[1]) / ppm, 2) if tf else ""))
    RES.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("채점할 결과 없음"); return
    with open(RES / "연결판_채점.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(RES / "연결판_남은후보.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cands_out[0])); w.writeheader(); w.writerows(cands_out)
    print("출발점", len({r["trial_id"] for r in rows}))
    for T in TS:
        rr = [r for r in rows if r["T"] == T]
        if rr:
            print(f"  {T}초({len(rr)}곳): Top-1 {sum(r['top1_ok'] for r in rr)} · Top-10 {sum(r['top10_ok'] for r in rr)} · Top-100 {sum(r['top100_ok'] for r in rr)} · "
                  f"노드식 층 1위 {sum(r['floor_rank_node'] == 1 for r in rr)} · 노드식 1위 위치 {sum(r['node_top1_ok'] for r in rr)} · 정답 층 후보 위치 {sum(r['truefloor_ok'] for r in rr)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--score", action="store_true")
    ap.add_argument("--times", default="", help="계산 시점(초), 기본 15,30,45,60. E4 입력용은 3,6,...,60"); ap.add_argument("--out", default="", help="출력 폴더(기본 OUT)")
    a = ap.parse_args()
    if a.score:
        return score()
    times = [int(x) for x in a.times.split(",")] if a.times else list(TS)
    run(a.tasks.split(",") if a.tasks else [t["task_id"] for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))], a.workers, times, a.out or None)


if __name__ == "__main__":
    main()
