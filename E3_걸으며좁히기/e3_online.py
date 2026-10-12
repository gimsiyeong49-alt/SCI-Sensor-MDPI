#!/usr/bin/env python3
"""E3 기존 방식 조건(T3 + 축척·방향 온라인 추정, 2026-10-10). E4 때 협업자 위치추정을 돌린 방식(scripts/loc_topk_series.py --scale-online)과 같은 규칙.
  - 정합: 출발 뒤 30·60초에 협업자 metric_fit_v2.fit_window_v2(층 판별과 같은 함수, 축척 목록 15~43 px/m 에서 연속 보정, 방향·위치 동시 정합)를
    층 지도마다 '출발부터 지금까지' 창으로 돌린다(출발 전 자료는 안 씀 = 로봇을 막 켠 상황). 정답은 안 읽는다.
  - 추정: 그 시점까지 끝난 정합만으로 방향 = 비용 가중(1/max(비용, 5 cm)) 합이 가장 큰 E2 방향 후보, 축척 = 그 방향 정합들의 가중 중앙값
    (loc_topk_series.estimate 와 같음). 첫 정합 전(6·15초)은 기존처럼 축척 24(협업자 기본) + 방향 4개.
  - 점수: 그 축척·방향에서 e3_dense 의 T3 와 같은 식(출발~지금 시간 누적 + 공간 감점 + 벽 관통 경로 제거).
저장: /data2/e3_dense_20261010/T3on_T<T>/<과제>.npz (e2_dense 형식, 층마다 축척 1개·방향 1~4개). 정합 기록: <같은 폴더>/../T3on_fits/<과제>.json
    python3 scripts/e3_online.py [--tasks 4F_T1] [--workers 20]
"""
import argparse, csv, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3_dense as E  # noqa: E402

FITS = (30, 60)


def estimate(fits, T):
    f = [x for x in fits if x["ok"] and x["tf"] <= T]
    if not f:
        return None
    w = {}
    for x in f:
        w[x["mode"]] = w.get(x["mode"], 0.0) + 1.0 / max(x["cost"], 0.05)
    mode = max(w, key=w.get)
    g = sorted((x["ppm"], 1.0 / max(x["cost"], 0.05)) for x in f if x["mode"] == mode)
    half, acc = sum(v for _, v in g) / 2, 0.0
    for ppm, v in g:
        acc += v
        if acc >= half:
            return ppm, mode


def job(arg):
    tid, F, building, t_s, m = arg
    Z = E._loc(F, building, m); L = Z.L
    import node_local as NL, run_floor_id as R, floor_id as FI
    from metric_fit import set_scale
    from metric_fit_v2 import _maps, CROSS_MAX_M, TRAJ_DT, fit_window_v2
    from rt_localizer import _A
    if "TRAV_W0" not in E._ST:
        E._ST.update(TRAV_W0=NL.TRAV_W, FREE_W0=NL.FREE_W)
    t0 = time.time()
    meta2 = json.loads(str(np.load(E.E2SRC / f"{tid}.npz")["meta"]))["maps"][str(m)]
    Z.BETA, Z.MIRROR, Z.lam_sem = math.radians(E.BETA_MOUNT_DEG), 1, 0.0
    modes = [float(p) for p in meta2["modes"]]; Z.modes = modes; Z.phis = modes
    scales = FI.map_scales(E._ST["b"], R, int(E._ST["b"]["n_maps"]))[m]
    k_mid = [float(v) for v in R.SCALES].index(float(R.MID))
    fits = []
    for tf in FITS:
        try:
            r = fit_window_v2(Z, L.T[0] + t_s + tf, scales, floor=m, W=float(tf), dt=E.DT)
            phi_g = (r["phi"] + math.degrees(r["beta"] - Z.BETA)) % 360
            mode = min(modes, key=lambda p: abs((p - phi_g + 180) % 360 - 180))
            fits.append(dict(tf=tf, ok=True, ppm=float(r["ppm"]), phi=float(phi_g), mode=float(mode), cost=float(r["cost_m"])))
        except Exception as e:                       # 정합 실패는 기록하고 그 시점 추정에서 뺀다
            fits.append(dict(tf=tf, ok=False, err=str(e)[:200]))
        Z.BETA, Z.MIRROR = math.radians(E.BETA_MOUNT_DEG), 1  # fit_window_v2 가 되돌리지만 한 번 더 확인
    idx = np.flatnonzero(Z.allow.reshape(L.shp)); cells_px = L.Pf[idx].astype(float); wall_tr = _maps(Z, m)["wall_tr"]
    lp = {k: np.log(L.PRIOR[k] + 1e-6) for k in L.PRIOR}; zero = np.zeros(2); out = {}
    for T in E.TS:
        est = estimate(fits, T)
        ppm, phis = (est[0], [est[1]]) if est else (float(scales[k_mid]), modes)
        set_scale(Z, ppm); lst = E.frames_for(L, t_s, T); frs = []
        for req, i in lst:
            pts, yaw = L._scan(L.T[i])
            frs.append(dict(pts=pts, xy=(L.X[i], L.Y[i]), yaw=yaw, t=float(L.T[i]), sample_seed_t=req, sample_id=L.SID[i], scene=L.depth_scene(L.T[i] - L.T[0])))
        NL.TRAV_W, NL.FREE_W, Z.w_nogo = 0.0, 0.0, 0.0
        g = [{p: np.asarray(Z.localize_buf([dict(f)], phis, f["scene"], 1.0)["P"][p]) - L._shift(lp[f["scene"]], zero) for p in phis} for f in frs]
        NL.TRAV_W, NL.FREE_W, Z.w_nogo = E._ST["TRAV_W0"], E._ST["FREE_W0"], 3.0
        last = frs[-1]; anchor = np.asarray(last["xy"], float); open_m = L._obs_open(last["t"])
        fw = 0.0 if last["scene"] == "dead-end" else E._ST["FREE_W0"]; free = -fw * np.maximum(0, open_m * 0.6 - L.freeD_grid)
        sel = np.where((L.T >= frs[0]["t"]) & (L.T <= last["t"]))[0]; keep, lt = [], -1e9
        for i in sel:
            if L.T[i] - lt >= TRAJ_DT:
                keep.append(i); lt = L.T[i]
        track = np.c_[np.asarray(L.X)[keep], np.asarray(L.Y)[keep]] - anchor
        parts = []
        for pi, p in enumerate(phis):
            bel = np.zeros(L.shp); trav = np.zeros(L.shp)
            for j, f in enumerate(frs):
                od = np.asarray(f["xy"], float) - anchor; off = L.PPM * (_A(np.deg2rad(p)) @ od)
                bel += L._shift(g[j][p], off) + L._shift(lp[f["scene"]], off)
                px = np.clip(np.round(L.GXX + off[0]).astype(int), 0, L.W - 1); py = np.clip(np.round(L.GYY + off[1]).astype(int), 0, L.H - 1)
                trav += np.where(L.free[py, px], 0.0, -1.0)
            P = bel + NL.TRAV_W * trav + free - Z.w_nogo * Z.pen_g
            ex = E.crossing_m(wall_tr, cells_px, track, p, L.PPM) > CROSS_MAX_M
            parts.append((0, pi, np.where(ex, -np.inf, P.ravel()[idx]).astype(np.float32)))
        out[T] = dict(ppm=float(ppm), phis=[float(p) for p in phis], parts=parts, n_frames=len(frs), est=bool(est))
    set_scale(Z, R.MID)
    return dict(tid=tid, m=m, beta=float(Z.BETA), mirror=1, cell_px=L.Pf[idx].astype(np.float32), out=out, fits=fits, sec=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--workers", type=int, default=20); a = ap.parse_args()
    tasks = list(csv.DictReader(open(E.G / "tasks/E2_tasks.csv")))
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in a.tasks.split(",")]
    nmap = {"kci": 9, "gong": 7, "sw": 8}; OUT = E.OUT
    done = lambda tid: all((OUT / f"T3on_T{T}" / f"{tid}.npz").exists() for T in E.TS)
    jobs = [(t["task_id"], t["floor"], t["building"], float(t["start_sec"]), m) for t in tasks if not done(t["task_id"]) for m in range(1, nmap[t["building"]] + 1)]
    jobs.sort(key=lambda j: (j[2], j[1], j[4])); print(f"과제 {len({j[0] for j in jobs})}개, 작업 {len(jobs)}개", flush=True)
    t0 = time.time(); got = {}
    with Pool(a.workers, initializer=E.init) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            got.setdefault(r["tid"], []).append(r)
            t = [x for x in tasks if x["task_id"] == r["tid"]][0]
            if len(got[r["tid"]]) == nmap[t["building"]]:
                R_ = sorted(got.pop(r["tid"]), key=lambda x: x["m"])
                (OUT / "T3on_fits").mkdir(parents=True, exist_ok=True)
                (OUT / "T3on_fits" / f"{r['tid']}.json").write_text(json.dumps({x["m"]: dict(fits=x["fits"], sec=x["sec"],
                                                                         per_T={T: dict(ppm=x["out"][T]["ppm"], phis=x["out"][T]["phis"], estimated=x["out"][T]["est"]) for T in E.TS}) for x in R_}, ensure_ascii=False))
                for T in E.TS:
                    fl_, sc_, ph_, ce_, scs = [], [], [], [], []
                    for x in R_:
                        for si, pi, v in x["out"][T]["parts"]:
                            fl_.append(np.full(len(v), x["m"], np.uint8)); sc_.append(np.full(len(v), si, np.uint8)); ph_.append(np.full(len(v), pi, np.uint8))
                            ce_.append(np.arange(len(v), dtype=np.int32)); scs.append(v)
                    meta = dict(task=r["tid"], T=T, cond="T3on", maps={x["m"]: dict(beta=x["beta"], mirror=1, modes=x["out"][T]["phis"], scales=[x["out"][T]["ppm"]],
                                sec=x["sec"], n_frames=x["out"][T]["n_frames"], scale_estimated=x["out"][T]["est"]) for x in R_})
                    od = OUT / f"T3on_T{T}"; od.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(od / f"{r['tid']}.npz", floor=np.concatenate(fl_), scale_idx=np.concatenate(sc_), phi_idx=np.concatenate(ph_),
                                        cell=np.concatenate(ce_), score=np.concatenate(scs), meta=json.dumps(meta), **{f"cellpx_{x['m']}": x["cell_px"] for x in R_})
                print(f"  {r['tid']} 저장 ({time.time() - t0:.0f}s)", flush=True)
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
