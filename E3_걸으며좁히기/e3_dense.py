#!/usr/bin/env python3
"""E3 걸으며 후보 좁히기 점수판(2026-10-10). E2 원후보 전체(모든 층 × 갈 수 있는 칸 × 방향 4 × 축척 7)를 FAST-LIO 이동으로 따라가며 시점마다 점수를 낸다.
협업자 위치추정(rt_localizer.RTLocalizer, 의미 가중치 0)을 고치지 않고 부른다. 정답은 읽지 않는다(채점은 e2_dense_score.py --src-dir).

E2 후보를 다시 만들지 않는다: 장착 각도 228°, 거울 1, 방향 후보 4개는 E2(T0_nospace) 저장값 그대로, 축척 목록도 같다.
시점 T(출발 뒤 6·15·30·45·60초)는 한 출발점 안의 중간 평가 시점이다. 관측은 협업자 창 고르기(window_indices, 3초 간격)를 그대로 써서
출발 시각 t_s 부터 t_s+T 까지. T 를 3의 배수로 둔 것은 첫 관측이 E2 관측(t_s)과 같아야 하기 때문이다.

계산: 스캔 하나의 점수판(벽 맞추기 g)을 프레임마다 한 번 구하고(협업자 localize_buf 를 한 프레임으로 불러 g = P - 장면사전),
시점·조건마다 협업자 localize_buf 와 같은 순서·같은 식(L._shift 이동, 장면사전, 갈 수 없는 칸, 빈 공간, 금지 구역)으로 더한다.
--verify 로 협업자 localize(t_s+T, W=T) 와 숫자가 같은지 확인한다.

조건(ablation_config: 70쪽 아래 COND 주석과 README):
  T1      이동만: 출발 관측 + 지금 관측 두 장, 물리 조건 없음
  T2      시간 누적: 출발~지금 3초마다 관측 전부, 물리 조건 없음
  T2X     T2 + 벽 관통 경로 제거: 그 후보 자리로 옮긴 주행 경로(0.2초 간격)가 문 자리를 뺀 벽을 0.3 m 넘게 지나면 후보에서 뺌
          (협업자 metric_fit_v2 의 X 항과 같은 정의·기준: _maps 의 wall_tr, 1 px 조밀화, 지도 밖도 관통)
  T2S     T2 + 협업자 공간 감점(갈 수 없는 칸 TRAV_W 0.8·빈 공간 FREE_W 2.0·금지 구역 w_nogo 3.0)
  T3      T2 + 벽 관통 경로 제거 + 공간 감점 (지침서 T3_PHYSICAL)
  T2d,T3d 장면 판단을 깊이로(E2 의 B·C 계열 비교용, 깊이 영상은 e2_scene_labels 와 같은 식)
저장(/data2/e3_dense_20261010/<조건>_T<T>/<과제>.npz): e2_dense.py 와 같은 형식(floor, scale_idx, phi_idx, cell, score, meta, cellpx_<층>).
    python3 scripts/e3_dense.py --verify --tasks 4F_T1          # 협업자 누적과 같은지(4층 지도·축척 24·T=15)
    python3 scripts/e3_dense.py [--tasks 4F_T1,...] [--workers 20]
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

PKG = Path("/data/kci_work/floor_id_pkg/floor_id")
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(Path(__file__).parent))
G = Path("reports/_01_E2_초기위치/_작업자료")
E2SRC = Path("/data2/e2_dense_20261009/T0_nospace")
OUT = Path("/data2/e3_dense_20261010")
DATA = {"4F": "/data/kci_work/a131_full_v0_4"}
TS = (6, 15, 30, 45, 60)                     # 모든 조건을 저장하는 시점
TS_FINE = tuple(range(3, 61, 3))             # 주요 조건(T2·T3)은 3초마다 20개 시점(후보가 줄어드는 과정, 2026-10-10 사용자 요청)
COND_FINE = ("T2", "T3")
TS_ALL = tuple(sorted(set(TS) | set(TS_FINE)))
DT = 3.0
COND = ("T1", "T2", "T2X", "T2S", "T3", "T2d", "T3d")
COND_SAVE = COND
BETA_MOUNT_DEG = 228.0
_ST = {}


def init():
    _ST.update(cache={})


def _loc(F, building, m):
    import floor_id as FI
    if _ST.get("building") != building:
        cfg = FI.load_cfg(); b = FI.setup_env(cfg, building); _ST.update(building=building, b=b, cache={})
        try:                                         # 협업자 door_mask 는 층 번호로만 기억해 건물이 바뀌면 다른 건물 지도가 섞인다 → 비운다(계산은 그대로)
            import metric_fit_negev as MN
            MN._door.clear()
        except ImportError:
            pass
    import rt_localizer as RT, run_floor_id as R
    key = (F, m)
    if key not in _ST["cache"]:
        _ST["cache"] = {key: RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)}
    return _ST["cache"][key]


def depth_label(F, tsec):
    from e2_scene_labels import depth_scene
    return depth_scene(F, tsec)[0]


def frames_for(L, t_s, T):
    from scan_timing import window_indices
    return window_indices(L.T, L.T[0] + t_s + T, T, DT)


def crossing_m(wall_tr, cells_px, track_rel_m, phi_deg, ppm):
    """후보 칸마다 그 자리로 옮긴 주행 경로가 벽(문 제외)을 지난 길이(m). 협업자 metric_fit_v2._crossing_m 과 같은 정의(1 px 조밀화, 지도 밖 = 관통)."""
    from rt_localizer import _A
    rel = (track_rel_m @ _A(np.deg2rad(phi_deg)).T) * ppm                      # 지금 자리 기준 과거 경로(px)
    seg = np.diff(rel, axis=0); n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
    dense = np.vstack([rel[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [rel[-1:]])
    H, W = wall_tr.shape; out = np.zeros(len(cells_px))
    for k in range(0, len(cells_px), 512):
        c = cells_px[k:k + 512]
        x = np.rint(c[:, 0][:, None] + dense[:, 0][None]).astype(int); y = np.rint(c[:, 1][:, None] + dense[:, 1][None]).astype(int)
        inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        bad = ~inb; bad[inb] = wall_tr[y[inb], x[inb]]
        out[k:k + 512] = bad.sum(1) / ppm
    return out


INJ_AT = 20.0      # 출발 뒤 몇 초에 넣나(30·45·60초 시점 평가 전)


def inject(L, t_s, kind):
    """오돔 주입 시험(SYNTHETIC): 메모리 사본만 바꾼다(원본 파일 그대로). 바꾸기 전 배열을 돌려준다."""
    saved = (L.X.copy(), L.Y.copy(), L.YAW.copy())
    t0 = L.T[0] + t_s; ti = t0 + INJ_AT; a = L.T >= ti; i0 = int(np.searchsorted(L.T, ti)); s0 = int(np.searchsorted(L.T, t0))
    if kind == "jump1m":                        # 한 순간 1 m 옆으로 튀고 그 뒤는 그대로 이어짐
        L.X[a] += 1.0
    elif kind == "yaw10":                       # 한 순간 방향이 10° 틀어지고 그 뒤 이동도 틀어진 방향으로
        c, s_ = np.cos(np.radians(10)), np.sin(np.radians(10)); dx, dy = L.X[a] - L.X[i0], L.Y[a] - L.Y[i0]
        L.X[a], L.Y[a] = L.X[i0] + c * dx - s_ * dy, L.Y[i0] + s_ * dx + c * dy; L.YAW[a] += np.radians(10)
    elif kind == "scale10":                     # 출발부터 이동 거리가 10% 크게 나옴
        b = L.T >= t0; L.X[b] = L.X[s0] + 1.10 * (L.X[b] - L.X[s0]); L.Y[b] = L.Y[s0] + 1.10 * (L.Y[b] - L.Y[s0])
    elif kind == "drop5":                       # 5초 동안 오돔이 멈췄다가(값 고정) 다시 실제 값으로
        b = (L.T >= ti) & (L.T < ti + 5.0); L.X[b], L.Y[b], L.YAW[b] = L.X[i0], L.Y[i0], L.YAW[i0]
    else:
        raise ValueError(kind)
    return saved


def job(arg):
    tid, F, building, t_s, m, verify = arg[:6]; kind = arg[6] if len(arg) > 6 else None
    Z = _loc(F, building, m); L = Z.L                                  # 건물 설정(경로 추가) 뒤에 협업자 모듈을 부른다
    import node_local as NL, run_floor_id as R, floor_id as FI
    from metric_fit import set_scale
    from metric_fit_v2 import _maps, CROSS_MAX_M, TRAJ_DT
    from rt_localizer import _A
    if "TRAV_W0" not in _ST:
        _ST.update(TRAV_W0=NL.TRAV_W, FREE_W0=NL.FREE_W)
    t0 = time.time()
    saved = inject(L, t_s, kind) if kind else None
    try:
        return _job(Z, L, NL, R, FI, set_scale, _maps, CROSS_MAX_M, TRAJ_DT, _A, tid, F, t_s, m, verify, t0)
    finally:
        if saved is not None:
            L.X, L.Y, L.YAW = saved


def _job(Z, L, NL, R, FI, set_scale, _maps, CROSS_MAX_M, TRAJ_DT, _A, tid, F, t_s, m, verify, t0):
    meta2 = json.loads(str(np.load(E2SRC / f"{tid}.npz")["meta"]))["maps"][str(m)]
    Z.BETA, Z.MIRROR, Z.lam_sem = math.radians(BETA_MOUNT_DEG), 1, 0.0
    modes = [float(p) for p in meta2["modes"]]
    scales = FI.map_scales(_ST["b"], R, int(_ST["b"]["n_maps"]))[m]
    assert np.allclose(scales, meta2["scales"]), "E2 와 축척 목록이 다름"
    idx = np.flatnonzero(Z.allow.reshape(L.shp)); cells_px = L.Pf[idx].astype(float)
    wall_tr = _maps(Z, m)["wall_tr"]
    # 시점마다 관측 프레임(협업자 창 고르기 그대로)과 프레임 자료
    fl = {T: frames_for(L, t_s, T) for T in TS_ALL}
    fr = {}
    for T, lst in fl.items():
        assert abs(lst[0][0] - (L.T[0] + t_s)) < 1e-6, (tid, T, "첫 관측이 E2 관측과 다름")
        for req, i in lst:
            key = (req, i)
            if key not in fr:
                pts, yaw = L._scan(L.T[i])
                fr[key] = dict(pts=pts, xy=(L.X[i], L.Y[i]), yaw=yaw, t=float(L.T[i]), sample_seed_t=req, sample_id=L.SID[i],
                               scene=L.depth_scene(L.T[i] - L.T[0]), scene_d=depth_label(F, float(L.T[i] - L.T[0])))
    lp = {k: np.log(L.PRIOR[k] + 1e-6) for k in L.PRIOR}
    zero = np.zeros(2)
    want = lambda c, T: T in TS or c in COND_FINE
    out = {(c, T): [] for c in COND for T in TS_ALL if want(c, T)}; nx = {T: 0 for T in TS_ALL}
    for si, s in enumerate(scales):
        set_scale(Z, s)
        NL.TRAV_W, NL.FREE_W, Z.w_nogo = 0.0, 0.0, 0.0                 # 한 프레임 벽 맞추기 g 만 꺼내기
        g = {}
        for key, f in fr.items():
            P1 = Z.localize_buf([dict(f)], modes, f["scene"], 1.0)["P"]
            g[key] = {p: np.asarray(P1[p]) - L._shift(lp[f["scene"]], zero) for p in modes}
        NL.TRAV_W, NL.FREE_W, Z.w_nogo = _ST["TRAV_W0"], _ST["FREE_W0"], 3.0
        for T, lst in fl.items():
            last = fr[lst[-1]]; anchor = np.asarray(last["xy"], float); actual = last["t"]; open_m = L._obs_open(actual)
            fw_l = 0.0 if last["scene"] == "dead-end" else _ST["FREE_W0"]
            fw_d = 0.0 if last["scene_d"] == "dead-end" else _ST["FREE_W0"]
            free_l = -fw_l * np.maximum(0, open_m * 0.6 - L.freeD_grid); free_d = -fw_d * np.maximum(0, open_m * 0.6 - L.freeD_grid)
            sel = np.where((L.T >= L.T[lst[0][1]]) & (L.T <= actual))[0]; keep, lt = [], -1e9
            for i in sel:                                               # 협업자 metric_fit_v2 와 같은 0.2초 조밀 경로
                if L.T[i] - lt >= TRAJ_DT:
                    keep.append(i); lt = L.T[i]
            track = np.c_[np.asarray(L.X)[keep], np.asarray(L.Y)[keep]] - anchor
            for p in modes:
                bel_l = np.zeros(L.shp); bel_d = np.zeros(L.shp); trav = np.zeros(L.shp); bel1 = np.zeros(L.shp)
                for j, key in enumerate(lst):
                    f = fr[key]; od = np.asarray(f["xy"], float) - anchor; off = L.PPM * (_A(np.deg2rad(p)) @ od)
                    sg = L._shift(g[key][p], off)
                    bel_l += sg + L._shift(lp[f["scene"]], off); bel_d += sg + L._shift(lp[f["scene_d"]], off)
                    if j == 0 or j == len(lst) - 1:
                        bel1 += sg + L._shift(lp[f["scene"]], off)
                    px = np.clip(np.round(L.GXX + off[0]).astype(int), 0, L.W - 1); py = np.clip(np.round(L.GYY + off[1]).astype(int), 0, L.H - 1)
                    trav += np.where(L.free[py, px], 0.0, -1.0)
                sp_l = NL.TRAV_W * trav + free_l - Z.w_nogo * Z.pen_g; sp_d = NL.TRAV_W * trav + free_d - Z.w_nogo * Z.pen_g
                X = crossing_m(wall_tr, cells_px, track, p, L.PPM); ex = X > CROSS_MAX_M; nx[T] += int(ex.sum())
                v = lambda a: a.ravel()[idx].astype(np.float32)
                boards = {"T1": v(bel1), "T2": v(bel_l), "T2X": np.where(ex, -np.inf, v(bel_l)).astype(np.float32), "T2S": v(bel_l + sp_l),
                          "T3": np.where(ex, -np.inf, v(bel_l + sp_l)).astype(np.float32), "T2d": v(bel_d), "T3d": np.where(ex, -np.inf, v(bel_d + sp_d)).astype(np.float32)}
                if verify and si == 3 and T == 15 and p == modes[0]:
                    for space, mine in (("off", bel_l), ("on", bel_l + sp_l)):
                        NL.TRAV_W, NL.FREE_W, Z.w_nogo = (0.0, 0.0, 0.0) if space == "off" else (_ST["TRAV_W0"], _ST["FREE_W0"], 3.0)
                        ref = np.asarray(Z.localize(L.T[0] + t_s + T, W=T, dt=DT, phis=modes)["P"][p])
                        print(f"검산 {tid} 지도{m} 축척{s:.1f} T{T} 방향{p:g} 공간감점 {space}: 최대 차이 {np.abs(ref - mine).max():.2e}", flush=True)
                    NL.TRAV_W, NL.FREE_W, Z.w_nogo = _ST["TRAV_W0"], _ST["FREE_W0"], 3.0
                pi = modes.index(p)
                for c in COND:
                    if want(c, T):
                        out[(c, T)].append((si, pi, boards[c]))
    set_scale(Z, R.MID); NL.TRAV_W, NL.FREE_W, Z.w_nogo = _ST["TRAV_W0"], _ST["FREE_W0"], 3.0
    return dict(tid=tid, m=m, beta=float(Z.BETA), mirror=int(Z.MIRROR), modes=modes, scales=[float(x) for x in scales], cell_px=L.Pf[idx].astype(np.float32),
                parts=out, sec=round(time.time() - t0, 1), n_frames={T: len(l) for T, l in fl.items()}, n_cross_excluded=nx)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--workers", type=int, default=20); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--inject", default="", help="오돔 주입 시험: jump1m | yaw10 | scale10 | drop5 (T2·T3 만 저장, 폴더 /data2/e3_inject_20261010/<종류>)")
    a = ap.parse_args()
    global OUT, COND_SAVE
    if a.inject:
        OUT = Path("/data2/e3_inject_20261010") / a.inject; COND_SAVE = ("T2", "T3")
    tasks = list(csv.DictReader(open(G / "tasks/E2_tasks.csv")))
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in a.tasks.split(",")]
    nmap = {"kci": 9, "gong": 7, "sw": 8}
    if a.verify:
        t = tasks[0]; m = 4 if t["building"] == "kci" else 1; init()
        job((t["task_id"], t["floor"], t["building"], float(t["start_sec"]), m, True)); return
    keys = [(c, T) for c in COND_SAVE for T in TS_ALL if T in TS or c in COND_FINE]
    done = lambda tid: all((OUT / f"{c}_T{T}" / f"{tid}.npz").exists() for c, T in keys)
    jobs = [(t["task_id"], t["floor"], t["building"], float(t["start_sec"]), m, False, a.inject or None) for t in tasks if not done(t["task_id"]) for m in range(1, nmap[t["building"]] + 1)]
    jobs.sort(key=lambda j: (j[2], j[1], j[4]))
    print(f"과제 {len({j[0] for j in jobs})}개, 작업 {len(jobs)}개", flush=True)
    t0 = time.time(); got = {}
    with Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            got.setdefault(r["tid"], []).append(r)
            t = [x for x in tasks if x["task_id"] == r["tid"]][0]
            if len(got[r["tid"]]) == nmap[t["building"]]:
                R_ = sorted(got.pop(r["tid"]), key=lambda x: x["m"])
                for c, T in keys:
                    if True:
                        fl_, sc_, ph_, ce_, scs = [], [], [], [], []
                        for x in R_:
                            for si, pi, v in x["parts"][(c, T)]:
                                fl_.append(np.full(len(v), x["m"], np.uint8)); sc_.append(np.full(len(v), si, np.uint8)); ph_.append(np.full(len(v), pi, np.uint8))
                                ce_.append(np.arange(len(v), dtype=np.int32)); scs.append(v)
                        meta = dict(task=r["tid"], T=T, cond=c, e2_source=str(E2SRC), inject=a.inject or None, inject_at_s=INJ_AT if a.inject else None, maps={x["m"]: dict(beta=x["beta"], mirror=x["mirror"], modes=x["modes"], scales=x["scales"],
                                    sec=x["sec"], n_frames=x["n_frames"][T], n_cross_excluded=x["n_cross_excluded"][T]) for x in R_})
                        od = OUT / f"{c}_T{T}"; od.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(od / f"{r['tid']}.npz", floor=np.concatenate(fl_), scale_idx=np.concatenate(sc_), phi_idx=np.concatenate(ph_),
                                            cell=np.concatenate(ce_), score=np.concatenate(scs), meta=json.dumps(meta), **{f"cellpx_{x['m']}": x["cell_px"] for x in R_})
                print(f"  {r['tid']} 저장 ({time.time() - t0:.0f}s)", flush=True)
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
