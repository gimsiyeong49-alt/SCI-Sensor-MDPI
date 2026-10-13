#!/usr/bin/env python3
"""E2 가벼운 시험 2: 기존 상위 후보마다 협업자 층 판별(metric_fit_v2.fit_window_v2)의 후보 처리를 제자리 한 장 관측에 그대로 적용(2026-10-10).
  ① 연속 미세 정합: 위치 ±1.5칸·방향 ±8°·축척 0.8~1.25배를 least_squares(soft_l1, f_scale 0.2, 60회)로, 벽 속 점 보정 거리장(_maps F) 기준
  ② 감점: 정합 비용(가까운 70% 평균, m) + 0.5·갈 수 없는 칸 + 0.1·w_nogo·금지 구역 + 1.5·빈 공간 점(0.6 m) + 3.0·건물 밖 점(0.5 m)
  ③ 안 보인 벽 T4(180방향, 문 너머 제외) 0.6 배를 더한 merit_v2  (벽 관통 경로 X 는 제자리라 0)
  ④ 미세 정합한 자세·축척에서 양방향 점수(twoway)
협업자 코드는 고치지 않고 그 함수·상수를 부른다(fit_window_v2 안의 계산을 후보 하나 단위로 꺼낸 것). 정답은 채점에만 쓴다.
순위: (a) 기존 점수 (b) 양방향(기존 자세) (c) merit_v2(미세 정합) (d) 양방향(미세 정합 자세) — 다시 매긴 뒤 같은 층 2 m·30° 안은 하나로 합친다.
    python3 scripts/e2_refine_test.py --src T0_nospace --tasks 4F_T1,...
결과: reports/_01_E2_초기위치/_작업자료/미세정합시험/<src>/<과제>.json
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_dense as ED  # noqa: E402
from e2_dense_score import distinct  # noqa: E402

G = Path("reports/_01_E2_초기위치/_작업자료"); R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/36_채점용정답")
OUT = G / "미세정합시험"; wrap = lambda a: (a + 180) % 360 - 180
BETA = math.radians(228.0)
_Z = {}


def env(building):
    import floor_id as FI
    if ED._ST.get("building") != building:
        cfg = FI.load_cfg(); b = FI.setup_env(cfg, building); ED._ST.update(building=building, b=b, cache={})
        try:
            import metric_fit_negev as MN
            MN._door.clear()
        except ImportError:
            pass
        _Z.clear()


def loc(building, m, F):
    env(building)
    if m not in _Z:
        import rt_localizer as RT, run_floor_id as Rr
        _Z[m] = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=Rr.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
    return _Z[m]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="T0_nospace"); ap.add_argument("--top", type=int, default=1000); ap.add_argument("--tasks", default="")
    ap.add_argument("--save-top", action="store_true", help="방법별로 다시 매긴 Top-100 후보(층·x·y·방향·축척·그 방법 점수)도 저장(그림·후보 목록용)")
    ap.add_argument("--out-name", default="", help="결과 하위 폴더 이름(기본 = --src)")
    ap.add_argument("--lock-scale", action="store_true", help="축척은 그대로 두고 위치·방향만 미세 정합(고정 24·정답 축척 조건용: 조건의 뜻 유지)")
    a = ap.parse_args(); od = OUT / (a.out_name or a.src); od.mkdir(parents=True, exist_ok=True)
    tasks = list(csv.DictReader(open(G / "tasks/E2_tasks.csv"))); gtt = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in a.tasks.split(",")]
    tasks.sort(key=lambda t: (t["building"], t["floor"]))
    for t in tasks:
        k, F, b = t["task_id"], t["floor"], t["building"]
        if (od / f"{k}.json").exists():
            continue
        t0 = time.time(); m_true = int(gtt[k]["true_map"]); ts = float(t["start_sec"])
        env(b)
        import rt_localizer as RT, run_floor_id as Rr, twoway as TW
        from rt_localizer import _A, _R
        from metric_fit_v2 import _maps, _sample, FAR_M, OUT_M, W_FREE, W_OUT, W_MISS
        from metric_fit_negev import unexplained_walls
        from scipy.optimize import least_squares
        from node_local import GRID_STEP
        d = dict(np.load(Path("/data2/e2_dense_20261009") / a.src / f"{k}.npz")); meta = json.loads(str(d["meta"]))
        acc = distinct(d, meta, a.top)
        Ls = loc(b, 1, F) if F[0].isdigit() and b == "kci" else loc(b, 1, F)
        import e2_dense as _E
        Zs = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=1, ppm=Rr.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None) \
            if ("scan", F) not in _Z else _Z[("scan", F)]
        _Z[("scan", F)] = Zs; L0 = Zs.L
        tt = L0.T[0] + ts; i0 = int(np.argmin(abs(L0.T - tt))); pts_full, yaw = L0._scan(L0.T[i0]); pts_full = np.asarray(pts_full, float)
        pts = pts_full[np.random.RandomState(int(tt) % 97).choice(len(pts_full), 400, replace=False)] if len(pts_full) > 400 else pts_full
        cloud = pts @ _R(yaw + BETA).T; sample = cloud
        project = lambda x, P: P @ _A(x[2]).T * np.exp(x[3]) + x[:2]
        res = []
        for c in acc:
            m = c["m"]; Z = loc(b, m, F); L = Z.L; M = _maps(Z, m); CV = float(max(L.H, L.W))
            dist = lambda x, P: _sample(M["F"], project(x, P), CV) / np.exp(x[3])
            phi = math.radians(c["hd"]) - BETA
            x0 = np.r_[c["px"], phi, math.log(c["s"])]; sh = 1.5 * GRID_STEP
            ds = (-1e-6, 1e-6) if a.lock_scale else (np.log(.8), np.log(1.25))
            lo = x0 + [-sh, -sh, -np.deg2rad(8), ds[0]]; hi = x0 + [sh, sh, np.deg2rad(8), ds[1]]
            x = least_squares(lambda x: np.minimum(dist(x, sample), 5.), x0, bounds=(lo, hi), loss="soft_l1", f_scale=.2,
                              diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60).x
            px, py = np.rint(x[:2]).astype(int); s = float(np.exp(x[3]))
            inb = 0 <= px < L.W and 0 <= py < L.H
            err = dist(x, cloud); cost = float(np.mean(np.sort(err)[:int(.7 * len(err))]))
            blocked = 0.0 if (inb and Z.ok_px[py, px]) else 1.0; nogo = float(Z.pen_px[py, px]) if inb else 1.0
            cpx = project(x, cloud); T2 = float(np.mean(err > FAR_M)); T3 = float(np.mean(_sample(M["out_d"], cpx, CV) / s > OUT_M))
            part = cost + .5 * blocked + .1 * Z.w_nogo * nogo + W_FREE * T2 + W_OUT * T3
            t4 = unexplained_walls(M["wall"], M["door"], cpx, np.array([x[:2]]), s)[0]
            wall = getattr(L, "wall_real", L.wall)
            tw0 = float(TW.twoway_score(wall, c["px"][0], c["px"][1], c["s"], TW.sector_obs(pts_full, math.radians(c["hd"]) + yaw))[0])
            tw1 = float(TW.twoway_score(wall, x[0], x[1], s, TW.sector_obs(pts_full, x[2] + yaw + BETA))[0])
            res.append(dict(m=m, x=float(x[0]), y=float(x[1]), hd=float((math.degrees(x[2] + BETA)) % 360), s=s, x0=float(c["px"][0]), y0=float(c["px"][1]),
                            hd0=float(c["hd"] % 360), s0=float(c["s"]), score=float(c["score"]), merit=float(part + W_MISS * t4), tw0=tw0, tw1=tw1,
                            cost=cost, T2=T2, T3=T3, T4=float(t4), blocked=blocked, ok=bool(inb and Z.ok_px[py, px])))
        # 채점(정답은 여기서만)
        S = json.load(open(R / F / "summary.json")); ppm = S["ppm"]; yaw_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
        gp = list(csv.DictReader(open(R / F / "warp_clean" / "gt_warp.csv"))); gt = np.array([float(q["relative_sec"]) for q in gp]); gx = np.array([[float(q["x_px"]), float(q["y_px"])] for q in gp])
        pg = gx[int(np.argmin(abs(gt - ts)))]
        tops = {}
        def rank(key, desc, refined, name=None):
            order = sorted(range(len(res)), key=lambda i: (-res[i][key] if desc else res[i][key]))
            kept, first_true, first_floor = [], None, None
            for i in order:
                r = res[i]; X, Y, H, Sc = (r["x"], r["y"], r["hd"], r["s"]) if refined else (r["x0"], r["y0"], r["hd0"], r["s0"])
                if refined and not r["ok"]:
                    continue
                if any(q[0] == r["m"] and math.hypot(X - q[1], Y - q[2]) / min(Sc, q[4]) <= 2.0 and abs(wrap(H - q[3])) <= 30 for q in kept):
                    continue
                kept.append((r["m"], X, Y, H, Sc)); n = len(kept)
                if name and n <= 100:                     # 그림용: 다시 매긴 Top-100 후보(층·지도 px·방향·축척)
                    tops.setdefault(name, []).append([int(r["m"]), round(X, 1), round(Y, 1), round(H, 1), round(Sc, 3), round(r[key], 4)])
                if first_floor is None and r["m"] == m_true:
                    first_floor = n
                if first_true is None and r["m"] == m_true and math.hypot(X - pg[0], Y - pg[1]) / ppm <= 3.0 and abs(wrap(H - yaw_gt)) <= 30:
                    first_true = n
                if n >= 1000:
                    break
            return first_true, first_floor
        out = dict(task=k, n=len(res), sec=round(time.time() - t0, 1))
        for nm, key, desc, ref in (("orig", "score", True, False), ("tw0", "tw0", True, False), ("merit", "merit", False, True), ("tw1", "tw1", True, True)):
            out[f"rank_{nm}"], out[f"floor_rank_{nm}"] = rank(key, desc, ref, nm if a.save_top else None)
        if a.save_top:
            out["top100"] = tops
        tc = [r for r in res if r["m"] == m_true and math.hypot(r["x"] - pg[0], r["y"] - pg[1]) / ppm <= 3.0]
        out["true_scale_err_before"] = float(np.median([abs(r["s0"] - ppm) / ppm * 100 for r in tc])) if tc else None
        out["true_scale_err_after"] = float(np.median([abs(r["s"] - ppm) / ppm * 100 for r in tc])) if tc else None
        (od / f"{k}.json").write_text(json.dumps(out))
        print(k, {kk: out[kk] for kk in ("rank_orig", "rank_tw0", "rank_merit", "rank_tw1", "sec")}, flush=True)


if __name__ == "__main__":
    main()
