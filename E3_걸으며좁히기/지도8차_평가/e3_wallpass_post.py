#!/usr/bin/env python3
"""E3 합치기 결과에 '벽 반대편 통과' 판정을 덧붙인다(2026-10-13, 현장 수정 문서 docs/전달_실시간현장수정_20261013.md).
로봇(realtime/loc_worker.py)처럼 후보 추적에는 안 넣고 순위에만 쓰므로, 저장된 후보 자세(x, y, 방향, 축척)에
그 시점까지 걸은 오돔 경로(합치기와 같은 0.2초 조밀 경로, 기준 = 그 시점 마지막 프레임)를 얹어 다시 계산하면 된다.
  · 확인: 같은 경로로 다시 잰 벽 관통 X 가 저장값과 같은지(X_re vs X)
  · 판정: scripts/wall_pass_test.passes (벽 안 구간 앞뒤 칸이 벽을 안 지나고 직선 3배·직선+2 m 안에 못 이어지면 통과 1번)
  · 벽 = 노드와 같은 wall_tr(wall_real − 문 마스크, 8차 지도)
원본은 건드리지 않고 행마다 wp(통과 횟수)·wp_ev(직선 m, 우회 m)·X_re 를 더한 사본을 --out 에 쓴다.
    KCI_FLOORID_PKG=/data2/floor_id_pkg_v8_20261012/floor_id python3 scripts/e3_wallpass_post.py --src /data2/e3_merged_v8_20261012 --out /data2/e3_merged_v8_wp_20261013 --tasks 4F_T1
"""
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

G = Path("reports/_01_E2_초기위치/_작업자료"); BETA_DEG = 228.0


def job(arg):
    tid, F, building, t_s, src, dst = arg
    import e2_dense as ED
    from wall_pass_test import passes
    sys.path.insert(0, os.environ.get("KCI_FLOORID_PKG", "/data/kci_work/floor_id_pkg/floor_id"))
    import floor_id as FI
    cfg = FI.load_cfg(); FI.setup_env(cfg, building)
    import rt_localizer as RT, run_floor_id as R, metric_fit_negev as MN
    from rt_localizer import _A, window_indices
    from metric_fit_v2 import _maps, _crossing_m, TRAJ_DT
    DT = float(cfg["consts"]["dt_s"]); res = []
    for f in sorted((Path(src) / tid).glob("*.json")):
        g = Path(dst) / tid / f.name
        if g.exists():
            continue
        J = json.load(open(f)); m = int(J["map"]); t0 = time.time()
        MN._door.clear()
        Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
        L = Z.L; Wtr = _maps(Z, m)["wall_tr"]; T_ = np.asarray(L.T); X_ = np.asarray(L.X); Y_ = np.asarray(L.Y)
        start = T_[0] + t_s
        for Ts, o in J["out"].items():
            if not o["rows"]:
                continue
            t_now = start + float(Ts); idx = [i for _, i in window_indices(T_, t_now, float(Ts), DT)]
            il = idx[-1]; anchor = np.array([X_[il], Y_[il]], float)
            sel = np.where((T_ >= start) & (T_ <= T_[il]))[0]; keep, last = [], -1e9
            for i in sel:
                if T_[i] - last >= TRAJ_DT:
                    keep.append(i); last = T_[i]
            dense = np.c_[X_[keep], Y_[keep]] - anchor
            if len(dense) < 2:
                dense = np.c_[X_[idx], Y_[idx]] - anchor
            for r in o["rows"]:
                th = math.radians(r["hd"] - BETA_DEG); s = r["s"]
                tr = dense @ _A(th).T * s + [r["x"], r["y"]]
                r["X_re"] = float(_crossing_m(Wtr, tr, s))
                ev = passes(Wtr, tr, s)
                r["wp"] = len(ev); r["wp_ev"] = [[round(e, 2), round(g_, 1) if np.isfinite(g_) else None] for e, g_ in ev[:5]]
        g.parent.mkdir(parents=True, exist_ok=True); g.write_text(json.dumps(J))
        res.append((m, round(time.time() - t0, 1)))
    return tid, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=""); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if Path(a.out).resolve() == Path(a.src).resolve():
        sys.exit("원본에 덮어쓰지 않음")
    T_ = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    tasks = a.tasks.split(",") if a.tasks else list(T_)
    jobs = [(k, T_[k]["floor"], T_[k]["building"], float(T_[k]["start_sec"]), a.src, a.out) for k in tasks]
    t0 = time.time()
    if a.workers <= 1:
        for j in jobs:
            print(*job(j), f"({time.time() - t0:.0f}s)", flush=True)
        return
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        for tid, res in pool.imap_unordered(job, jobs):
            print(tid, res, f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
