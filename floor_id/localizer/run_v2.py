#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v2.py — 감점 강화판(metric_fit_v2) 층 판별 실행. 1~9층.
run_negev_0914 와 같은 데이터·창·시점 격자·판정 규칙(양방향 점수 최대). 자세 선택만 v2.
3F: USB 로스백 손상 → 노트북 사본, 4F: 로스백은 HANDOFF_4F_CONTINUE (장면캐시는 둘 다 데이터 폴더에 있음).
출력 형식은 viz_floorcases_f.py 가 읽는 negfit 형식과 같다(선택 자세를 'neg' 키로 저장).
사용법: /usr/bin/python3 run_v2.py 6F 2F [--out-dir out/update0914/negfit_v2]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np


def main():
    os.environ['SEMLOC_SCORE'] = 'max'
    ap = argparse.ArgumentParser()
    ap.add_argument('floors', nargs='+')
    ap.add_argument('--out-dir', default='out/update0914/negfit_v2')
    a = ap.parse_args()
    import run_floor_id as R, twoway as TW
    import metric_fit_v2 as V
    R.DATA_BY_FLOOR['3F_d'] = (R.DATA_BY_FLOOR['3F_d'][0], '/home/user/a131_bags/3F/3flower_0.db3')
    R.DATA_BY_FLOOR['4F_d'] = (R.DATA_BY_FLOOR['4F_d'][0], '/home/user/HANDOFF_4F_CONTINUE/rosbag/4F_full_walk_0.db3')
    os.makedirs(a.out_dir, exist_ok=True)
    for fl in a.floors:
        key, tf = f'{fl}_d', int(fl[0])
        out = Path(a.out_dir) / f'negfit_{fl}.json'
        rows, arrays = [], {}
        if out.exists():
            rows = json.load(open(out))['rows']
            if out.with_suffix('.npz').exists():
                arrays = dict(np.load(out.with_suffix('.npz')))
        done = {(int(r['ts']), r['map']) for r in rows}
        t0 = time.time()
        FL = R.build(key)[0]
        L0 = FL[tf].L; span = L0.T[-1] - L0.T[0]
        TS = [float(t) for t in np.arange(70.0, span - 2.0, 60.0)]
        meta = dict(floor=fl, data_dir=R.DATA_BY_FLOOR[key][0], bag=str(R.DATA_BY_FLOOR[key][1]), maps=list(range(1, 10)), ts=TS,
                    algorithm='metric_fit_v2', consts=dict(W=60.0, DT=R.DT, SCALES=list(map(float, R.SCALES)), W_FREE=V.W_FREE,
                    W_OUT=V.W_OUT, W_MISS=V.W_MISS, W_CROSS=V.W_CROSS, FAR_M=V.FAR_M, OUT_M=V.OUT_M, CROSS_MAX_M=V.CROSS_MAX_M,
                    TRAJ_DT=V.TRAJ_DT, N_T4=V.N_T4, RMAX=TW.RMAX),
                    no_depth={f: bool(getattr(FL[f].L, 'no_depth', False)) for f in FL}, build_s=round(time.time() - t0, 1))
        assert not any(meta['no_depth'].values()), f'NO_DEPTH 금지: {meta["no_depth"]}'
        print(f'=== {fl} · 시점 {TS} · 준비 {meta["build_s"]}s · 이어받기 {len(done)}', flush=True)
        for ts in TS:
            for f in range(1, 10):
                if (int(ts), f) in done: continue
                Z = FL[f]; L = Z.L; t = L.T[0] + ts; t1 = time.time()
                r = V.fit_window_v2(Z, t, R.SCALES, floor=f, W=60.0, dt=R.DT)
                pts, yaw = L._scan(t)
                d_obs = TW.sector_obs(np.asarray(pts, float) * [1, r['mirror']], np.deg2rad(r['phi']) + yaw + r['beta'])
                tw = float(TW.twoway_score(getattr(L, 'wall_real', L.wall), *r['pos'], r['ppm'], d_obs)[0])
                q = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()
                     if k in ('phi', 'ppm', 'cost_m', 'wall_cost_m', 'blocked_fraction', 'nogo', 'T2', 'T3', 'X_cross_m', 'T4',
                              'T4_with_doors', 'n_wall_hits', 'n_door_rays', 'rank_in_part', 'n_candidates', 'n_feasible',
                              'half_wall_px', 'median_m', 'inlier_10cm', 'scene', 'mirror')}
                q.update(pos=[float(v) for v in r['pos']], beta_deg=float(np.degrees(r['beta'])), twoway=tw, merit=r['cost_m'])
                rows.append(dict(ts=ts, map=f, true=tf, neg=q, sec=round(time.time() - t1, 1)))
                tag = f'neg_{int(ts)}_{f}'
                arrays[f'cloud_{tag}'] = r['cloud_px'].astype(np.float32); arrays[f'traj_{tag}'] = r['trajectory_px'].astype(np.float32)
                arrays[f'frames_{tag}'] = r['frame_px'].astype(np.float32); arrays[f'dist_{tag}'] = np.asarray(r['distances_m'], np.float32)
                arrays[f'miss_{tag}'] = np.asarray(r['miss_wall_px'], np.float32)
                arrays[f'fidx_{int(ts)}_{f}'] = r['fidx']
                print(f"  {fl} {ts:>4.0f}s 지도{f}F ppm {r['ppm']:5.1f} 비용 {100*r['wall_cost_m']:.1f}cm T2 {r['T2']:.2f} T3 {r['T3']:.2f} "
                      f"X {r['X_cross_m']:.2f}m T4 {r['T4']:.2f} 가능후보 {r['n_feasible']}/{r['n_candidates']} 양방향 {tw:.3f} · {rows[-1]['sec']}s",
                      flush=True)
            json.dump(dict(meta, rows=rows), open(out, 'w'), ensure_ascii=False, indent=1)
            np.savez_compressed(out.with_suffix('.npz'), **arrays)
            win = {r['map']: r for r in rows if int(r['ts']) == int(ts)}
            if len(win) == 9:
                o = sorted(win, key=lambda m: -win[m]['neg']['twoway'])
                print(f'    {fl} {ts:.0f}s v2: 판정 {o[0]}F · 정답 {o.index(tf)+1}위', flush=True)
        for Z in FL.values():
            try: Z.L.close()
            except Exception: pass
        print('저장', out, flush=True)


if __name__ == '__main__':
    main()
