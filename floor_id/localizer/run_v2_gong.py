#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v2_gong.py — 공학관(G) 층 판별. 감점 강화판(metric_fit_v2) 그대로, 지도만 공학관 7개 층으로 바꾼다.

정보관과 다른 점은 지도 뿌리뿐이다(환경변수로 지정, 코드 분기 없음):
  SEMLOC_MAPS_ROOT=/home/user/gong_maps_perfloor      base_map·door_mask
  SEMLOC_SEMMAPS_ROOT=/home/user/gong_semmaps         sem/mcl 마스크 (mcl 3종은 mcl_interface 내보내기로 생성)
데이터는 G1·G2(각 1개 층 주행)뿐이고, 판정은 G1F~G7F 7장 중 고른다.
사용법: SEMLOC_MAPS_ROOT=... SEMLOC_SEMMAPS_ROOT=... /usr/bin/python3 run_v2_gong.py G1 [G2] [--out-dir out/gong/negfit_v2]
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np

DELIV = '/home/user/kci_a131_1F2F5F_20260902'
USB = '/media/collab/USB/flower_dataset'
BAG = {'G1': f'{USB}/G-1flower/G-1flower_0-001.db3', 'G2': f'{USB}/G-2flower/G-2flower_0.db3'}
N_MAPS = 7


def main():
    os.environ['SEMLOC_SCORE'] = 'max'
    os.environ.setdefault('SEMLOC_MAPS_ROOT', '/home/user/gong_maps_perfloor')
    os.environ.setdefault('SEMLOC_SEMMAPS_ROOT', '/home/user/gong_semmaps')
    ap = argparse.ArgumentParser()
    ap.add_argument('floors', nargs='+')
    ap.add_argument('--out-dir', default='out/gong/negfit_v2')
    a = ap.parse_args()
    import rt_localizer as RT, run_floor_id as R, twoway as TW
    import metric_fit_v2 as V
    os.makedirs(a.out_dir, exist_ok=True)
    for fl in a.floors:
        tf = int(fl[1])
        dd, bg = f'{DELIV}/{fl}', BAG[fl]
        out = Path(a.out_dir) / f'negfit_{fl}.json'
        rows, arrays = [], {}
        if out.exists():
            rows = json.load(open(out))['rows']
            if out.with_suffix('.npz').exists():
                arrays = dict(np.load(out.with_suffix('.npz')))
        done = {(int(r['ts']), r['map']) for r in rows}
        t0 = time.time()
        FL = {}
        for m in range(1, N_MAPS + 1):
            Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=dd, bag=bg)
            Z.bootstrap(Z.L.T[0], boot_s=15.0)
            FL[m] = Z
        L0 = FL[tf].L; span = L0.T[-1] - L0.T[0]
        TS = [float(t) for t in np.arange(70.0, span - 2.0, 60.0)]
        meta = dict(floor=fl, building='gong', data_dir=dd, bag=bg, maps=list(range(1, N_MAPS + 1)), ts=TS,
                    algorithm='metric_fit_v2', maps_root=os.environ['SEMLOC_MAPS_ROOT'], semmaps_root=os.environ['SEMLOC_SEMMAPS_ROOT'],
                    consts=dict(W=60.0, DT=R.DT, SCALES=list(map(float, R.SCALES)), W_FREE=V.W_FREE, W_OUT=V.W_OUT,
                                W_MISS=V.W_MISS, W_CROSS=V.W_CROSS, CROSS_MAX_M=V.CROSS_MAX_M, RMAX=TW.RMAX),
                    no_depth={m: bool(FL[m].L.no_depth) for m in FL}, build_s=round(time.time() - t0, 1))
        assert not any(meta['no_depth'].values()), f'NO_DEPTH 금지: {meta["no_depth"]}'
        print(f'=== {fl} · 지도 {list(FL)} · 시점 {TS} · 준비 {meta["build_s"]}s · 이어받기 {len(done)}', flush=True)
        for ts in TS:
            for m in range(1, N_MAPS + 1):
                if (int(ts), m) in done: continue
                Z = FL[m]; L = Z.L; t = L.T[0] + ts; t1 = time.time()
                r = V.fit_window_v2(Z, t, R.SCALES, floor=m, W=60.0, dt=R.DT)
                pts, yaw = L._scan(t)
                d_obs = TW.sector_obs(np.asarray(pts, float) * [1, r['mirror']], np.deg2rad(r['phi']) + yaw + r['beta'])
                tw = float(TW.twoway_score(getattr(L, 'wall_real', L.wall), *r['pos'], r['ppm'], d_obs)[0])
                q = {k: r[k] for k in ('phi', 'ppm', 'cost_m', 'wall_cost_m', 'blocked_fraction', 'nogo', 'T2', 'T3', 'X_cross_m',
                                       'T4', 'n_wall_hits', 'n_door_rays', 'n_candidates', 'n_feasible', 'half_wall_px',
                                       'median_m', 'inlier_10cm', 'scene', 'mirror')}
                q.update(pos=[float(v) for v in r['pos']], beta_deg=float(np.degrees(r['beta'])), twoway=tw, merit=r['cost_m'])
                rows.append(dict(ts=ts, map=m, true=tf, neg=q, sec=round(time.time() - t1, 1)))
                tag = f'neg_{int(ts)}_{m}'
                arrays[f'cloud_{tag}'] = r['cloud_px'].astype(np.float32)
                arrays[f'traj_{tag}'] = r['trajectory_px'].astype(np.float32)
                arrays[f'dist_{tag}'] = np.asarray(r['distances_m'], np.float32)
                print(f"  {fl} {ts:>4.0f}s 지도G{m}F ppm {r['ppm']:5.1f} 비용 {100*r['wall_cost_m']:.1f}cm T2 {r['T2']:.2f} "
                      f"T3 {r['T3']:.2f} X {r['X_cross_m']:.2f}m T4 {r['T4']:.2f} 가능후보 {r['n_feasible']}/{r['n_candidates']} "
                      f"양방향 {tw:.3f} · {rows[-1]['sec']}s", flush=True)
            json.dump(dict(meta, rows=rows), open(out, 'w'), ensure_ascii=False, indent=1)
            np.savez_compressed(out.with_suffix('.npz'), **arrays)
            win = {r['map']: r for r in rows if int(r['ts']) == int(ts)}
            if len(win) == N_MAPS:
                o = sorted(win, key=lambda m: -win[m]['neg']['twoway'])
                print(f'    {fl} {ts:.0f}s 판정 G{o[0]}F · 정답 {o.index(tf)+1}위', flush=True)
        for Z in FL.values():
            try: Z.L.close()
            except Exception: pass
        print('저장', out, flush=True)


if __name__ == '__main__':
    main()
