#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_v2_sw.py — 소융대(S) 층 판별. 감점 강화판(metric_fit_v2) 그대로, 지도만 소융대 8개 층으로 바꾼다.

정보관과 다른 점은 지도 뿌리뿐이다(환경변수로 지정, 코드 분기 없음):
  SEMLOC_MAPS_ROOT=/home/user/sw_maps_perfloor        base_map·door_mask
  SEMLOC_SEMMAPS_ROOT=/home/user/sw_semmaps           sem/mcl 마스크 (sowoong_map_ws/output/map_bundle 을 가리키는 링크)
데이터는 S1·S6(각 1개 층 주행)뿐이고, 판정은 S1F~S8F 8장 중 고른다.
사용법: /usr/bin/python3 run_v2_sw.py S1 [S6] [--out-dir out/sw/negfit_v2_mapscale] [--fixed-scale]

축척 탐색 (2026-09-21):
  정보관·공대는 모든 지도에 같은 픽셀 축척 목록 R.SCALES(15~43 px/m, 미세조정 ×0.8~1.25)를 쓴다.
  소융대는 안내도 사진 해상도가 층마다 다르다(S2F~S5F 가로 3700~4000px, S1F·S6F~S8F 6000px 이상).
  같은 물리 축척이 고해상도 지도에서는 픽셀로 더 커서, S6 데이터의 정답 S6F 자세는 66 px/m 가
  필요한데 고정 목록의 도달 한계는 53.8 이다. 실제로 S6 70초 구간이 S6F 에서 복도를 가로질러
  놓였고(비용 61.8cm, 가능 후보 0/42), 범위를 넓히자 복도를 따라 3.6cm·18/87 로 맞았다.
  그래서 기본값은 **지도별 축척**: 각 지도 yaml 해상도의 px/m 를 건물 내 중앙값으로 나눈 비율만큼
  R.SCALES 를 옮긴다. 모든 지도에서 같은 물리 축척 범위를 탐색하게 되는 것이다.
  yaml 은 base_map 링크를 따라간 실제 파일 옆에서 읽는다(경로를 박지 않는다).
  --fixed-scale 을 주면 정보관·공대와 같은 고정 목록으로 돈다.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import yaml

DELIV = '/home/user/kci_a131_1F2F5F_20260902'
USB = '/media/collab/USB/flower_dataset'
BAG = {'S1': f'{USB}/S-1flower/S-1flower_0.db3', 'S6': f'{USB}/S-6flower/S-6flower_0.db3'}
N_MAPS = 8


def map_ppm(maps_root, m):
    """base_map 링크를 따라간 실제 파일 옆 yaml 의 해상도 -> px/m."""
    real = Path(os.path.realpath(f'{maps_root}/{m}F/maps/base_map.png'))
    y = yaml.safe_load(open(real.with_suffix('.yaml')))
    return 1.0 / float(y['resolution'])


def main():
    os.environ['SEMLOC_SCORE'] = 'max'
    os.environ.setdefault('SEMLOC_MAPS_ROOT', '/home/user/sw_maps_perfloor')
    os.environ.setdefault('SEMLOC_SEMMAPS_ROOT', '/home/user/sw_semmaps')
    ap = argparse.ArgumentParser()
    ap.add_argument('floors', nargs='+')
    ap.add_argument('--out-dir', default='out/sw/negfit_v2_mapscale')
    ap.add_argument('--fixed-scale', action='store_true', help='정보관·공대와 같은 고정 축척 목록(R.SCALES)')
    a = ap.parse_args()
    import rt_localizer as RT, run_floor_id as R, twoway as TW
    import metric_fit_v2 as V
    os.makedirs(a.out_dir, exist_ok=True)
    if a.fixed_scale:
        SC = {m: [float(v) for v in R.SCALES] for m in range(1, N_MAPS + 1)}
        ppm = ref = None
    else:
        ppm = {m: map_ppm(os.environ['SEMLOC_MAPS_ROOT'], m) for m in range(1, N_MAPS + 1)}
        ref = float(np.median(list(ppm.values())))
        SC = {m: [float(v) * ppm[m] / ref for v in R.SCALES] for m in ppm}
        print('지도별 축척 탐색(px/m):', {f'S{m}F': f'{SC[m][0]:.1f}~{SC[m][-1]:.1f}' for m in SC}, flush=True)
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
        meta = dict(floor=fl, building='sw', data_dir=dd, bag=bg, maps=list(range(1, N_MAPS + 1)), ts=TS,
                    algorithm='metric_fit_v2', maps_root=os.environ['SEMLOC_MAPS_ROOT'], semmaps_root=os.environ['SEMLOC_SEMMAPS_ROOT'],
                    scale_mode='fixed' if a.fixed_scale else 'per_map_yaml_ratio', map_ppm_yaml=ppm, scale_ref_ppm=ref,
                    scales_per_map={m: [round(v, 2) for v in SC[m]] for m in SC},
                    consts=dict(W=60.0, DT=R.DT, SCALES=list(map(float, R.SCALES)), W_FREE=V.W_FREE, W_OUT=V.W_OUT,
                                W_MISS=V.W_MISS, W_CROSS=V.W_CROSS, CROSS_MAX_M=V.CROSS_MAX_M, RMAX=TW.RMAX),
                    no_depth={m: bool(FL[m].L.no_depth) for m in FL}, build_s=round(time.time() - t0, 1))
        assert not any(meta['no_depth'].values()), f'NO_DEPTH 금지: {meta["no_depth"]}'
        print(f'=== {fl} · 지도 {list(FL)} · 시점 {TS} · 준비 {meta["build_s"]}s · 이어받기 {len(done)}', flush=True)
        for ts in TS:
            for m in range(1, N_MAPS + 1):
                if (int(ts), m) in done: continue
                Z = FL[m]; L = Z.L; t = L.T[0] + ts; t1 = time.time()
                r = V.fit_window_v2(Z, t, SC[m], floor=m, W=60.0, dt=R.DT)
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
                print(f"  {fl} {ts:>4.0f}s 지도S{m}F ppm {r['ppm']:5.1f} 비용 {100*r['wall_cost_m']:.1f}cm T2 {r['T2']:.2f} "
                      f"T3 {r['T3']:.2f} X {r['X_cross_m']:.2f}m T4 {r['T4']:.2f} 가능후보 {r['n_feasible']}/{r['n_candidates']} "
                      f"양방향 {tw:.3f} · {rows[-1]['sec']}s", flush=True)
            json.dump(dict(meta, rows=rows), open(out, 'w'), ensure_ascii=False, indent=1)
            np.savez_compressed(out.with_suffix('.npz'), **arrays)
            win = {r['map']: r for r in rows if int(r['ts']) == int(ts)}
            if len(win) == N_MAPS:
                o = sorted(win, key=lambda m: -win[m]['neg']['twoway'])
                print(f'    {fl} {ts:.0f}s 판정 S{o[0]}F · 정답 {o.index(tf)+1}위', flush=True)
        for Z in FL.values():
            try: Z.L.close()
            except Exception: pass
        print('저장', out, flush=True)


if __name__ == '__main__':
    main()
