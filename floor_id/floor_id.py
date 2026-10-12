#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor_id.py — 온보드 층 판별 진입점. 깊이 카메라 없이 LiDAR 만으로 돈다.

    ./floor_id.py --building sw --data /경로/주행데이터 [--out out/sw]
    ./floor_id.py --building kci --data /경로/1F --at 70 130 190

건물 설정은 config/buildings.yaml 에서 읽고, 지도는 패키지 안 maps/ 를 쓴다.
경로를 손댈 필요가 없다. 자세한 설명은 docs/USAGE.md.
"""
import argparse, json, os, sys, time
from pathlib import Path

PKG = Path(__file__).resolve().parent


def load_cfg():
    import yaml
    return yaml.safe_load((PKG / 'config' / 'buildings.yaml').read_text(encoding='utf-8'))


def setup_env(cfg, key):
    """지도 뿌리와 장면 분류원을 환경변수로 넣는다. 알고리즘 코드는 이 변수만 본다."""
    b = cfg['buildings'][key]
    os.environ['SEMLOC_MAPS_ROOT'] = str(PKG / b['maps_root'])
    os.environ['SEMLOC_SEMMAPS_ROOT'] = str(PKG / b['semmaps_root'])
    os.environ['SEMLOC_SCENE'] = cfg.get('scene_source', 'lidar')
    os.environ['SEMLOC_NO_DEPTH'] = '1'          # 깊이 토픽을 아예 열지 않는다
    os.environ['SEMLOC_SCORE'] = 'max'
    c = cfg.get('consts', {})
    os.environ['SCENE_FWD_DEG'] = str(c.get('scene_fwd_deg', 7.0))
    os.environ['SCENE_SIDE_LO'] = str(c.get('scene_side_lo_deg', 17.5))
    os.environ['SCENE_SIDE_HI'] = str(c.get('scene_side_hi_deg', 35.0))
    near = float(c.get('scene_side_near_m', 0) or 0)
    os.environ['SCENE_SIDE_NEAR_M'] = str(near if near > 0 else 1e9)
    os.environ['GT_REALIGN'] = '0'               # 채점용 보정은 온보드에서 쓰지 않는다
    sys.path.insert(0, str(PKG / 'localizer'))
    return b


def map_scales(b, R, n_maps):
    """per_map 이면 지도 yaml 해상도 비율로 축척 목록을 옮긴다(소융대처럼 층별 해상도가 다른 건물)."""
    if b.get('scale') != 'per_map':
        return {m: [float(v) for v in R.SCALES] for m in range(1, n_maps + 1)}
    import yaml as _y
    import numpy as np
    ppm = {}
    for m in range(1, n_maps + 1):
        d = Path(os.environ['SEMLOC_MAPS_ROOT']) / f'{m}F' / 'maps'
        # 패키지: base_map.png 옆 yaml. 연구 저장소(링크): 링크를 따라간 실제 지도 옆 yaml.
        for y in (d / 'base_map.yaml', Path(os.path.realpath(d / 'base_map.png')).with_suffix('.yaml')):
            if y.exists():
                ppm[m] = 1.0 / float(_y.safe_load(y.read_text())['resolution'])
                break
    if len(ppm) < n_maps:
        # 조용히 고정 축척으로 떨어지면 성적이 달라진다(2026-10-01 패키지에 yaml 이 빠져 있었다).
        miss = [m for m in range(1, n_maps + 1) if m not in ppm]
        raise SystemExit(f'축척 per_map 인데 base_map.yaml 이 없는 층: {miss} — tools/sync_maps.sh 로 지도를 다시 넣을 것')
    ref = float(np.median(list(ppm.values())))   # 연구 실행기 run_v2_sw.py 와 같은 기준(짝수 층이면 가운데 둘 평균)
    return {m: [float(v) * ppm[m] / ref for v in R.SCALES] for m in ppm}


def main():
    ap = argparse.ArgumentParser(description='온보드 층 판별 (깊이 카메라 불필요)')
    ap.add_argument('--building', required=True, help='config/buildings.yaml 의 키 (kci/gong/sw)')
    ap.add_argument('--data', required=True, help='주행 데이터 폴더 (odom_link.csv · samples/ · far_points.npz)')
    ap.add_argument('--out', default=None, help='결과 폴더 (기본 out/<건물>)')
    ap.add_argument('--at', type=float, nargs='*', default=None,
                    help='판별할 시점(초, 데이터 시작 기준). 생략하면 70초부터 60초 간격 전부')
    ap.add_argument('--window', type=float, default=None, help='정합 창 길이(초). 기본은 설정값')
    a = ap.parse_args()

    cfg = load_cfg()
    if a.building not in cfg['buildings']:
        sys.exit(f"건물 키가 없다: {a.building} (가능: {', '.join(cfg['buildings'])})")
    b = setup_env(cfg, a.building)
    data = Path(a.data).resolve()
    for need in ('odom_link.csv', 'samples', 'far_points.npz'):
        if not (data / need).exists():
            sys.exit(f'주행 데이터에 {need} 가 없다: {data}')

    # 패키지 밖을 보고 있지 않은지 먼저 확인한다. 알고리즘 모듈에는 옛 연구 경로가
    # 기본값으로 남아 있어, 환경변수가 안 먹으면 조용히 엉뚱한 지도를 읽을 수 있다.
    for var in ('SEMLOC_MAPS_ROOT', 'SEMLOC_SEMMAPS_ROOT'):
        root = Path(os.environ[var])
        if not root.is_dir():
            sys.exit(f'{var} 가 가리키는 지도 폴더가 없다: {root}')
        if PKG not in root.parents and root != PKG:
            sys.exit(f'{var} 가 패키지 밖을 가리킨다: {root}')
        miss = [m for m in range(1, int(b['n_maps']) + 1) if not (root / f'{m}F').is_dir()]
        if miss:
            sys.exit(f'{var}: {miss} 층 지도가 없다 ({root})')

    import numpy as np
    import rt_localizer as RT, run_floor_id as R, twoway as TW
    import metric_fit_v2 as V

    W = a.window if a.window else float(cfg['consts']['window_s'])
    DT = float(cfg['consts']['dt_s'])
    n_maps = int(b['n_maps'])
    out_dir = Path(a.out) if a.out else PKG / 'out' / a.building
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    FL = {}
    for m in range(1, n_maps + 1):
        Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m,
                           ppm=R.MID, data_dir=str(data), bag=None)
        Z.bootstrap(Z.L.T[0], boot_s=15.0)
        FL[m] = Z
    SC = map_scales(b, R, n_maps)
    L0 = FL[1].L
    span = L0.T[-1] - L0.T[0]
    TS = a.at if a.at else [float(t) for t in np.arange(70.0, span - 2.0, 60.0)]
    print(f"[{b['name']}] 지도 {n_maps}개 · 시점 {len(TS)}개 · 준비 {time.time()-t0:.1f}s · "
          f"장면 {os.environ['SEMLOC_SCENE']} · 창 {W}s", flush=True)

    rows = []
    for ts in TS:
        win = {}
        for m in range(1, n_maps + 1):
            Z = FL[m]; L = Z.L; t = L.T[0] + ts; t1 = time.time()
            r = V.fit_window_v2(Z, t, SC[m], floor=m, W=W, dt=DT)
            pts, yaw = L._scan(t)
            d_obs = TW.sector_obs(np.asarray(pts, float) * [1, r['mirror']],
                                  np.deg2rad(r['phi']) + yaw + r['beta'])
            tw = float(TW.twoway_score(getattr(L, 'wall_real', L.wall), *r['pos'], r['ppm'], d_obs)[0])
            win[m] = dict(map=m, twoway=tw, ppm=r['ppm'], phi=r['phi'],
                          pos=[float(v) for v in r['pos']], wall_cost_m=r['wall_cost_m'],
                          scene=r['scene'], sec=round(time.time() - t1, 1))
        order = sorted(win, key=lambda m: -win[m]['twoway'])
        best = order[0]
        rows.append(dict(ts=ts, verdict=best, ranking=order, cands=[win[m] for m in order]))
        print(f"  {ts:6.0f}s  판정 {best}층  (2위 {order[1]}층)  "
              f"위치 px {win[best]['pos'][0]:.0f},{win[best]['pos'][1]:.0f}  "
              f"ppm {win[best]['ppm']:.1f}  φ {win[best]['phi']:.0f}  "
              f"벽비용 {100*win[best]['wall_cost_m']:.1f}cm  장면 {win[best]['scene']}", flush=True)
        (out_dir / 'result.json').write_text(json.dumps(
            dict(building=a.building, name=b['name'], data=str(data), n_maps=n_maps,
                 window_s=W, dt_s=DT, scene_source=os.environ['SEMLOC_SCENE'],
                 scales={k: [round(v, 2) for v in s] for k, s in SC.items()}, rows=rows),
            ensure_ascii=False, indent=1))
    print(f'저장 {out_dir/"result.json"}', flush=True)


if __name__ == '__main__':
    main()
