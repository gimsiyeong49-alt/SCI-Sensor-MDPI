#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scene_label_agree.py — LiDAR 장면 라벨이 깊이 카메라 라벨을 얼마나 재현하는지 잰다.

깊이 라벨은 각 주행 폴더의 scene_cache_v1.npy(1초 격자)에 이미 있다.
같은 시점을 node_local.lidar_scene 규칙으로 다시 분류해 혼동표를 낸다.

  /usr/bin/python3 scene_label_agree.py <데이터폴더>[,<폴더>...]
  SCENE_SIDE_NEAR_M=12 /usr/bin/python3 scene_label_agree.py ...   (파라미터 바꿔 보기)
"""
import collections, csv, os, sys
from pathlib import Path
import numpy as np

FWD_DEG = float(os.environ.get('SCENE_FWD_DEG', 7.0))
SIDE_LO = float(os.environ.get('SCENE_SIDE_LO', 17.5))
SIDE_HI = float(os.environ.get('SCENE_SIDE_HI', 35.0))
SIDE_NEAR = float(os.environ.get('SCENE_SIDE_NEAR_M', 9.0))


def scene_from(fwd, L, R):
    wide = max(L, R)
    if fwd < 4.5: return 'dead-end'
    if (fwd > 12 and wide < 7.5) or wide < 4.5: return 'corridor'
    return 'junction'


def dist(pts):
    r = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    near = r <= SIDE_NEAR
    def med(m):
        return float(np.median(r[m])) if m.sum() >= 5 else np.nan
    return (med(np.abs(ang) <= FWD_DEG),
            med((ang >= SIDE_LO) & (ang <= SIDE_HI) & near),
            med((ang <= -SIDE_LO) & (ang >= -SIDE_HI) & near))


def run(data_dir):
    d = Path(data_dir)
    cache = d / 'scene_cache_v1.npy'
    if not cache.exists():
        print(f'{d.name}: 장면 캐시 없음'); return
    dep = list(np.load(cache, allow_pickle=True))
    rows = list(csv.DictReader(open(d / 'odom_link.csv')))
    T = np.array([float(r['t_sec']) for r in rows]); SID = [r['sample_id'] for r in rows]
    far = np.load(d / 'far_points.npz')
    t0 = T[0]

    def pts_at(sec):
        i = int(np.argmin(np.abs(T - (t0 + max(0.0, sec)))))
        z = np.load(d / 'samples' / SID[i] / 'data_short.npz')
        a = z['a1_3_1_all_candidate_xy'].astype(float); role = z['a1_3_1_all_candidate_role']
        p = a[role != 4]
        s = SID[i]
        if s in far.files and len(far[s]):
            p = np.vstack([p, far[s].astype(float)])
        return p

    got = []
    for k in range(len(dep)):
        samp = [dist(p) for p in (pts_at(k - t) for t in (0.0, 0.5, 1.0, 1.5)) if len(p) >= 20]
        if not samp: continue
        f = float(np.nanmedian([s[0] for s in samp]))
        if not np.isfinite(f): continue
        def nm(v):
            v = [x for x in v if np.isfinite(x)]
            return float(np.median(v)) if v else 0.0
        got.append((dep[k], scene_from(f, nm([s[1] for s in samp]), nm([s[2] for s in samp]))))
    if not got: return
    cm = collections.Counter(got)
    print(f'{d.name:6s} n={len(got):4d}  일치율 {100*sum(a==b for a,b in got)/len(got):5.1f}%')
    for a in ('corridor', 'junction', 'dead-end'):
        tot = sum(v for (x, _), v in cm.items() if x == a)
        if tot:
            br = ' '.join(f'{b}:{cm[(a,b)]}' for b in ('corridor','junction','dead-end') if cm[(a,b)])
            print(f'   깊이 {a:9s} {tot:4d} → LiDAR {br}')


if __name__ == '__main__':
    print(f'설정: FWD±{FWD_DEG}° 옆 {SIDE_LO}~{SIDE_HI}° 근거리 {SIDE_NEAR}m')
    for dd in sys.argv[1].split(','):
        run(dd)
