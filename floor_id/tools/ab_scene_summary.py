#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_scene_summary.py — 세 건물 × 장면 분류원 4종 성적을 한 표로."""
import json, sys
from pathlib import Path
from ab_scene_compare import load

SETS = [
    ('정보관', 'out/update0914', {'깊이':'negfit_v2','없음':'negfit_v2_nodepth','LiDAR9':'negfit_v2_lidarscene','LiDAR12':'negfit_v2_lidar12'}),
    ('공대',   'out/gong',       {'깊이':'negfit_v2','없음':'negfit_v2_nodepth','LiDAR9':'negfit_v2_lidarscene','LiDAR12':'negfit_v2_lidar12'}),
    ('소융대', 'out/sw',         {'깊이':'negfit_v2_mapscale','없음':'negfit_v2_nodepth','LiDAR9':'negfit_v2_lidarscene','LiDAR12':'negfit_v2_lidar12'}),
]
cols = ['깊이','없음','LiDAR9','LiDAR12']
print(f"{'건물':6s} {'구간':>5s} | " + ' | '.join(f'{c:^13s}' for c in cols))
print('-'*(14 + 16*len(cols)))
tot = {c: [0,0,0] for c in cols}
for bld, root, mp in SETS:
    ds = {}
    for c in cols:
        p = Path(root)/mp[c]
        ds[c] = load(p) if p.exists() else {}
    have = [c for c in cols if ds[c]]
    if not have: continue
    ks = set.intersection(*[set(ds[c]) for c in have])
    cells = []
    for c in cols:
        if not ds[c] or not ks:
            cells.append('  (미완)     '); continue
        a = sum(ds[c][k][1] == 1 for k in ks); b = sum(ds[c][k][1] <= 3 for k in ks)
        tot[c][0] += a; tot[c][1] += b; tot[c][2] += len(ks)
        cells.append(f'1위{a:3d} 상위3{b:3d}')
    print(f"{bld:6s} {len(ks):5d} | " + ' | '.join(f'{x:^13s}' for x in cells))
print('-'*(14 + 16*len(cols)))
cells = []
for c in cols:
    n = tot[c][2]
    cells.append(f'{tot[c][0]:3d}({100*tot[c][0]/n:3.0f}%) {tot[c][1]:3d}({100*tot[c][1]/n:3.0f}%)' if n else '  (미완)   ')
print(f"{'합계':6s} {tot['깊이'][2]:5d} | " + ' | '.join(f'{x:^13s}' for x in cells))
