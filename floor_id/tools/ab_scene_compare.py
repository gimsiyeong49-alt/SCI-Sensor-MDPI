#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_scene_compare.py — 장면 분류원(깊이 / 없음 / LiDAR)별 층 판별 성적 비교.

판정 규칙은 실행기와 같다: 한 구간에서 모든 지도의 twoway 점수를 내림차순 정렬, 1등이 판정.
결과 폴더의 negfit_*.json 만 읽으므로 실행 로그가 없어도 된다.

  /usr/bin/python3 ab_scene_compare.py 깊이=out/gong/negfit_v2 없음=out/gong/negfit_v2_nodepth ...
"""
import json, sys
from pathlib import Path


def load(root):
    """{(주행, 시점): (판정지도, 정답순위, 정답층)}"""
    out = {}
    for f in sorted(Path(root).glob('negfit_*.json')):
        d = json.load(open(f))
        rows = d['rows']
        n_maps = len(d['meta']['maps']) if 'meta' in d else len(set(r['map'] for r in rows))
        fl = f.stem.replace('negfit_', '')
        by = {}
        for r in rows:
            by.setdefault(int(r['ts']), {})[r['map']] = r
        for ts, win in by.items():
            if len(win) < n_maps:
                continue                       # 미완 구간은 제외
            tf = list(win.values())[0]['true']
            o = sorted(win, key=lambda m: -win[m]['neg']['twoway'])
            out[(fl, ts)] = (o[0], o.index(tf) + 1, tf)
    return out


def main():
    names, data = [], []
    for arg in sys.argv[1:]:
        name, root = arg.split('=', 1)
        names.append(name); data.append(load(root))
    if not data:
        print('인자 없음'); return
    ks = sorted(set.intersection(*[set(d) for d in data]), key=lambda x: (x[0], x[1]))
    if not ks:
        for n, d in zip(names, data):
            print(f'{n}: 완료 구간 {len(d)}개')
        print('겹치는 완료 구간이 없다'); return
    w = max(10, max(len(n) for n in names) + 2)
    print(' 구간     | ' + ' | '.join(f'{n:^{w}s}' for n in names))
    print('-' * (11 + (w + 3) * len(names)))
    base = data[0]
    stat = [[0, 0] for _ in names]; diff = [0 for _ in names]
    for k in ks:
        cells = []
        for i, d in enumerate(data):
            p, r, _ = d[k]
            stat[i][0] += r == 1; stat[i][1] += r <= 3
            if d[k][:2] != base[k][:2]: diff[i] += 1
            mk = '' if i == 0 or d[k][:2] == base[k][:2] else '*'
            cells.append(f'{p:>3d}층 {r:2d}위 {mk:1s}'.center(w))
        print(f'{k[0]:>4s} {k[1]:5d}s | ' + ' | '.join(cells))
    n = len(ks)
    print('-' * (11 + (w + 3) * len(names)))
    for i, nm in enumerate(names):
        print(f'{nm:14s} 1위 {stat[i][0]:3d} ({100*stat[i][0]/n:4.0f}%)   상위3 {stat[i][1]:3d} ({100*stat[i][1]/n:4.0f}%)'
              + ('' if i == 0 else f'   기준선과 다른 구간 {diff[i]}/{n}'))
    print(f'구간 {n}개')


if __name__ == '__main__':
    main()
