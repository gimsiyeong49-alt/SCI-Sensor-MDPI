#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_rt.py — 완전 인과(실시간 가능) 파이프라인 평가.

시동: 로그 시작 t=0 부터 boot_s 초만 써서 φ mode 4개 복구 (1회).
추정: 각 지점에서 '그 시각의 pose 를 직전 W초로' 추정. 미래 데이터 0.
캘리브: 창 시작 이전 관측만.
비교: 기존 비인과 격자판 성적(dev 중앙 1.67m·6/8, held 중앙 2.93m·3/5).
"""
import sys, numpy as np, rt_localizer as RT
DEV={f'd{i+1}':t for i,t in enumerate([95,119,179,199,251,279,343,375])}
HELD={'자판기':37,'LINC':54,'게시판':145,'화분끝':214,'사물함':303}
W=float(sys.argv[1]) if len(sys.argv)>1 else 30.0
BOOT=float(sys.argv[2]) if len(sys.argv)>2 else 15.0
LAM=float(sys.argv[3]) if len(sys.argv)>3 else 0.5
Z=RT.RTLocalizer(lam_sem=LAM, w_nogo=3.0, verbose=False)
phis,_=Z.bootstrap(Z.L.T[0], boot_s=BOOT)
print(f"실시간 인과 파이프라인 | 시동 {BOOT:.0f}s → φ modes {phis} | 창 {W:.0f}s(과거) | λ_sem={LAM}")
print(f"미래 데이터 사용 0 · 전역 PCA 사용 0 · GT 사용 0 (채점만)\n")
print(f"{'지점':>7} | {'오차':>8} | {'φ':>4} {'장면':>9} {'의미':>4}")
print('-'*46)
out={}
for tag,S in [('dev',DEV),('held',HELD)]:
    es=[]
    for nm,ts in S.items():
        t_now=Z.L.T[0]+ts
        r=Z.localize(t_now, W=W, phis=phis)
        gt=Z.L.gt_px(t_now)
        e=float(np.hypot(*(r['pos']-gt))/Z.L.PPM); es.append(e)
        print(f"{nm:>7} | {e:>6.2f} m | {r['phi']:>4} {r['scene']:>9} {r['nsem']:>4}",flush=True)
    e=np.array(es); out[tag]=e
    print(f"  ── {tag}: 중앙 {np.median(e):.2f}m 평균 {e.mean():.2f}m | 3m이내 {int((e<3).sum())}/{len(e)}\n")
print("=== 대조 (비인과 기존 격자판) ===")
print("  dev  중앙  1.67m | 3m이내 6/8")
print("  held 중앙  2.93m | 3m이내 3/5")
