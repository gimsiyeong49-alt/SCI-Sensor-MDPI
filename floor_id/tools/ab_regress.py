# -*- coding: utf-8 -*-
"""ab_regress.py — 오늘 넣은 변경 중 무엇이 4층 위치추정을 깨뜨렸나.
기준: 변경 전 중앙 0.71m · 3m이내 91% (4F · λ_sem=0 · 창60s · PPM24 · φ시동확정)"""
import os, sys
for k in sys.argv[1:]:
    n,v=k.split('='); os.environ[n]=v
import numpy as np, rt_localizer as RT
TS=np.arange(70.0,396.0,10.0)
Z=RT.RTLocalizer(lam_sem=0.0,w_nogo=3.0,verbose=False,floor=4,ppm=24.0)
L=Z.L; phi,_=Z.fix_phi(L.T[0],boot_s=15.0)
es=[]
for ts in TS:
    t=L.T[0]+ts
    if t-60.0<L.T[0]: continue
    r=Z.localize(t,W=60.0,dt=3.0,phis=[phi]); g=L.gt_px(r['t'])
    es.append(float(np.hypot(*(np.asarray(r['pos'])-g))/L.PPM))
e=np.array(es)
tag=' '.join(sys.argv[1:]) or '(현재 상태)'
print(f'{tag:52} 중앙 {np.median(e):>6.2f}m  <3m {100*(e<3).mean():>3.0f}%  >20m {100*(e>20).mean():>3.0f}%  φ={phi}')
