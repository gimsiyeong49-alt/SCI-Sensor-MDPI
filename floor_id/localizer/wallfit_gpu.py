#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wallfit_gpu.py — 벽정합 루프 GPU 가속. CPU 결과와 수치 동일해야 한다.

대상: rt_localizer.localize_buf 의 핵심
    ex/ey = 후보(3791) × 스캔점(400) 좌표  →  DFpx 조회  →  하위70% 평균  →  -(d/5)²
    프레임 21개 × 방향 × 스케일마다 반복 = 시점당 3200만 조회 × N

주의: 부분추출·정렬 규약을 CPU 판과 **한 글자도 다르게 하지 않는다**.
     (리팩터링에서 linspace 로 바꿨다가 사물함 0.16→31.10m 로 무너진 전례)
"""
import numpy as np
try:
    import torch
    _OK = torch.cuda.is_available()
except Exception:
    torch = None; _OK = False

_CACHE = {}

def available(): return _OK

def prep(L):
    """그 지도의 거리장·자유공간을 GPU 에 1회 올린다."""
    if not _OK: return None
    # ★캐시를 L 객체에 직접 붙인다. 전역 dict + id(L) 로 잡으면 층을 바꿔가며
    #   L 을 버릴 때 파이썬이 같은 주소를 재사용해 이전 층 격자가 되돌아온다
    #   (실측: 4F 실행 중 'cannot reshape 7245 into (117,58)' 로 중단).
    g = getattr(L, '_gpu_cache', None)
    if g is not None and g['Pf'].shape[0] == len(L.Pf) and g['H'] == L.H and g['W'] == L.W:
        return g
    if True:
        L._gpu_cache = dict(
            # ★float64: t=37s 에서 float32 최대차가 2.3e-03 이었다. argmax 는 일치했으나
            #   동점 부근에서 뒤집힐 여지가 있어 정밀도를 올린다(VRAM 8GB 로 충분).
            DF=torch.as_tensor(np.ascontiguousarray(L.DFpx, dtype=np.float64), device='cuda'),
            Pf=torch.as_tensor(np.ascontiguousarray(L.Pf, dtype=np.float64), device='cuda'),
            H=L.H, W=L.W)
    return L._gpu_cache

def wallfit(L, pts, hd, ppm):
    """CPU 판과 동일한 벽정합 점수 [N후보]. pts:(M,2) m, hd:rad, ppm:px/m."""
    g = prep(L)
    c, s = float(np.cos(hd)), float(np.sin(hd))
    P = torch.as_tensor(np.ascontiguousarray(pts, dtype=np.float64), device='cuda')
    dx = (P[:, 0]*c - P[:, 1]*s)*ppm
    dy = -(P[:, 0]*s + P[:, 1]*c)*ppm
    ex = g['Pf'][:, 0].unsqueeze(1) + dx.unsqueeze(0)
    ey = g['Pf'][:, 1].unsqueeze(1) + dy.unsqueeze(0)
    ci = torch.clamp(torch.round(ex).long(), 0, g['W']-1)
    ri = torch.clamp(torch.round(ey).long(), 0, g['H']-1)
    dd = torch.clamp(g['DF'][ri, ci], 0, 10)
    k = int(ex.shape[1]*0.7)
    v = torch.sort(dd, dim=1).values[:, :k].mean(1)
    return (-(v/5.0)**2).cpu().numpy()

def wallfit_cpu(L, pts, hd, ppm):
    """대조용 — rt_localizer 원본과 같은 식."""
    c, s = np.cos(hd), np.sin(hd)
    ex = L.Pf[:, 0][:, None] + ((pts[:, 0]*c-pts[:, 1]*s)*ppm)[None]
    ey = L.Pf[:, 1][:, None] + (-(pts[:, 0]*s+pts[:, 1]*c)*ppm)[None]
    ci = np.clip(np.round(ex).astype(int), 0, L.W-1)
    ri = np.clip(np.round(ey).astype(int), 0, L.H-1)
    dd = np.clip(L.DFpx[ri, ci], 0, 10)
    return -(np.sort(dd, 1)[:, :int(ex.shape[1]*0.7)].mean(1)/5)**2

if __name__ == '__main__':
    import time, rt_localizer as RT
    print('GPU 사용 가능:', available())
    Z = RT.RTLocalizer(lam_sem=0.0, verbose=False, floor=4, ppm=24.144); L = Z.L
    rs = np.random.RandomState(0)
    print(f'\n{"프레임":>6} {"CPU(ms)":>9} {"GPU(ms)":>9} {"배속":>6} {"최대차":>11} {"순위상관":>9}')
    print('-'*58)
    for t in (37.0, 145.0, 303.0):
        pts, yaw = L._scan(L.T[0]+t)
        if len(pts) > 400:
            pts = pts[np.random.RandomState(int(L.T[0]+t) % 97).choice(len(pts), 400, replace=False)]
        hd = np.deg2rad(306)+yaw+RT.BETA
        t0=time.time(); a=wallfit_cpu(L, pts, hd, L.PPM); tc=(time.time()-t0)*1e3
        b=wallfit(L, pts, hd, L.PPM)                      # 워밍업
        t0=time.time(); b=wallfit(L, pts, hd, L.PPM); tg=(time.time()-t0)*1e3
        d=float(np.abs(a-b).max())
        cor=float(np.corrcoef(a,b)[0,1])
        print(f'{t:>6.0f} {tc:>9.1f} {tg:>9.1f} {tc/max(tg,1e-9):>5.1f}x {d:>11.2e} {cor:>9.6f}')
        print(f'       argmax  CPU {int(np.argmax(a))}  GPU {int(np.argmax(b))}  '
              f'{"일치" if np.argmax(a)==np.argmax(b) else "★불일치"}')
    L.close()
