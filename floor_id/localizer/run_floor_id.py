#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_floor_id.py — 층 판별 실행기. 4층 주행 데이터를 9개 층 지도와 겨룬다.

사용법:
    python3 run_floor_id.py                 # 기본: 창 60초, t=125,155,185
    python3 run_floor_id.py 60 125 155 185  # 창 60초, 시각 지정
    python3 run_floor_id.py 30 145          # 창 30초

쓰는 것 : odom(m) · LiDAR(m) · 무축척 안내도 9장
안 쓰는 것: CAD 정합값 · CORRIDOR_M · yaml resolution · GT(채점 표시에만)

결정성: 스캔 부분추출 시드가 시각에 고정(RandomState(int(t)%97))되어
        같은 입력이면 항상 같은 결과가 나온다.
"""
import os, sys, time, numpy as np
import rt_localizer as RT
from rosbag_paths import usb_bag

SCALES = np.array([15.0, 19.5, 24.0, 28.5, 33.0, 38.0, 43.0])  # coarse search brackets; refined continuously
MID    = 24.0
DT     = 3.0
TRUE_F = 4          # 채점 표시용(알고리즘 미사용)
GAP_TH = 0.05       # metric-cost gap (m); below this, withhold floor ID


def set_ppm(Z, p):
    from metric_fit import set_scale
    set_scale(Z, p)


DELIV = '/home/user/kci_a131_1F2F5F_20260902'   # exp 전달본 (samples+odom_link+far, 로스백 없음)
DATA_BY_FLOOR = {   # 주행데이터 위치 (지도와 독립)
    '4F': ('/home/user/HANDOFF_4F_CONTINUE/data_a131',
           usb_bag(4)),
    '2F': ('/home/user/a131_out/2F', usb_bag(2)),
    '1F': ('/home/user/a131_out/1F', usb_bag(1)),
    # 5F: a1_3_1 데이터셋이 이미 완성돼 있다(협업자 인계, samples 3532 + odom_link.csv).
    #     depth 는 새로 받은 5flower 원본 bag(6666프레임) 사용 — 4층과 동일한 scene 분류.
    '5F': ('/home/user/a131_full_v0_5_tffix', usb_bag(5)),
    # ★백을 반드시 물린다. 깊이 없이 돌리면 장면이 'corridor' 로 고정되어 PRIOR 가 바뀌고
    #   φ 선택이 뒤집힌다(4F 실측: φ 305→125, 중앙 0.71m→30.69m). 깊이는 장면분류에만
    #   쓰이지만 그 결과가 방향 확정을 좌우한다.
    '1F_d': (f'{DELIV}/1F', usb_bag(1)),
    '2F_d': (f'{DELIV}/2F', usb_bag(2)),
    '3F_d': (f'{DELIV}/3F', usb_bag(3)),
    '5F_d': (f'{DELIV}/5F', usb_bag(5)),
    '6F_d': (f'{DELIV}/6F', usb_bag(6)),
    '7F_d': (f'{DELIV}/7F', usb_bag(7)),
    '8F_d': (f'{DELIV}/8F', usb_bag(8)),
    '9F_d': (f'{DELIV}/9F', usb_bag(9)),
    '4F_d': (f'{DELIV}/4F', usb_bag(4)),
}

def build(data_floor='4F'):
    """9개 층 지도를 열되 주행데이터는 data_floor 것 하나로 고정한다."""
    dd, bg = DATA_BY_FLOOR[data_floor]
    FL, MODES = {}, {}
    for f in range(1, 10):
        Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=f, ppm=MID,
                           data_dir=dd, bag=bg)
        Z.bootstrap(Z.L.T[0], boot_s=15.0)
        FL[f], MODES[f] = Z, list(Z.phis)
    return FL, MODES


SCORE_MODE = os.environ.get('SEMLOC_SCORE', 'contrast')
# 'max'      : 최고점 그대로 — 벽이 빽빽한 지도가 유리(3F 27.8%, 4F 28.3% 가 판정 86% 독점).
# 'contrast' : 최고점 - 허용후보 평균. 빽빽하면 평균도 같이 올라 밀도 이득이 상쇄된다.
#              "이 층에서 이 자리가 다른 자리보다 얼마나 특별한가" 를 재는 셈.


def _stat(Z, t, W, phi):
    """한 (층·방향·축척) 에서의 점수 통계."""
    L = Z.L
    r = Z.localize(t, W=W, dt=DT, phis=[phi])
    if SCORE_MODE == 'max': return float(r['score'])
    P = r['P'][phi][Z.allow.reshape(L.shp)]
    P = P[np.isfinite(P)]
    if not len(P): return -9e9
    return float(r['score'] - P.mean())


def score(FL, MODES, f, ts, W):
    """Compare refined metric residuals; raw pixel scores favour tiny scales."""
    Z = FL[f]
    return Z.localize_scaled(Z.L.T[0] + ts, SCALES, W=W, dt=DT)['score']


def main():
    a = sys.argv[1:]
    W  = float(a[0]) if a else 60.0
    # ★기본 시각은 성공·실패가 섞인 대표 집합. (125/155/185 만 쓰면 전부 성공해
    #   실제 성적 60% 가 가려진다 — 유리한 구간만 보이는 것을 피한다)
    TS = [float(x) for x in a[1:]] or [125.0, 185.0, 245.0, 305.0, 365.0]
    print(f'층 판별 · 창 {W:.0f}초 · 격자22 · 스케일 {len(SCALES)}개 · 판정보류 임계 {GAP_TH}')
    print(f'외부 축척값 미사용 (CAD·CORRIDOR_M·yaml). GT 는 정오 표시에만.\n')
    t0 = time.time(); FL, MODES = build()
    print(f'지도 9장 준비 {time.time()-t0:.1f}s\n')
    print(f"{'시각':>6} | " + ' '.join(f'{f}F' for f in range(1, 10)) +
          f" | {'판정':>6} {'격차':>6} {'4F':>3} {'시간':>6}")
    print('-' * 84)
    ok = held = 0
    for ts in TS:
        t1 = time.time()
        S = {f: score(FL, MODES, f, ts, W) for f in range(1, 10)}
        rk = sorted(S, key=lambda f: -S[f]); gap = S[rk[0]] - S[rk[1]]
        r4 = rk.index(TRUE_F) + 1
        if gap < GAP_TH:
            verd, held = '보류', held + 1
        else:
            verd = f'{rk[0]}F' + ('✓' if rk[0] == TRUE_F else '✗'); ok += (rk[0] == TRUE_F)
        print(f'{ts:>5.0f}s | ' + ' '.join(f'{S[f]:4.1f}' for f in range(1, 10)) +
              f' | {verd:>6} {gap:>6.2f} {r4:>3} {time.time()-t1:>5.0f}s', flush=True)
    ans = len(TS) - held
    print(f'\n  답한 {ans}/{len(TS)} · 그중 정답 {ok}/{ans}' if ans else '\n  전부 보류')
    print(f'  보류 {held} (격차 < {GAP_TH})')


if __name__ == '__main__':
    main()
