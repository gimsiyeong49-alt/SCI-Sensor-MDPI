#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loo_cal.py — leave-one-out 캘리브레이션.

배경: 평가 창을 18→30초로 늘리면 13개 지점의 창이 로즈백 31~401초를 거의 전부 덮어,
      "평가창을 피한다"는 기존 규칙으로는 캘리브 표본이 40개→5개로 붕괴한다.
      실측(2026-08-24): 앞 31초만 쓰면 `stairs` 가중이 0.89→**0.00** 으로 죽는다
      (그 구간에 계단이 안 나와 '변하지 않는 cue'로 오판). 평균 가중 차이 0.211.
      반면 전체 프레임을 쓰면 현행과 평균 0.090 차이 — 실질적으로 동일.

근거: 캘리브레이션은 **GT 도 위치도 쓰지 않는 비지도 통계**(VLM 관측값의 분산)다.
      따라서 평가 창 프레임이 섞여도 정답이 샐 수 없다. 다만 '자기 자신을 보고 가중을
      정했다'는 의심을 원천 차단하기 위해 **그 지점의 창만 제외**한다.
"""
import json
from semloc import CueCalibration

MARGIN = 6.0        # 창 앞뒤 여유(초)


def loo_times(obs, t0, horizon, margin=MARGIN):
    """t0 창([t0-margin, t0+horizon+margin])을 뺀 나머지 전체 프레임 시각."""
    lo, hi = t0 - margin, t0 + horizon + margin
    return [k for k in obs if not (lo <= float(k) <= hi)]


def make(obs, t0, horizon, mode='raw', margin=MARGIN):
    """지점 t0 평가용 캘리브레이션 (그 창 제외)."""
    ts = loo_times(obs, t0, horizon, margin)
    return CueCalibration(obs, ts, mode=mode), len(ts)
