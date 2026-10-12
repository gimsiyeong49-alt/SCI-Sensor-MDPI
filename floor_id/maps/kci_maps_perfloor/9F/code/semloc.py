#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semloc.py — VLM 의미관측 + 의미지도 -> 후보 semantic 우도, 기존 LiDAR belief와 융합.

  이미지 --(Qwen3-VL-2B yes/no)--> cue별 확률 p
       --(cue별 baseline 보정)--> z (VLM 절대확률은 보정 안 돼 있어 dev로 재중심화)
  의미지도 --(후보x heading 가시성)--> 기대확률 e
       --> 베르누이 log-lik  z*log(e)+(1-z)*log(1-e)  , cue별 판별력 가중
  여러 프레임을 odom 정렬해 누적 (기존 belief와 같은 규약)
"""
import json, numpy as np
from semantic_map import SemanticMap, FUSED_CUES

EPS = 1e-3


def _logit(p, e=1e-3):
    p = np.clip(p, e, 1 - e); return np.log(p / (1 - p))


class CueCalibration:
    """dev 프레임에서 cue별 판별력(w)을 추정. z 변환 모드 2가지.

    mode='raw'      : VLM 확률을 그대로 z 로 쓴다.
        z 와 지도 기대값 e 는 '같은 사건의 확률'이어야 베르누이 비교가 성립한다.
    mode='recenter' : dev 중앙값이 0.5 가 되도록 logit 이동.
        ★결함 확인됨(2026-08-24): door_right 처럼 dev 의 98.9% 에서 yes 인 cue 는
        p=0.982 -> z=0.38 로 뒤집혀 '문 없음' 증거가 된다. 절대확률 e 와 비교 불가.
        치우친 cue(inside_room 등) 교정 목적이었으나 부작용이 더 크다.
    """

    def __init__(self, obs_by_time, cal_times, cues=FUSED_CUES, mode='raw'):
        self.mode = mode
        self.cues = cues; self.mu = {}; self.w = {}
        raw = {}
        for c in cues:
            v = np.array([obs_by_time[t][c]['p'] for t in cal_times if c in obs_by_time[t]])
            raw[c] = v
            self.mu[c] = float(np.median(_logit(v)))
            self.w[c] = float(np.std(_logit(v)))
        s = np.array([self.w[c] for c in cues]); s = s / (s.max() + 1e-9)
        for c, x in zip(cues, s): self.w[c] = float(x)
        self.raw = raw

    def z(self, obs, c):
        """관측확률 -> z. mode 참고."""
        p = float(obs[c]['p'])
        if self.mode == 'raw':
            return float(np.clip(p, 0.02, 0.98))
        return float(1 / (1 + np.exp(-(_logit(p) - self.mu[c]))))

    def report(self):
        print(f"[CueCalibration] mode={self.mode}")
        print(f"{'cue':15s} {'dev중앙p':>9s} {'판별력w':>8s}  dev분포(min/max)")
        for c in self.cues:
            v = self.raw[c]
            print(f"  {c:13s} {1/(1+np.exp(-self.mu[c])):9.3f} {self.w[c]:8.2f}  "
                  f"{v.min():.2f}/{v.max():.2f}")


def frame_ll(sm, cal, obs, heading, cues=FUSED_CUES, use_q=True, adaptive=True):
    """단일 프레임 -> 후보별 semantic log-lik [N].

    adaptive=True: cue 별 기여를 **후보 간 기대값 분산**으로 재가중한다.
      근거(2026-08-24 실측): 같은 λ 를 모든 cue 에 고정하면
        · 사물함 `left_open` 은 참 0.98 vs 추정 0.22 로 크게 갈리는데도 LiDAR 에 묻히고,
        · 게시판 후방은 참 8.0 m vs 추정 6.5 m 로 거의 안 갈리는데도 같은 힘으로 들어간다.
      → 지도가 후보들을 실제로 가르는 cue 에만 힘을 준다. 분산이 작은 cue 는 자동 소멸.
      가중 합은 균등가중과 같게 정규화해 전체 스케일을 보존한다.
    """
    e = sm.expected(heading)
    act = [c for c in cues if c in obs and cal.w[c] > 0.05]
    if not act:
        return np.zeros(sm.N)
    mask = sm.allowed if hasattr(sm, 'allowed') else np.ones(sm.N, bool)
    if adaptive:
        sd = np.array([float(np.std(e[c][mask])) for c in act])
        gain = (len(act) * sd / sd.sum()) if sd.sum() > 1e-9 else np.ones(len(act))
    else:
        gain = np.ones(len(act))
    ll = np.zeros(sm.N)
    for c, g in zip(act, gain):
        w = cal.w[c]
        q = float(obs[c].get('q', 1.0)) if use_q else 1.0
        ec = np.clip(e[c], EPS, 1 - EPS)
        z = cal.z(obs, c)
        ll += g * w * q * (z * np.log(ec) + (1 - z) * np.log(1 - ec))
    return ll


def frame_ll_bycue(sm, cal, obs, heading, cues=FUSED_CUES, use_q=True):
    """cue별 log-lik 성분을 따로 반환 {cue: [N]}. gain 은 곱하지 않는다.
    2패스 재가중(경쟁 후보 분산)에서 gain 만 바꿔 재조합하기 위한 것."""
    e = sm.expected(heading)
    out = {}
    for c in cues:
        if c not in obs or cal.w[c] <= 0.05: continue
        w = cal.w[c]
        q = float(obs[c].get('q', 1.0)) if use_q else 1.0
        ec = np.clip(e[c], EPS, 1 - EPS)
        z = cal.z(obs, c)
        out[c] = w * q * (z * np.log(ec) + (1 - z) * np.log(1 - ec))
    return out, e


def cue_gains(e_list, cues, mask):
    """cue별 gain = 후보집합(mask) 안에서의 기대값 분산 정규화.
    mask 를 '전체 허용후보' 로 주면 기존(전역) 방식, '경쟁 상위후보' 로 주면 2패스 방식."""
    act = [c for c in cues if c in e_list[0]]
    if not act: return {}
    sd = []
    for c in act:
        v = np.concatenate([e[c][mask] for e in e_list])
        sd.append(float(np.std(v)))
    sd = np.array(sd)
    g = (len(act) * sd / sd.sum()) if sd.sum() > 1e-9 else np.ones(len(act))
    return dict(zip(act, g))
