#!/usr/bin/env python3
"""표지 멈춤 이벤트를 실시간으로 돌린다. 녹화 데이터를 실시간처럼 흘려 넣어 기존 결과와 대조할 수도 있다.

기존(녹화 데이터 전용)은 단계별 스크립트 다섯 개를 차례로 돌렸다.
  small_proposals → accumulate_proposals(--causal --window 1.2) → full_run_events regions
  → full_run_events siglip → event_trigger_sim prepare(quant 0.9, warm 50, track) + simulate(D 5)
여기서는 같은 함수를 그대로 불러 쓰고, 한꺼번에 처리하던 것을 프레임이 들어올 때마다 처리한다.
알고리즘과 값은 바꾸지 않는다(4층 전체 주행 멈춤 48번이 그대로 나와야 한다. 이 대조는 --rule event).

멈춤 규칙(--rule)
  - ask(기본, 2026-09-30~): 13_위치불확실성 최종본과 같다. 모호할 때 표지 같은 구간이 보이면
    아직 요청 안 한 벽 자리 중 1등 하나만 조준한다. 매초 계산하면 13번 요청 45번이 그대로 나온다.
  - event: 12번 줄기. 표지 같은 구간이 5 m 안에 처음 들어오면 상위 3곳.

실시간에서 달라지는 점
  - 1초마다 장면 하나를 처리한다. 그 시각 바로 뒤 프레임이 들어와야 '가장 가까운 프레임'이 정해지므로
    한 프레임(약 0.07초) 늦게 처리한다.
  - 같은 프레임의 작은 후보는 한 번만 뽑아 둔다(기존은 장면마다 다시 뽑았다. 결과는 같다).
  - SigLIP은 vlm_env의 작업자 프로세스가 매긴다(live_siglip_worker.py).

녹화 대조:
    python3 scripts/live_event_runner.py replay --out reports/.../13_실시간화/replay_4F_full
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import pickle
import struct
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import region_memory as M  # noqa: E402
import live_fast  # noqa: E402
import small_proposals  # noqa: E402
from combined_policy import match  # noqa: E402
from event_criteria import salience  # noqa: E402
from full_run_events import THR, nms  # noqa: E402
from geometry_filter import vanishing_point  # noqa: E402
from wall_region_events import depth_ok, regions  # noqa: E402

# 확정본 값(2026-09-22 4층 전체 주행 "상위 10% +트랙"). 바꾸면 기존 결과와 달라진다.
P = dict(window=1.2, step=0.2, tol=8.0, zmax=7.0, top_regions=8, radius=0.7,
         quant=0.9, warm=50, D=5.0, K=3, track_step=0.2)
VLM_PY = "/data2/vlm_env/bin/python"


def _propose(bgr, depth=None, far_m=0.0, scales=(1, 2)):
    if far_m > 0 and depth is not None:         # 2026-10-04 --far-2x-m: 2배 확대 단계는 먼 곳(깊이) 칸만
        return live_fast.propose_far(bgr, depth, far_m)
    return live_fast.propose_fast(bgr, scales)  # 기본 (1, 2) = small_proposals.propose 와 같은 결과(verify 로 확인)


def project_world(xyz, vp, pose):
    """region_memory.region_world 의 역. 월드 (x, y, 카메라 기준 높이) → 지금 화면 (u, v, 깊이 z).
    카메라 뒤(z ≤ 0.3 m)면 None. 화면 밖 좌표도 그대로 낸다."""
    X, Y, a = pose
    dx, dy = xyz[0] - X, xyz[1] - Y
    fwd, left = np.cos(a) * dx + np.sin(a) * dy, -np.sin(a) * dx + np.cos(a) * dy
    th = np.arctan2(vp[1] - M.CY, M.FY) if vp is not None else 0.0
    z = fwd * np.cos(th) + xyz[2] * np.sin(th)
    yc = fwd * np.sin(th) - xyz[2] * np.cos(th)
    if z <= 0.3:
        return None
    return float(M.CX + M.FX * -left / z), float(M.CY + M.FY * yc / z), float(z)


class Frames:
    """최근 프레임 보관. tb = 누적용 시각(녹화 대조 때 bag 시각), th = 트랙용 시각(header 시각).
    실시간에서는 둘 다 같은 촬영 시각이다. 기존 스크립트가 쓰던 시각 기준을 그대로 따른다."""

    def __init__(self, keep_sec=4.0):
        self.keep = keep_sec
        self.items = []                          # dict(id, tb, th, bgr, gray)

    def add(self, fid, tb, th, bgr):
        self.items.append(dict(id=fid, tb=tb, th=th, bgr=bgr, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
        while self.items and self.items[-1]["tb"] - self.items[0]["tb"] > self.keep:
            self.items.pop(0)

    def nearest_tb(self, t):
        """accumulate_proposals.nearest 와 같은 규칙(같은 거리면 앞 프레임)."""
        times = [f["tb"] for f in self.items]
        i = bisect.bisect_left(times, t)
        if i == 0:
            return self.items[0]
        if i >= len(times):
            return self.items[-1]
        return self.items[i] if abs(times[i] - t) < abs(times[i - 1] - t) else self.items[i - 1]

    def nearest_th(self, t):
        """event_trigger_sim._propagate 와 같은 규칙(np.argmin)."""
        times = np.array([f["th"] for f in self.items])
        return self.items[int(np.argmin(abs(times - t)))]

    def newest(self):
        return self.items[-1] if self.items else None


class SiglipClient:
    def __init__(self, py=VLM_PY):
        worker = str(Path(__file__).parent / "live_siglip_worker.py")
        self.p = subprocess.Popen([py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=os.getcwd())
        assert self._read().get("ready"), "SigLIP 작업자가 준비되지 않았습니다"

    def _read(self):
        head = self.p.stdout.read(4)
        if len(head) < 4:
            raise RuntimeError("SigLIP 작업자가 죽었습니다")
        return pickle.loads(self.p.stdout.read(struct.unpack("<I", head)[0]))

    def score(self, bgr, regs):
        b = pickle.dumps({"bgr": bgr, "regions": regs, "board": getattr(self, "want_board", False)}, protocol=4)
        self.p.stdin.write(struct.pack("<I", len(b)) + b)
        self.p.stdin.flush()
        r = self._read()
        self.last_adv = np.array(r.get("adv", []))      # 표지 − 헷갈리는 것(2026-10-01)
        self.last_board = np.array(r.get("board", []))  # 넓은 조각의 게시판 − 작은 표지(2026-10-01)
        self.last_ceil = np.array(r.get("ceil", []))    # 넓은 조각의 천장 조명 − 작은 표지(2026-10-03)
        self.last_tiles = r.get("tiles", [])            # 구간마다 SigLIP 최고 조각(2026-10-05)
        return np.array(r["sim"]), r["sec"]

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:      # torch 정리가 오래 걸린다(2026-09-30). 결과는 이미 받았다
            self.p.kill()


class VlmClient(SiglipClient):
    """시설확인용 VLM 작업자(live_vlm_worker.py). 화면에 보이는 것을 열린 질문으로 받는다."""

    def __init__(self, py=VLM_PY):
        worker = str(Path(__file__).parent / "live_vlm_worker.py")
        self.p = subprocess.Popen([py, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, cwd=os.getcwd())
        assert self._read().get("ready"), "VLM 작업자가 준비되지 않았습니다"

    def ask(self, bgr):
        b = pickle.dumps({"bgr": bgr}, protocol=4)
        self.p.stdin.write(struct.pack("<I", len(b)) + b)
        self.p.stdin.flush()
        r = self._read()
        return r["answer"], r["sec"]

    def ask_async(self, bgr):
        """질문을 보내고 바로 돌아온다. 답은 wait() 로 받는다(다음 장면에서. 0.36초라 보통 이미 와 있다)."""
        b = pickle.dumps({"bgr": bgr}, protocol=4)
        self.p.stdin.write(struct.pack("<I", len(b)) + b)
        self.p.stdin.flush()

    def wait(self):
        r = self._read()
        return r["answer"], r["sec"]

    def boxes(self, bgr, label):
        b = pickle.dumps({"bgr": bgr, "box_label": label}, protocol=4)
        self.p.stdin.write(struct.pack("<I", len(b)) + b)
        self.p.stdin.flush()
        r = self._read()
        return r.get("boxes", []), r["sec"]


# 시설 단어(열린 질문 답에서 찾음)와 상자 질문 이름. 2026-10-01: 처음엔 계단만 받았으나 4층 106초 엘베를 놓쳐
# (내 판정 실수로 엘베 답을 전부 오답 처리) 셋 다 받고, 대신 상자 깊이 5 m 안일 때만 이벤트로 낸다.
FACILITY_WORDS = {"계단": ("stair",), "엘리베이터": ("elevator",),
                  "화장실": ("restroom", "toilet", "bathroom", "washroom")}
FACILITY_BOX = {"계단": "staircase", "엘리베이터": "elevator door", "화장실": "restroom entrance or restroom sign"}


class FacilityMap:
    """의미지도로 '지금 시설이 보여야 하나'를 판단한다(2026-10-04). 위치추정 top-K 중 1위 점수 −1.5 안의 후보마다
    (위치, 방향 yaw)에서 앞쪽 ±half_fov 도, max_m 미터 안에 그 시설(상자 가까운 점 또는 중심)이 있는지 본다.
    하나라도 그렇다고 하면 참. 1위 하나만 믿지 않는다(1위가 튀거나 틀리면 같이 틀어지므로).
    지도는 미터가 정확하지 않아(안내도 기반) 넉넉하게 본다. yaw 는 지도 m 좌표(오른쪽 x, 위쪽 y) 기준
    (2026-10-04 G1·7·3층 확실 구간에서 1위 이동 방향과 맞춰 확인). 위치추정 기록은 t 이전 것만 쓴다."""
    TYPES = {"계단": "stair", "엘리베이터": "elevator", "화장실": "restroom"}

    def __init__(self, topk_csv, sem_json, ppm, kinds, max_m=8.0, half_fov=60.0, near=1.5, base_png=""):
        tk = {}
        for r in (csv.DictReader(open(topk_csv)) if topk_csv else []):   # 실시간은 빈 채로 시작해 add 로 받는다
            tk.setdefault(float(r["relative_sec"]), []).append(
                (float(r["score"]), float(r["x_px"]), float(r["y_px"]), float(r["yaw_rad"])))
        self.ts = sorted(tk)
        self.tk = tk
        d = json.load(open(sem_json))
        self.fac = {k: [s["bbox_px"] for s in d["spaces"] if s.get("type") == v] for k, v in self.TYPES.items()
                    if k in kinds}
        self.ppm, self.max_m, self.half = ppm, max_m, np.radians(half_fov)
        self.near = near
        self.topo = None                         # base_png 를 주면 위상 방식(scripts/topo_facility.py, 미터 안 씀)
        if base_png:
            from topo_facility import TopoMap
            self.topo = TopoMap(base_png, sem_json)

    def add(self, t, cands, ppm=None):
        """실시간: 위치추정 기록 하나(cands = [(score, x_px, y_px, yaw_rad), ...])를 붙인다. 시각은 늘어나는 순서."""
        if self.ts and t <= self.ts[-1]:
            return
        self.ts.append(t)
        self.tk[t] = [tuple(float(v) for v in c[:4]) for c in cands]
        if ppm:
            self.ppm = ppm

    def expect(self, t):
        """t 에서 앞쪽에 보여야 할 시설 종류 목록."""
        i = bisect.bisect_right(self.ts, t + 1e-6) - 1
        if i < 0:
            return []
        c = self.tk[self.ts[i]]
        top = max(x[0] for x in c)
        if self.topo is not None:                # 2026-10-04 위상: 보임·앞쪽·가까운 순서(안내도는 미터 지도가 아님)
            return [k for k in self.fac if any(self.topo.match(x, y, yaw, k) for sc, x, y, yaw in c if sc >= top - self.near)]
        out = []
        for k, boxes in self.fac.items():
            for sc, x, y, yaw in c:
                if sc < top - self.near:
                    continue
                for bx, by, bw, bh in boxes:
                    pts = [(min(max(x, bx), bx + bw), min(max(y, by), by + bh)), (bx + bw / 2, by + bh / 2)]
                    for px, py in pts:
                        dx, dy = (px - x) / self.ppm, -(py - y) / self.ppm      # 픽셀 y 아래 → 지도 y 위
                        dist = np.hypot(dx, dy)
                        if dist <= self.max_m and (dist < 1.0 or
                                                   abs(np.angle(np.exp(1j * (np.arctan2(dy, dx) - yaw)))) <= self.half):
                            out.append(k)
                            break
                    if k in out:
                        break
                if k in out:
                    break
        return out


class LiveEvents:
    def __init__(self, siglip, pool=None, slow=False, sparse=False):
        self.slow = slow                         # True 면 기존 파이썬 반복(대조용)
        self.sparse = sparse                     # True 면 '드물게 찾고 따라가기'(알고리즘 변경, 검증 대상)
        self.hist = []                           # sparse: 지난 장면들의 (프레임, 후보)
        self.n_prop = 0                          # 후보를 뽑은 프레임 수(부하 비교용, 미리 뽑기 포함)
        self.n_hist, self.rate = 2, 1            # sparse: 지난 장면 몇 개, 1초에 몇 장
        self.vlm = siglip
        self.pool = pool
        self.props = {}                          # 프레임 id → 작은 후보(한 번만 뽑음)
        self.pending = {}                        # 프레임 id → 미리 보낸 작업(AsyncResult)
        self.sent_g = set()                      # 미리 뽑기를 보낸 격자 시각
        self.spots, self.seen = [], []
        self.fired, self.fired_t = set(), set()
        self.fired_at = {}                       # 진단(2026-10-05): 벽 자리 → [멈춘 시각, 상자]
        self.next_tid, self.prev = 0, None
        # 멈춤 규칙. event: 12번 줄기(5 m 안에 처음 → 상위 K개). ask: 13번 방식(모호할 때 표지 같은 구간이
        # 보이면 아직 요청 안 한 벽 자리 중 1등 하나만 조준). ask_dmax > 0 이면 그 거리 안의 구간만.
        # 2026-09-30 기본을 ask 로 바꿨다(4층: event 28번 멈춤·51곳 조준·번호판 9 → ask 46번·46곳·10).
        self.rule, self.ask_dmax, self.ask_near = "ask", 0.0, False
        self.sign_conf, self.adv_of = False, {}  # --sign-confusers(2026-10-01, 시험 결과 쓰지 않음)
        self.sign_max_w = 0.85                   # --sign-max-width: 조준 후보 실제 폭 상한(m), 0이면 안 봄(2026-10-04 기본 0.85)
        self.sign_max_h = 0.0                    # --sign-max-height: 조준 후보 실제 높이 상한(m). 문·문틀·게시판을 뺀다
        self.sign_edge = 0                       # --sign-edge-px: 화면 가장자리에서 이 픽셀 안에 닿은(잘린) 구간은 뺀다
        self.sign_max_group_w = 0.0              # --sign-max-group-width: 잘리기 전 원래 묶음의 실제 폭 상한(m)
        self.log_groups, self.group_of = False, {}
        self.cam_pitch, self.ceil_min_elev = 11.0, 14.6  # --cam-pitch-deg, --ceil-min-elev(천장 조명 판정)
        self.ceil_filter, self.ceil_of = False, {}  # --ceiling-filter: 천장 조명(포화 또는 천장 점수+화면 위쪽)은 조준 안 함
        self.board_max, self.board_of = 0.02, {}  # 2026-10-04 기본 0.02. --board-max: 넓은 조각 '게시판 − 작은 표지'가 이보다 크면 조준 안 함
        self.last_like_n = 0                     # 이번 장면 조준 후보 수(시설확인 조건)
        self.fac_vlm, self.fac_run, self.fac_last_t, self.fac_last_p = None, 0, -1e9, None
        self.fac_log = []                        # 시설확인 질문 기록(답이 시설이 아니어도)
        self.fac_second = False                  # 한 번 묻고 못 찾았으면 다음 장면에 한 번 더
        self.fac_pending = None                  # (종류, 찾은 시각): 찾았지만 5 m 밖 → 몇 초 동안 거리만 다시 잰다
        # --floor-band: 바닥 기준 높이 띠(2026-10-04). 바닥면은 주행 앞부분 깊이로 잰다(카메라 장착 가정 없음).
        self.floor_band, self.floor_fits, self.floor_up, self.floor_camh = (0.6, 2.2), [], None, None
        self.floor_need = 20                     # 바닥면을 이만큼 찾으면 위 방향·카메라 높이를 고정(중앙값)
        self.ask_same_m, self.fired_xyz = 0.0, []
        self.ask_same_sec, self.fired_t = 0.0, []    # --ask-same-sec(2026-10-08): 최근 이 초 안에 멈춘 자리만 같은 대상으로 본다(0 = 시간 제한 없음)
        self.ask_min_scenes = 1                  # --ask-min-scenes  # --ask-same-m: 멈춘 자리 기억(벽 자리 기억 0.7 m 와 별도)
        self.scales = (1,)                       # --prop-scales: 작은 후보를 뽑을 배율(2026-10-04 2배 빼기 시험)
        self.far_m, self.depth_of = 0.0, None   # --far-2x-m: 2배 확대 후보 뽑기를 이 거리(m)보다 먼 곳만(0이면 전체)
        self.fac_v2, self.fac_map, self.fac_ask = False, None, None   # --facility-v2(2026-10-04)
        self.fac_nomap_accept = False            # --fac-nomap-accept: 지도에 그 종류가 없으면 답만으로 받음(G1 엘베 4건 전부 오답이라 기본 끔)
        self.fac_free_kinds = ("계단",)         # v2: 지도 예측 없이도 받는 시설(나머지는 지도가 예측할 때만)
        self.fac_hit_p = None                    # v2: 마지막으로 시설을 찾은 자리(이 근처 dedup_m 안에서는 다시 안 물음)
        self.sim_floor = 0.0                     # --sim-floor: 이번 장면 SigLIP 점수 바닥값(0 = 안 봄)
        self.spot_recent = 0                     # --spot-recent: 벽 자리 점수를 최근 N번 관측 평균으로(0 = 전체 평균, 기존)
        self.cand_tile = False                   # --cand-tile: 구간 대신 SigLIP 최고 조각을 후보로(2026-10-05)
        self.sig_w = 1.0                         # --sig-weight: 최종 순위에서 SigLIP 순위 비중(1 = 기존 반반)
        self.fac_kinds = ("계단",)               # 이벤트로 받을 시설. 2026-10-01 5·6층: 엘베 4번 중 1번만 맞아 계단만

    # ---- 1. 작은 후보 + 과거 1.2초 누적 (accumulate_proposals.main 과 같은 계산) ----
    def _depth(self, f):
        """프레임 f 의 깊이(2배 확대 영역 고르기용). depth_of 가 없거나 far_m 이 0 이면 None."""
        if not self.far_m or self.depth_of is None:
            return None
        return self.depth_of(f["tb"])

    def prefetch(self, buf, t_next):
        """프레임이 들어올 때마다 부른다. 다음 장면(t_next)이 쓸 격자 시각(t_next-1.2 … t_next, 0.2초 간격)
        가운데 가장 가까운 프레임이 정해진 것(그 시각 뒤 프레임이 들어온 것)은 후보 뽑기를 미리 보낸다.
        빗나가도 장면을 처리할 때 _need_props 가 다시 뽑으므로 결과는 같다."""
        if self.pool is None or t_next is None:
            return
        newest = buf.newest()["tb"]
        if self.sparse:
            grid = [0.0] + ([-0.5] if self.rate == 2 else [])
        else:
            grid = np.arange(-P["window"], 0.0 + 1e-6, P["step"])
        for off in grid:
            g = round(t_next + float(off), 6)
            if g in self.sent_g or newest <= g:
                continue
            self.sent_g.add(g)
            f = buf.nearest_tb(t_next + off)
            if f["id"] not in self.props and f["id"] not in self.pending:
                self.pending[f["id"]] = self.pool.apply_async(_propose, (f["bgr"], self._depth(f), self.far_m, self.scales))
                self.n_prop += 1
        if len(self.sent_g) > 100:
            self.sent_g = {g for g in self.sent_g if g > t_next - 5}

    def _need_props(self, frames):
        for f in frames:
            if f["id"] in self.pending and f["id"] not in self.props:
                self.props[f["id"]] = self.pending.pop(f["id"]).get()
        for k in [k for k in self.pending if self.pending[k].ready()]:
            if k not in self.props:
                self.props[k] = self.pending[k].get()
            self.pending.pop(k)
        todo = [f for f in frames if f["id"] not in self.props]
        self.n_prop += len(todo)
        if not todo:
            return
        args_ = [(f["bgr"], self._depth(f), self.far_m, self.scales) for f in todo]
        outs = self.pool.starmap(_propose, args_) if self.pool else [_propose(*x) for x in args_]
        for f, b in zip(todo, outs):
            self.props[f["id"]] = b
        if len(self.props) > 200:
            for k in sorted(self.props)[:len(self.props) - 200]:
                self.props.pop(k)

    def accumulate(self, buf, t):
        base = buf.nearest_tb(t)
        offsets = [o for o in np.arange(-P["window"], 0.0 + 1e-6, P["step"]) if abs(o) > 1e-6]
        neigh = [buf.nearest_tb(t + off) for off in offsets]
        self._need_props([base] + neigh)
        boxes = self.props[base["id"]]
        support = [0] * len(boxes)
        added = []
        for off, nf in zip(offsets, neigh):
            if nf["id"] == base["id"]:
                continue
            nb = self.props[nf["id"]]
            if not nb:
                continue
            c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in nb], np.float32).reshape(-1, 1, 2)
            pts, st, _ = cv2.calcOpticalFlowPyrLK(nf["gray"], base["gray"], c, None, winSize=(21, 21), maxLevel=3)
            if not self.slow:
                live_fast.attach(boxes, nb, pts, st, off, P["tol"], support, added)
                continue
            for k, (p, ok) in enumerate(zip(pts.reshape(-1, 2), st.reshape(-1))):
                if not ok or not (0 <= p[0] < 640 and 0 <= p[1] < 480):
                    continue
                best, bd = None, 1e9
                for j, b in enumerate(boxes + [a["box"] for a in added]):
                    d = abs((b[0] + b[2]) / 2 - p[0]) + abs((b[1] + b[3]) / 2 - p[1])
                    if d < bd:
                        best, bd = j, d
                if best is not None and bd <= P["tol"]:
                    if best < len(boxes):
                        support[best] += 1
                    else:
                        added[best - len(boxes)]["support"] += 1
                else:
                    w, h = nb[k][2] - nb[k][0], nb[k][3] - nb[k][1]
                    added.append({"box": (float(p[0] - w / 2), float(p[1] - h / 2),
                                          float(p[0] + w / 2), float(p[1] + h / 2)), "support": 1})
        recs = [{"box": [float(v) for v in b], "support": s} for b, s in zip(boxes, support)]
        recs += [{"box": [float(v) for v in a["box"]], "support": a["support"]} for a in added]
        return base, recs

    def accumulate_sparse(self, buf, t, max_gap=3.5):
        """드물게 찾고 따라가기(2026-09-30 시험). 후보는 장면 프레임에서만 뽑고, 지난 n_hist 장면의 후보를
        0.2초 간격 프레임의 광학 흐름으로 지금 프레임까지 옮겨 와 '계속 보이는지'를 센다.
        이어 붙이기 규칙(허용 거리 8 px, 못 만나면 새 후보로 추가)은 기존과 같다. 기존은 과거 1.2초의
        이웃 프레임 6장에서 후보를 새로 뽑았다(1초에 5장). 여기서는 1초에 1장만 뽑는다."""
        n_hist = self.n_hist
        base = buf.nearest_tb(t)
        extra = []
        if self.rate == 2:                       # 1초에 2장: 0.5초 전 프레임도 후보를 뽑아 '지난 장면'처럼 쓴다
            mid = buf.nearest_tb(t - 0.5)
            if mid["id"] != base["id"]:
                self._need_props([mid])
                extra = [dict(fid=mid["id"], th=mid["th"], gray=mid["gray"], boxes=self.props[mid["id"]])]
        self._need_props([base])
        boxes = self.props[base["id"]]
        support = [0] * len(boxes)
        added = []
        for h in self.hist[-n_hist:] + extra:
            if h["fid"] == base["id"] or base["th"] - h["th"] > max_gap or not h["boxes"]:
                continue
            nb = h["boxes"]
            cur = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in nb], np.float32).reshape(-1, 1, 2)
            ok = np.ones(len(nb), bool)
            prev = h["gray"]
            for tt in list(np.arange(h["th"] + P["step"], base["th"], P["step"])) + [base["th"]]:
                g = buf.nearest_th(tt)["gray"]
                if g is prev:
                    continue
                cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, cur, None, winSize=(21, 21), maxLevel=3)
                ok &= st.reshape(-1).astype(bool)
                prev = g
            live_fast.attach(boxes, nb, cur, ok.astype(np.uint8), 0.0, P["tol"], support, added)
        self.hist.append(dict(fid=base["id"], th=base["th"], gray=base["gray"], boxes=boxes))
        self.hist = self.hist[-n_hist:]
        recs = [{"box": [float(v) for v in b], "support": s} for b, s in zip(boxes, support)]
        recs += [{"box": [float(v) for v in a["box"]], "support": a["support"]} for a in added]
        return base, recs

    # ---- 2. 벽 구간(7 m, 바닥 제외, 조각 1개 점수, 상위 8개) ----
    def wall_regions(self, base, recs, dimg):
        bgr, gray = base["bgr"], base["gray"]
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        sal = [salience(gray, grad, r["box"]) for r in recs]
        vp = vanishing_point(bgr)
        keep = [depth_ok(dimg, r["box"], vp, P["zmax"])[0] for r in recs]
        if self.sign_max_group_w > 0 or self.log_groups:   # 같은 구간 + 잘리기 전 원래 묶음 상자(2026-10-01)
            rx = live_fast.regions_ext(recs, sal, THR, keep=keep, topk=1)[:P["top_regions"]]
            regs = [json.loads(json.dumps(b)) for _, b, _ in rx]
            self.group_of = {tuple(float(v) for v in r): g for r, (_, _, g) in zip(regs, rx)}
        else:
            regs = [json.loads(json.dumps(b)) for _, b in regions(recs, sal, THR, keep=keep, topk=1)[:P["top_regions"]]]
        return regs, vp

    # ---- 3. SigLIP 재정렬 + 겹침 합치기 ----
    def rerank(self, bgr, regs):
        if not regs:
            return [], 0.0
        sim, sec = self.vlm.score(bgr, regs)
        adv = getattr(self.vlm, "last_adv", np.zeros(0))
        self.adv_of = {tuple(float(v) for v in regs[i]): float(adv[i]) for i in range(min(len(regs), len(adv)))}
        bd = getattr(self.vlm, "last_board", np.zeros(0))
        self.board_of = {tuple(float(v) for v in regs[i]): float(bd[i]) for i in range(min(len(regs), len(bd)))}
        if self.ceil_filter:                     # (포화 비율, 천장 점수)
            ce = getattr(self.vlm, "last_ceil", np.zeros(0))
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            for i in range(min(len(regs), len(ce))):
                x0, y0, x1, y1 = (int(v) for v in regs[i][:4])
                sat = float((g[y0:y1, x0:x1] >= 245).mean()) if x1 > x0 and y1 > y0 else 0.0
                self.ceil_of[tuple(float(v) for v in regs[i])] = (sat, float(ce[i]))
        r_sig = np.argsort(np.argsort(-sim))
        # 최종 순위 = 벽 구간 순위 + sig_w × SigLIP 순위(작을수록 앞). sig_w 1 = 기존(반반). 2026-10-05 4층 분석: 번호판 든 구간이
        # SigLIP 1등 26/55 인데 최종 1등 15/55 — 반반 평균에서 밀림(scripts/door_phrase_funnel.py)
        w = self.sig_w
        order = nms(sorted(range(len(regs)), key=lambda i: (i + w * r_sig[i], i)), regs)
        if self.cand_tile:                       # 2026-10-05: 구간 대신 그 안에서 SigLIP 이 가장 높던 조각(≤100 px)을 후보로.
            # 4층 분석: 번호판이 든 구간은 번호판 면적의 13배라 폭 거르기·벽 자리 기억(0.7 m)에서 옆 포스터·문과 섞였다.
            # 거르기 값(게시판·헷갈림·천장·원래 묶음)은 구간에서 잰 것을 조각에도 그대로 붙인다.
            tiles = getattr(self.vlm, "last_tiles", [])
            out = []
            for i in order:
                if i < len(tiles) and tiles[i]:
                    k0, k1 = tuple(float(v) for v in regs[i]), tuple(float(v) for v in tiles[i])
                    for d in (self.adv_of, self.board_of, self.ceil_of, self.group_of):
                        if k0 in d:
                            d[k1] = d[k0]
                    out.append([float(sim[i]), [float(v) for v in tiles[i]]])
                else:
                    out.append([float(sim[i]), regs[i]])
            return out, sec
        return [[float(sim[i]), regs[i]] for i in order], sec

    # ---- 4. 벽 자리 기억 + 트랙 + 멈춤 판단 (event_trigger_sim.prepare + simulate 한 장면분) ----
    def _propagate(self, buf, t_a, t_b, pts):
        if len(pts) == 0:
            return []
        cur = np.array(pts, np.float32).reshape(-1, 1, 2)
        ok = np.ones(len(pts), bool)
        prev = None
        for t in list(np.arange(t_a, t_b, P["track_step"])) + [t_b]:
            g = buf.nearest_th(t)["gray"]
            if prev is not None:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, cur, None, winSize=(21, 21), maxLevel=3)
                ok &= st.reshape(-1).astype(bool)
                cur = nxt
            prev = g
        return [tuple(p) if o else None for p, o in zip(cur.reshape(-1, 2), ok)]

    def decide(self, buf, t, regs, dimg, vp, pose, allow_stop=True):
        ids, zs, xyzs, taken = [], [], [], set()
        for sim, b in regs:
            xyz, z = M.region_world(b, dimg, vp, pose)
            xyzs.append(xyz)
            if xyz is None:
                ids.append(None)
                zs.append(None)
                continue
            a = match(self.spots, xyz, P["radius"], None, taken)
            self.spots[a]["sims"].append(float(sim))
            self.spots[a].setdefault("ts", []).append(float(t))   # 2026-10-04 일관성 기록(멈춤 판단에는 안 씀)
            taken.add(a)
            ids.append(a)
            zs.append(z)
        self.seen.extend(float(s) for s, _ in regs)
        base = float(np.quantile(self.seen, P["quant"])) if len(self.seen) >= P["warm"] else float("inf")
        tids = [None] * len(regs)
        if self.prev is not None and self.prev["regs"]:
            cen = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in self.prev["regs"]]
            moved = self._propagate(buf, self.prev["t"], t, cen)
            for pj, p in enumerate(moved):
                if p is None:
                    continue
                for j, (_, b) in enumerate(regs):
                    if tids[j] is None and b[0] - 10 <= p[0] <= b[2] + 10 and b[1] - 10 <= p[1] <= b[3] + 10:
                        tids[j] = self.prev["tids"][pj]
                        break
        for j in range(len(regs)):
            if tids[j] is None:
                tids[j] = self.next_tid
                self.next_tid += 1
        # 벽 자리 점수 = 그 자리 관측 점수 평균. --spot-recent N 이면 최근 N번만(2026-10-05: 다가가며 번호판이 커져 지금 점수가
        # 기준을 넘어도, 멀리서 작게 보일 때의 낮은 점수가 평균을 끌어내려 멈추지 못했다. 4층 정답 구간 24장면 중 9장면)
        nr = self.spot_recent
        ms = [regs[j][0] if ids[j] is None else float(np.mean(self.spots[ids[j]]["sims"][-nr:] if nr > 0 else self.spots[ids[j]]["sims"]))
              for j in range(len(regs))]
        rm = np.argsort(np.argsort(-np.array(ms))) if regs else []
        order = sorted(range(len(regs)), key=lambda j: (j + rm[j], j))
        self.prev = dict(t=t, regs=regs, tids=tids)
        like = [j for j in range(len(regs)) if ms[j] >= base]
        if self.sim_floor > 0:                   # 2026-10-06 --sim-floor: 상대 기준(상위 10%)에 더해 이번 장면 SigLIP 점수 바닥값.
            # 표지판이 적은 층(G2F)에서는 상위 10% 안에 점수 낮은 배관·천장까지 들어왔다(판정 표지판 하위 5% = 0.078 아래)
            like = [j for j in like if regs[j][0] >= self.sim_floor]
        if self.sign_conf:                       # 헷갈리는 것(포스터·조명·창문·문…)이 이기는 구간은 조준 후보에서 뺀다
            like = [j for j in like if self.adv_of.get(tuple(float(v) for v in regs[j][1]), 1.0) > 0]
        if self.sign_max_w > 0:                  # 실제 폭이 큰 것(게시판·포스터)은 빼고, 깊이 없는 구간은 남긴다
            like = [j for j in like if zs[j] is None
                    or (regs[j][1][2] - regs[j][1][0]) * zs[j] / M.FX <= self.sign_max_w]
        if self.sign_max_h > 0:                  # 실제 높이가 큰 것(문·문틀·세로로 긴 게시물)은 뺀다
            like = [j for j in like if zs[j] is None
                    or (regs[j][1][3] - regs[j][1][1]) * zs[j] / M.FY <= self.sign_max_h]
        if self.sign_edge > 0:                   # 화면 끝에 잘린 구간은 뺀다
            h, w = dimg.shape[:2]
            e = self.sign_edge
            like = [j for j in like if regs[j][1][0] > e and regs[j][1][1] > e
                    and regs[j][1][2] < w - 1 - e and regs[j][1][3] < h - 1 - e]
        if self.sign_max_group_w > 0:            # 잘리기 전 원래 묶음이 큰 것(게시판의 한 조각)도 뺀다
            def gw(j):
                g = self.group_of.get(tuple(float(v) for v in regs[j][1]))
                return None if g is None or zs[j] is None else (g[2] - g[0]) * zs[j] / M.FX
            like = [j for j in like if gw(j) is None or gw(j) <= self.sign_max_group_w]
        if self.floor_band is not None and self.floor_up is not None:
            # 바닥 기준 높이 띠(2026-10-04): 사람 판정 117개(4개 층)에서 0.6~2.2 m 는 번호판·표지 47/47 유지, 오답 22/70 제거.
            lo, hi = self.floor_band
            def in_band(j):
                hgt = self.box_floor_height(dimg, regs[j][1])
                return hgt is None or lo <= hgt <= hi
            like = [j for j in like if in_band(j)]
        if self.ceil_filter:                     # 천장 조명: 하얗게 날아갔거나, 화면 위쪽에서 천장 점수가 높으면 뺀다
            def is_ceil(j):
                b = regs[j][1]
                sat, ce = self.ceil_of.get(tuple(float(v) for v in b), (0.0, -1.0))
                # 화면 높이는 '카메라 수평선 위 각도'로 본다(2026-10-04: 처음엔 상자 중심 y < 200 픽셀이었다.
                # 피치 11°·14.6° 이면 y < 200.4 와 같다. 장면별 소실점 기준은 소실점이 흔들려 조명을 3개 덜 걸렀다)
                hy = M.CY + M.FY * np.tan(np.radians(self.cam_pitch))
                elev = np.degrees(np.arctan((hy - (b[1] + b[3]) / 2) / M.FY))
                return sat > 0.05 or (ce > -0.01 and elev > self.ceil_min_elev)
            like = [j for j in like if not is_ceil(j)]
        if self.board_max:                       # 게시판·포스터로 보이는 구간은 조준 후보에서 뺀다(넓게 잘라 판단)
            like = [j for j in like if self.board_of.get(tuple(float(v) for v in regs[j][1]), -1.0) <= self.board_max]
        self.last_like_n = len(like)
        if self.log_groups:                      # 진단(2026-10-05): 구간마다 조준 후보에서 빠진 이유(판단에는 안 씀)
            self.last_why = dict(order=list(order), why=[self._why(j, regs, ms, base, zs, ids, dimg) for j in range(len(regs))],
                                 ms=[float(v) for v in ms], base=float(base), ids=[None if a is None else int(a) for a in ids],
                                 nobs=[None if a is None else len(self.spots[a]["sims"]) for a in ids],
                                 fired_at=[self.fired_at.get(a) for a in ids], z=zs,
                                 hgt=[None if self.floor_up is None else self.box_floor_height(dimg, r[1]) for r in regs])
        if self.rule == "ask":
            return self._ask(order, like, ids, zs, xyzs, regs, t, allow_stop), base
        near = [j for j in like if zs[j] is not None and zs[j] <= P["D"]]
        new = [j for j in near if ids[j] not in self.fired and tids[j] not in self.fired_t]
        if not new or not allow_stop:            # 위치가 확실하면 기억만 쌓고 멈추지도, '멈춤'으로 적지도 않는다
            return None, base
        picks = [regs[j][1] for j in order if j in like][:P["K"]]
        for j in near:
            if ids[j] is not None:
                self.fired.add(ids[j])
            self.fired_t.add(tids[j])
        return dict(t=t, picks=picks, z=[zs[j] for j in order if j in like][:P["K"]]), base

    def _why(self, j, regs, ms, base, zs, ids, dimg):
        """구간 j 가 조준 후보(like)·요청에서 빠지는 이유 목록(decide 의 거르기를 하나씩 따로 본다)."""
        b, out = regs[j][1], []
        if ms[j] < base:
            out.append("기준미달")
        if self.sim_floor > 0 and regs[j][0] < self.sim_floor:
            out.append("바닥값미달")
        if self.sign_max_w > 0 and zs[j] is not None and (b[2] - b[0]) * zs[j] / M.FX > self.sign_max_w:
            out.append("폭")
        if self.floor_band is not None and self.floor_up is not None:
            hgt = self.box_floor_height(dimg, b)
            if hgt is not None and not (self.floor_band[0] <= hgt <= self.floor_band[1]):
                out.append("높이띠")
        if self.board_max and self.board_of.get(tuple(float(v) for v in b), -1.0) > self.board_max:
            out.append("게시판")
        if ids[j] is not None and ids[j] in self.fired:
            out.append("이미요청")
        return out

    def _ask(self, order, like, ids, zs, xyzs, regs, t, allow_stop):
        """13번 방식(render_observation_link_video.py --dedup_m 0.7 과 같은 규칙).
        모호할 때만, 표지 같은 구간 중 이미 요청 보낸 벽 자리가 아닌 것의 1등 하나를 낸다.
        깊이가 없어 벽 자리를 못 정한 구간은 13번처럼 걸러지지 않는다."""
        if not allow_stop:
            return None
        cand = [j for j in order if j in like and ids[j] not in self.fired]
        if self.ask_min_scenes > 1:              # 2026-10-04: 그 벽 자리가 서로 다른 장면 N번 이상 잡혔을 때만(한 번 반짝 오답 거르기)
            cand = [j for j in cand if ids[j] is not None
                    and len(set(self.spots[ids[j]].get("ts", []))) >= self.ask_min_scenes]
        if self.ask_same_m > 0 and self.fired_xyz:   # 2026-10-04: 이미 멈춘 자리에서 수평 ask_same_m 안이면 같은 대상으로 본다
            F = np.array(self.fired_xyz)
            if self.ask_same_sec > 0:                # 2026-10-08: 오래전(돌아오며 다시 보기)은 막지 않는다. 위치추정이 60초 관측만 쓰므로 다시 읽으면 도움
                F = F[np.array(self.fired_t) >= t - self.ask_same_sec]
            if len(F):
                cand = [j for j in cand if xyzs[j] is None
                        or np.min(np.hypot(F[:, 0] - xyzs[j][0], F[:, 1] - xyzs[j][1])) > self.ask_same_m]
        if self.ask_dmax > 0:
            cand = [j for j in cand if zs[j] is not None and zs[j] <= self.ask_dmax]
        if self.ask_near:                        # 가까운 순(같으면 순위 순)
            cand = sorted(cand, key=lambda j: zs[j] if zs[j] is not None else 1e9)
        if not cand:
            return None
        j = cand[0]
        if ids[j] is not None:
            self.fired.add(ids[j])
            self.fired_at[ids[j]] = [float(t), [float(v) for v in regs[j][1]]]   # 진단: 그 벽 자리에 언제 어떤 상자로 멈췄나
        xyz = None if xyzs[j] is None else [float(v) for v in xyzs[j]]   # 멈춘 뒤 다시 찾기용(이번 관측 한 번의 좌표)
        if xyz is not None:
            self.fired_xyz.append(xyz); self.fired_t.append(float(t))
        ev = dict(t=t, picks=[regs[j][1]], z=[zs[j]], xyz=xyz, cur_sim=float(regs[j][0]))
        if ids[j] is not None:                   # 그 벽 자리가 지금까지 몇 번·몇 장면·몇 초 동안 잡혔나(진단용)
            sp = self.spots[ids[j]]
            ts_ = sp.get("ts", [t])
            ev.update(nobs=len(sp["sims"]), nscene=len(set(ts_)), span=float(t - min(ts_)), msim=float(np.mean(sp["sims"])))
        return ev

    # ---- 5. 멈춘 뒤 목표 다시 찾기 ----
    def retarget(self, buf, t, dimg, pose, xyz, box0, z0, match_regions=False, vtol=0.3):
        """멈추기로 한 구간을 멈춘 뒤 화면에서 다시 찾는다. 판단부터 로봇이 설 때까지 화면이 움직이므로
        판단 순간의 (u, v)를 그대로 쓰면 어긋난다. 기준은 그 구간의 월드 좌표(xyz)다.
          기본: 월드 좌표를 지금 자세(오돔)와 지금 화면의 소실점으로 화면에 투영한다. 상자 크기는 거리 비율로.
                카메라 뒤면 '조준 후보 없음'. 화면 밖이면 in_view=False(팬틸트는 돌 수 있으나 콘솔 /aim 은 화면 픽셀만 받음).
          match_regions: 투영 전에 지금 화면의 벽 구간 중 월드 좌표가 평면 0.7 m·높이 vtol 안인 것을 먼저 찾는다.
        2026-09-30 4층 평가(scripts/eval_retarget.py, 번호판 관측 90건, 로봇이 안 서고 계속 걸은 녹화라 불리한 조건):
          겨누는 점↔번호판 각도 중앙 — 그대로 3.4°(1초 뒤)·5.2°(2초 뒤), 투영 0.9°·1.2°, 벽 구간 먼저 1.2°·1.4°,
          벽 구간을 높이 없이 맞추면 2.9°·3.7°(같은 벽 자리의 아래 구간이 잡힘). 그래서 기본은 투영만.
          투영만 쓰면 멈춘 뒤 후보 뽑기가 필요 없다."""
        xyz = np.asarray(xyz, np.float64)
        base = buf.nearest_tb(t)
        if match_regions:
            base, recs = self.accumulate(buf, t)
            regs, vp = self.wall_regions(base, recs, dimg)
            best, bd = None, P["radius"]
            for b in regs:
                w, z = M.region_world(b, dimg, vp, pose)
                if w is None or (vtol is not None and abs(w[2] - xyz[2]) > vtol):
                    continue
                d = float(np.linalg.norm(w[:2] - xyz[:2]))
                if d < bd:
                    best, bd = (b, z), d
            if best is not None:
                return dict(how="region", box=[float(v) for v in best[0]], z=best[1], d=bd, in_view=True)
        else:
            vp = vanishing_point(base["bgr"])
        p = project_world(xyz, vp, pose)
        if p is None:
            return dict(how="none", box=None, z=None, in_view=False)
        u, v, z = p
        k = (z0 / z if z0 else 1.0) / 2
        hw, hh = (box0[2] - box0[0]) * k, (box0[3] - box0[1]) * k
        return dict(how="proj", box=[u - hw, v - hh, u + hw, v + hh], z=z, in_view=bool(0 <= u < 640 and 0 <= v < 480))

    def facility_check(self, t, bgr, dimg, pose, allow_stop, sign_stopped, persist=2, gap=3.0, dedup_m=5.0,
                       zmax=5.0, follow_s=4.0):
        """시설확인(2026-10-01; 10-06부터 이름을 '관측'으로 — 계단은 글자를 읽을 필요가 없어 멈추지 않는다). 게이트 켜짐 + 조준 후보 0개가 persist 장면 이어지면, 직전 확인에서 gap 초·
        오돔 dedup_m 넘게 지났을 때 VLM 에 보이는 것을 열린 질문으로 묻는다. 답에 시설 단어가 없으면 다음 장면
        (1초 뒤)에 한 번 더 묻는다(화면 끝에 걸친 엘베를 한 번 묻기로 놓쳤다, 4층 105초).
        단어가 나오면 상자 질문으로 위치를 받아, 상자 깊이(중앙값)가 zmax(5 m) 안인 것만 이벤트로 낸다.
        나중에 거리·방향으로 쓰려면 깊이가 믿을 만한 거리여야 한다. 상자는 팬틸트 조준점이 된다.
        RealSense 화면에 보인 것으로만 낸다. 무엇인지는 멈춘 뒤 팬틸트 고화질이 확인한다."""
        if self.fac_vlm is None:
            return None
        if self.fac_pending and allow_stop and not sign_stopped:
            k, t_found = self.fac_pending                 # 멀리서 찾은 시설: 다가오는 동안 상자 거리만 다시 잰다
            if t - t_found > follow_s:
                self.fac_pending = None
            else:
                ev = self._fac_box(t, bgr, dimg, [k], "", zmax, dict(t=t, answer="(거리 다시 재기)", second=True))
                if ev:
                    self.fac_pending = None
                    return ev
                return None
        if not allow_stop or sign_stopped or self.last_like_n > 0:
            self.fac_run, self.fac_second = 0, False
            return None
        self.fac_run += 1
        p = np.array(pose[:2], float)
        second = self.fac_second
        if not second and (self.fac_run < persist or t - self.fac_last_t < gap or
                           (self.fac_last_p is not None and np.linalg.norm(p - self.fac_last_p) < dedup_m)):
            return None
        ans, sec = self.fac_vlm.ask(bgr)
        if not second:
            self.fac_last_t, self.fac_last_p = t, p
        low = ans.lower()
        hit = [k for k, ws in FACILITY_WORDS.items() if k in self.fac_kinds and any(w in low for w in ws)]
        rec = dict(t=t, answer=ans, vlm_sec=round(sec, 3), second=second)
        self.fac_second = not hit and not second          # 못 찾으면 다음 장면에 한 번 더
        ev = self._fac_box(t, bgr, dimg, hit, ans, zmax, rec)
        if hit and ev is None and any(b.get("z") for b in rec.get("boxes", [])):
            self.fac_pending = (hit[0], t)                # 찾았지만 5 m 밖: follow_s 초 동안 거리만 다시 잰다
        return ev

    def facility_check_v2(self, t, bgr, dimg, pose, allow_stop, sign_stopped, persist=2, gap=3.0, dedup_m=5.0,
                          zmax=5.0, follow_s=4.0):
        """시설확인 v2(2026-10-04). 공학관·소융대 계단 0건 진단: 계단이 보이고 게이트도 켜졌는데 (1) 번호판 후보가
        있어 질문을 아예 안 했거나 (2) 계단 직전에 묻고 '5 m 이동 전 다시 안 묻기'에 막혔다. 바꾼 점:
          - 의미지도(FacilityMap)가 top-K 중 하나라도 앞쪽에 시설을 예측하면 번호판 후보와 상관없이 gap 초마다 묻는다.
          - 예측이 없으면 예전처럼 번호판 후보 0개가 persist 장면 이어질 때만 묻는다.
          - '5 m 안 다시 안 묻기'는 시설을 찾은 자리에만 쓴다(못 찾은 질문 뒤에는 시간 간격만).
          - 열린 질문은 보내 놓고 다음 장면에서 답을 받는다(장면 처리를 기다리게 하지 않음). 상자 질문은 답에 시설이
            나왔을 때만, 물었던 그 화면으로 한다."""
        if self.fac_vlm is None:
            return None
        ev = None
        if self.fac_ask is not None:             # 지난 장면에 보낸 질문의 답
            ta, abgr, adimg, ap, mode, aexp = self.fac_ask
            self.fac_ask = None
            ans, sec = self.fac_vlm.wait()
            low = ans.lower()
            # 계단(fac_free_kinds)은 답만으로 받고, 엘베·화장실은 그 질문 때 지도가 앞쪽에 예측한 것만 받는다(2026-10-04.
            # 지도 없이 받던 4·5·6층 엘베는 4번 중 1번만 맞았다)
            # 지도에 그 종류가 아예 없으면(예: 공학관·소융대 의미지도에 엘베 없음) 지도 정보 없음을 '없다'는 증거로 쓰지 않는다
            # (2026-10-04 사용자 지적: G1 광장 엘베를 못 찾음). 그때는 계단처럼 답 + 5 m 깊이만으로 받는다.
            no_map = lambda k: self.fac_nomap_accept and (self.fac_map is None or not self.fac_map.fac.get(k))
            hit = [k for k, ws in FACILITY_WORDS.items() if k in self.fac_kinds and any(w in low for w in ws)
                   and (k in self.fac_free_kinds or k in aexp or no_map(k))]
            rec = dict(t=ta, answer=ans, vlm_sec=round(sec, 3), second=False, mode=mode, t_answer=t)
            ev = self._fac_box(ta, abgr, adimg, hit, ans, zmax, rec)
            if ev:
                self.fac_hit_p = ap
            elif hit and any(b.get("z") for b in rec.get("boxes", [])):
                self.fac_pending = (hit[0], t)
        if ev is not None:
            return ev
        if self.fac_pending and allow_stop and not sign_stopped:
            k, t_found = self.fac_pending
            if t - t_found > follow_s:
                self.fac_pending = None
            else:
                ev = self._fac_box(t, bgr, dimg, [k], "", zmax, dict(t=t, answer="(거리 다시 재기)", second=True))
                if ev:
                    self.fac_pending = None
                    self.fac_hit_p = np.array(pose[:2], float)
                return ev
        if not allow_stop or sign_stopped:
            self.fac_run = 0
            return None
        p = np.array(pose[:2], float)
        if self.fac_hit_p is not None and np.linalg.norm(p - self.fac_hit_p) < dedup_m:
            return None
        exp = self.fac_map.expect(t) if self.fac_map is not None else []
        if exp:
            mode = "지도:" + ",".join(exp)
        else:
            if self.last_like_n > 0:
                self.fac_run = 0
                return None
            self.fac_run += 1
            if self.fac_run < persist:
                return None
            mode = "후보없음"
        if t - self.fac_last_t < gap:
            return None
        self.fac_vlm.ask_async(bgr)
        self.fac_last_t = t
        self.fac_ask = (t, bgr, dimg, p, mode, exp)
        return None

    def _fac_box(self, t, bgr, dimg, kinds, ans, zmax, rec):
        """종류마다 상자 질문 → 상자 깊이 중앙값이 zmax 안인 가장 가까운 것으로 이벤트."""
        self.fac_log.append(rec)
        best = None
        for k in kinds:
            bx, _ = self.fac_vlm.boxes(bgr, FACILITY_BOX[k])
            for b in bx:
                x0, y0, x1, y1 = (int(round(v)) for v in b)
                v = dimg[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                v = v[v > 0]
                z = float(np.median(v)) / 1000.0 if len(v) > 20 else None
                rec.setdefault("boxes", []).append(dict(label=k, box=[x0, y0, x1, y1], z=z))
                if z is not None and z <= zmax and (best is None or z < best[2]):
                    best = (k, [float(x0), float(y0), float(x1), float(y1)], z)
        if best is None:
            return None
        return dict(t=t, kind="시설", label=best[0], answer=ans or rec.get("answer", ""), picks=[best[1]], z=[best[2]],
                    vlm_sec=rec.get("vlm_sec"))

    def update_floor(self, dimg):
        """바닥면 보정(인과). 정해지기 전에는 높이 띠를 적용하지 않는다."""
        if self.floor_band is None or self.floor_up is not None or dimg is None:
            return
        import surface_orient as SO
        P, _, _ = SO.backproject(dimg)
        r = SO.floor_plane(P)
        if r:
            self.floor_fits.append(r)
        if len(self.floor_fits) >= self.floor_need:
            u = np.median(np.array([f[0] for f in self.floor_fits]), 0)
            self.floor_up = u / np.linalg.norm(u)
            self.floor_camh = float(np.median([f[1] for f in self.floor_fits]))
            print(f"[바닥면 고정] 위 {np.round(self.floor_up, 3)} 카메라 높이 {self.floor_camh:.2f} m "
                  f"(찾은 장면 {len(self.floor_fits)})", file=sys.stderr, flush=True)

    def box_floor_height(self, dimg, b):
        """상자 깊이 점의 바닥 위 높이 중앙(m). 깊이 없으면 None."""
        import surface_orient as SO
        x0, y0, x1, y1 = (int(v) for v in b[:4])
        z = dimg[max(0, y0):max(0, y1):2, max(0, x0):max(0, x1):2].astype(np.float32) / 1000.0
        if z.size == 0:
            return None
        vs, us = np.mgrid[max(0, y0):max(0, y1):2, max(0, x0):max(0, x1):2]
        ok = z > 0.2
        if ok.sum() < 5:
            return None
        P = np.stack([(us[ok] - SO.CX) / SO.FX * z[ok], (vs[ok] - SO.CY) / SO.FY * z[ok], z[ok]], 1)
        return float(np.median(P @ self.floor_up) + self.floor_camh)

    def scene(self, buf, t, dimg, pose, allow_stop=True):
        tm = {}
        t0 = time.perf_counter()
        self.update_floor(dimg)                  # 실시간 모드용(replay 는 장면 루프에서 먼저 부름; 정해지면 바로 돌아옴)
        base, recs = self.accumulate_sparse(buf, t) if self.sparse else self.accumulate(buf, t)
        tm["후보+누적"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        regs, vp = self.wall_regions(base, recs, dimg)
        tm["벽 구간"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        regs, tm["SigLIP(작업자)"] = self.rerank(base["bgr"], regs)
        tm["SigLIP 왕복"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        ev, thr = self.decide(buf, t, regs, dimg, vp, pose, allow_stop)
        tm["기억+판단"] = time.perf_counter() - t3
        tm["합계"] = time.perf_counter() - t0
        return ev, regs, thr, tm, base


def build_events(args):
    """replay 와 live 가 같이 쓰는 설정(2026-10-05 실시간도 녹화 실험과 같은 알고리즘이 되도록 replay 에서 떼어 냄).
    값과 순서는 replay 에 있던 그대로다. 녹화 깊이(--far-2x-m)와 지도 예측용 위치추정 기록은 부르는 쪽이 붙인다."""
    os.environ["KCI_SIGN_PROMPTS"] = args.sign_prompts   # SigLIP 작업자가 물려받는다
    search_cfg = None
    if getattr(args, "search_config", ""):                 # 2026-10-05: 층별 '찾을 대상' 설정(make_search_config.py)
        search_cfg = json.load(open(args.search_config))
    elif getattr(args, "fac_map_sem", ""):                 # 2026-10-06: 설정 파일이 없으면 의미지도를 직접 읽어 같은 규칙으로 정한다
        from make_search_config import build
        search_cfg = build(json.load(open(args.fac_map_sem)), Path(args.fac_map_sem).stem)
    if search_cfg is not None:
        print(f"[찾을 대상] 의미지도 종류 {search_cfg['map_kinds']} → 작은 표지 "
              f"{[p for x in search_cfg['small_sign'] for p in x['phrases']]}, 장면 {[x['name'] for x in search_cfg['scene_object']]}",
              file=sys.stderr, flush=True)
        if not search_cfg["small_sign"]:                   # 지도에 번호 있는 호실이 없으면 표준 문구(빈 문구로는 SigLIP 이 못 돈다)
            search_cfg["small_sign"] = [dict(map_kind="(기본)", phrases=["a small room name plate", "a small room name plate near a door",
                                                                       "a hallway sign perpendicular to the wall"])]
        os.environ["KCI_SIGN_PROMPTS_JSON"] = json.dumps([p for x in search_cfg["small_sign"] for p in x["phrases"]])
        for x in search_cfg["scene_object"]:
            FACILITY_WORDS[x["name"]] = tuple(x["answer_words"])
            FACILITY_BOX[x["name"]] = x["box_phrase"]
    vlm = SiglipClient()
    pool = Pool(args.workers) if args.workers > 1 else None
    L = LiveEvents(vlm, pool, sparse=args.sparse)
    L.n_hist, L.rate = args.sparse_hist, args.sparse_rate
    L.rule, L.ask_dmax, L.ask_near = args.rule, args.ask_dmax, args.ask_near
    L.sign_conf = args.sign_confusers
    L.sign_max_w = args.sign_max_width
    L.sign_max_group_w = args.sign_max_group_width
    L.sign_max_h, L.sign_edge = args.sign_max_height, args.sign_edge_px
    L.board_max = args.board_max
    L.fac_kinds = tuple(args.facility_kinds.split(","))
    if search_cfg is not None:
        L.fac_kinds = tuple(x["name"] for x in search_cfg["scene_object"])
    L.ceil_filter = args.ceiling_filter
    L.cam_pitch, L.ceil_min_elev = args.cam_pitch_deg, args.ceil_min_elev
    vlm.want_board = args.board_max != 0 or args.ceiling_filter
    L.log_groups = getattr(args, "save_regions", False)
    if args.facility:
        L.fac_vlm = VlmClient()
    L.fac_v2 = args.facility_v2
    L.far_m = args.far_2x_m
    L.scales = tuple(int(v) for v in args.prop_scales.split(","))
    L.ask_same_m = args.ask_same_m
    L.ask_same_sec = args.ask_same_sec
    L.ask_min_scenes = args.ask_min_scenes
    L.floor_band = tuple(float(v) for v in args.floor_band.split(",")) if args.floor_band else None
    L.fac_free_kinds = tuple(k for k in args.fac_free_kinds.split(",") if k)
    if search_cfg is not None:
        L.fac_free_kinds = tuple(x["name"] for x in search_cfg["scene_object"] if not x["need_map"])
    L.fac_nomap_accept = args.fac_nomap_accept
    L.sig_w = args.sig_weight
    L.cand_tile = args.cand_tile
    L.spot_recent = args.spot_recent
    L.sim_floor = args.sim_floor
    return L, vlm, pool


def replay(args):
    root = Path(args.root)
    rows = list(csv.DictReader(open(root / "index.csv")))
    scenes = [float(r[0]) for r in csv.reader(open(args.frames_csv))]
    dz = np.load(args.depth)
    odom = M.Odom(args.odom, args.t0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf = Frames()
    L, vlm, pool = build_events(args)
    if args.far_2x_m:
        dkeys = {round(float(k), 1) for k in dz.files}

        def depth_of(tb):                        # 녹화 깊이 기록은 장면 시각(1초 칸)에만 있어 가장 가까운 것
            for k in (round(tb), round(tb) - 1, round(tb) + 1):
                if float(k) in dkeys:
                    return dz[f"{float(k):.1f}"]
            return None
        L.depth_of = depth_of
    if args.fac_map_topk:
        L.fac_map = FacilityMap(args.fac_map_topk, args.fac_map_sem, args.fac_map_ppm, L.fac_kinds, base_png=args.fac_map_base)
    fac_events = []
    last_done = -1e9
    gate = None
    if args.gate:                                # 위치추정 모호함·튐일 때만 계산(확실하면 아예 안 함)
        # 튐: 1위가 오돔보다 3 m 넘게 더 움직임. 그런데 비슷한 후보가 모두 5 m 안(spread < 5)이면 같은 봉우리 안의
        # 흔들림이다(모호함 정의가 '5 m 밖 후보'와 같은 기준). 2026-09-30 4층: 튐만으로 켜진 11초가 전부 퍼짐 ≤ 6.4 m,
        # 여기서 난 멈춤 4번(65·83·100·108초)은 전부 쓸모없었다.
        # 계산 여부(gate)는 예전 그대로 두고 멈춤 허용(stop_ok)만 막는다. 계산까지 끄면 계산 장면이 바뀌어
        # 기억·추적이 달라지고 뒤쪽 복도 멈춤까지 흔들렸다(번호판 10 → 7). --jump-min-spread 0 이면 예전과 같다.
        rows_g = json.load(open(args.gate))
        gate = {float(x["t"]): bool(x["amb"] or x["jump"]) for x in rows_g}
        if args.stop_rule == "amb_sensor_ok":    # 2026-10-07 지침서 E4: 장소 애매 + 센서 정상일 때만(튐·오돔 이상만으로는 부르지 않음)
            stop_ok = {float(x["t"]): bool(x["amb"] and not x.get("sensor_bad", False)) for x in rows_g}
        else:
            stop_ok = {float(x["t"]): bool(x["amb"] or (x["jump"] and float(x.get("spread", 1e9)) >= args.jump_min_spread))
                       for x in rows_g}
    skipped = 0

    reg_log = open(out / "regions_per_scene.jsonl", "w") if args.save_regions else None

    def will_compute(t):
        """이 장면을 계산할지(게이트·간격 규칙). 건너뛸 장면은 후보를 미리 뽑지 않는다."""
        if gate is None or gate.get(t, False):
            return True
        gm = args.gate_mode
        return gm == "stoponly" or (gm.startswith("every") and t - last_done >= float(gm[5:]) - 1e-6)
    events, timing = [], []
    si = 0
    t_wall = time.time()
    t_start_rel = float(rows[0]["bag_relative_sec"])
    lags = []
    hcol = "header_relative_sec" if "header_relative_sec" in rows[0] else "bag_relative_sec"   # 3층 구간은 bag 시각뿐
    for r in rows:
        if args.max_t and float(r["bag_relative_sec"]) > args.max_t + 1.0:
            break
        tb, th = float(r["bag_relative_sec"]), float(r[hcol])
        buf.add(int(r["frame_index"]), tb, th, cv2.imread(str(root / r["filename"])))
        if args.realtime:                        # 녹화 속도 그대로 흘려 넣기(실제 로봇과 같은 조건)
            lag = (tb - t_start_rel) - (time.time() - t_wall)
            if lag > 0:
                time.sleep(lag)
        L.prefetch(buf, scenes[si] if si < len(scenes) and will_compute(scenes[si]) else None)
        # 장면 t는 그 뒤 프레임이 들어온 다음에 처리한다(가장 가까운 프레임이 정해진다)
        while si < len(scenes) and tb > scenes[si] and th > scenes[si]:
            t = scenes[si]
            si += 1
            key = f"{t:.1f}"
            if key not in dz:                    # 기존 regions 단계도 깊이 없는 초는 건너뛰었다
                continue
            if args.max_t and t > args.max_t:
                break
            on = True if gate is None else stop_ok.get(t, False)
            L.update_floor(dz[key])              # 바닥면 보정은 게이트와 상관없이(장면당 13 ms, 정해지면 안 함)
            if not will_compute(t):
                skipped += 1
                continue
            if L.prev is not None and t - L.prev["t"] > 3.5:
                L.prev = None                    # 오래 건너뛰면 트랙은 끊는다(켜질 때 새로 시작)
            last_done = t
            ev, regs, thr, tm, base = L.scene(buf, t, dz[key], odom(t), allow_stop=on)
            timing.append(tm)
            if args.save_regions:                # 진단용: 장면마다 SigLIP 재정렬 뒤 구간(+ 잘리기 전 원래 묶음 상자)
                grp = [L.group_of.get(tuple(float(v) for v in b)) for _, b in regs]
                reg_log.write(json.dumps(dict(t=t, thr=thr, regs=regs, groups=grp, allow=on, **getattr(L, "last_why", {}))) + "\n")
            if args.realtime:                    # 장면 시각 대비 판단이 끝난 시각(실제 지연)
                lags.append((time.time() - t_wall) - (t - t_start_rel))
            if args.facility:
                fc = L.facility_check_v2 if L.fac_v2 else L.facility_check
                fe = fc(t, base["bgr"], dz[key], odom(t), on, ev is not None)
                if fe:
                    fac_events.append(fe)
                    print(f"{t:6.1f}s 시설 관측(멈추지 않음) #{len(fac_events)}  {fe['label']} {fe['z'][0]:.1f} m  "
                          f"(VLM: {fe['answer'][:60]})", flush=True)
            if ev:
                ev["k"] = f"{args.name}_{t}"
                events.append(ev)
                print(f"{t:6.1f}s 멈춤 #{len(events)}  확인 {len(ev['picks'])}곳  (기준 {thr:.4f}, 이 장면 {tm['합계']*1000:.0f} ms)",
                      flush=True)
    json.dump(events, open(out / "events_live_replay.json", "w"), ensure_ascii=False)
    if args.facility:
        json.dump(fac_events, open(out / "facility_events.json", "w"), ensure_ascii=False, indent=1)
        json.dump(L.fac_log, open(out / "facility_asks.json", "w"), ensure_ascii=False, indent=1)
        print(f"시설확인: 질문 {len(L.fac_log)}번, 시설 관측 {len(fac_events)}번 {[int(e['t']) for e in fac_events]}")
        L.fac_vlm.close()
    vlm.close()
    if pool:
        pool.close()
    keys = list(timing[0])
    tsum = {k: dict(med_ms=round(1000 * float(np.median([x[k] for x in timing])), 1),
                    p95_ms=round(1000 * float(np.percentile([x[k] for x in timing], 95)), 1)) for k in keys}
    json.dump(dict(scenes=len(timing), wall_sec=round(time.time() - t_wall, 1), timing=tsum),
              open(out / "timing.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n장면 {len(timing)}개(건너뜀 {skipped}개), 후보 뽑은 프레임 {L.n_prop}장, 멈춤 {len(events)}번, "
          f"걸린 시간 {time.time() - t_wall:.0f}초")
    json.dump(dict(scenes=len(timing), skipped=skipped, n_prop=L.n_prop, events=len(events)),
              open(out / "summary.json", "w"), ensure_ascii=False)
    if lags:
        print(f"  장면 시각 → 판단 끝까지 실제 지연: 중앙 {1000*np.median(lags):.0f} ms, 95% {1000*np.percentile(lags, 95):.0f} ms, "
              f"최대 {1000*max(lags):.0f} ms")
        json.dump([round(x, 3) for x in lags], open(out / "lags_sec.json", "w"))
    for k, v in tsum.items():
        print(f"  {k:12s} 중앙 {v['med_ms']:7.1f} ms   95% {v['p95_ms']:7.1f} ms")
    if args.reference:
        ref = json.load(open(args.reference))
        compare(ref, events)


class Go2Source:
    """조준 콘솔(/live.mjpg)에서 컬러 프레임과 촬영 시각을 받아 온다. 받는 스레드가 따로 돈다."""

    def __init__(self, base):
        import threading
        self.base = base.rstrip("/")
        self.q = []
        self.lock = threading.Lock()
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import urllib.request
        r = urllib.request.urlopen(self.base + "/live.mjpg", timeout=10)
        while self.alive:
            line = r.readline()
            if not line:
                break
            if not line.startswith(b"--f"):
                continue
            hdr = {}
            while True:
                h = r.readline().strip()
                if not h:
                    break
                k, _, v = h.decode().partition(":")
                hdr[k.strip().lower()] = v.strip()
            jpg = r.read(int(hdr["content-length"]))
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            with self.lock:
                self.q.append((float(hdr["x-stamp"]), int(hdr["x-seq"]), img))
        self.alive = False

    def pop_all(self):
        with self.lock:
            out, self.q = self.q, []
        return out

    def depth(self, stamp=None):
        """최신 깊이(mm)와 촬영 시각. stamp 는 흉내 콘솔만 쓴다(그 시각 깊이). 실제 콘솔은 늘 최신 것을 준다."""
        import urllib.request
        try:
            q = f"?stamp={stamp:.6f}" if stamp is not None else ""
            r = urllib.request.urlopen(self.base + "/live/depth.png" + q, timeout=3)
            st = float(r.headers["X-Stamp"])
            return cv2.imdecode(np.frombuffer(r.read(), np.uint8), cv2.IMREAD_UNCHANGED), st
        except Exception:
            return None, None

    def odom(self, stamp=None):
        import urllib.request
        try:
            q = f"?stamp={stamp:.6f}" if stamp is not None else ""
            d = json.loads(urllib.request.urlopen(self.base + "/live/odom" + q, timeout=3).read())
            return (d["x"], d["y"], d["yaw"]), d["stamp"]
        except Exception:
            return None, None

    def aim(self, u, v, z=None):
        """z(m)를 주면 콘솔이 그 거리로 겨눈다(화면 밖 픽셀도 됨). 그때는 가동 범위 밖이면 콘솔이 거절한다."""
        import urllib.request
        q = f"u={int(round(u))}&v={int(round(v))}&focus=0" + (f"&z={z:.3f}&strict=1" if z else "")
        return urllib.request.urlopen(f"{self.base}/aim?{q}", timeout=5).read().decode()


class LocFeed:
    """위치추정 1초 기록을 받아 게이트와 지도 예측에 넘긴다(2026-10-05).
    기록 한 줄 = {"stamp": 로봇 시계(초), "level": "AMBIGUOUS"|"CONFIDENT", "margin": 점수 차, "ppm": 그때 추정 축척,
                  "odom": [x, y](m), "odom_yaw": yaw(rad, 선택, 센서 이상 판정용), "sensor_bad": bool(선택, OdomGuard 가 그 초에 오돔 튐을 잡았으면 true), "cands": [[score, x_px, y_px, yaw_rad], ...]} — loc_topk_series.py 의 timeline/topk 한 초와 같은 내용.
    url 의 GET ?after=<stamp> 가 그 뒤 기록들의 JSON 목록을 돌려준다(흉내 콘솔 /live/loc). 게이트 식은 make_gate_json.gate_row 그대로."""

    def __init__(self, url, origin):
        import threading
        from make_gate_json import gate_row
        self.url, self.origin, self.gate_row = url, origin, gate_row
        self.lock = threading.Lock()
        self.rows, self.ts, self.prev, self.last_stamp = [], [], None, -1e18
        self.new = []                            # 지도 예측(FacilityMap)에 아직 안 넘긴 기록
        self.alive, self.err = True, None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import urllib.request
        while self.alive:
            try:
                got = json.loads(urllib.request.urlopen(f"{self.url}?after={self.last_stamp:.6f}", timeout=3).read())
                self.err = None
            except Exception as e:               # 끊겨도 계속 시도. 게이트는 오래된 기록을 '모름'으로 다룬다
                self.err, got = str(e), []
            for d in got:
                if d["stamp"] <= self.last_stamp:
                    continue
                t = d["stamp"] - self.origin
                row, self.prev = self.gate_row(t, d["level"], d["margin"], [c[:3] for c in d["cands"]],
                                               d["odom"], d["ppm"], self.prev, odom_yaw=d.get("odom_yaw"))
                if d.get("sensor_bad"):                  # 2026-10-08: 위치추정 쪽 sensor_check.OdomGuard 가 튐을 잡아 알려 주면 그대로 쓴다
                    row["sensor_bad"] = True
                with self.lock:
                    self.rows.append(row)
                    self.ts.append(t)
                    self.new.append((t, d["cands"], d["ppm"]))
                self.last_stamp = d["stamp"]
            time.sleep(0.2)

    def at(self, t, stale=3.0):
        """장면 t 의 게이트 줄(replay 는 같은 초의 줄을 썼다: t+0.5 안의 가장 늦은 기록). 없거나 stale 초보다 오래되면 None."""
        with self.lock:
            i = bisect.bisect_right(self.ts, t + 0.5) - 1
            if i < 0 or t - self.ts[i] > stale:
                return None
            return self.rows[i]

    def pop_new(self):
        with self.lock:
            out, self.new = self.new, []
        return out


def live(args):
    """Go2 조준 콘솔에서 실시간으로 받아 멈춤 이벤트를 낸다. 알고리즘·설정은 replay 와 같다(build_events, add_algo_args).
    2026-10-05: 새 구조(--search-config 문구·계단 VLM)와 게이트(--loc-url 위치추정 기록 → 애매·튐일 때만 멈춤, --gate-mode)를
    replay 와 같은 순서로 넣었다. --loc-url 이 없으면 예전처럼 게이트 없이 늘 계산·멈춤 허용.
    위치추정 기록이 끊기거나 3초보다 오래되면 그 장면은 '애매'로 본다(멈춰서 확인하는 쪽이 안전).
    로봇 위치는 Go2 오돔(x, y, yaw)을 그대로 쓴다(RealSense가 로봇 정면을 봄). 오돔이 없으면 그 장면은 건너뛴다."""
    out = Path(args.out)
    (out / "events").mkdir(parents=True, exist_ok=True)
    src = Go2Source(args.go2)
    L, vlm, pool = build_events(args)
    if args.far_2x_m:                            # 실시간은 가장 최근 깊이로(녹화는 장면 시각의 깊이)
        last_depth = {}
        L.depth_of = lambda tb: last_depth.get("d")
    buf = Frames()
    t_first, origin, next_scene, n_ev = None, None, None, 0
    loc = None
    pending = []                                 # 멈춤 뒤 다시 찾을 목표: (다시 찾을 시각, 기록)
    log = open(out / "events.jsonl", "a", encoding="utf-8")
    flog = open(out / "facility_events.jsonl", "a", encoding="utf-8")
    events, fac_events, timing = [], [], []
    last_done, skipped, n_stale = -1e9, 0, 0

    def gate_now(t):
        """(계산할지, 멈춤 허용) — replay 의 will_compute / stop_ok 와 같은 식."""
        if loc is None:
            return True, True
        g = loc.at(t)
        if g is None and args.loc_missing == "amb":   # 위치추정 기록 없음(끊김·시작 전) → 애매로
            return True, True
        # quiet: replay 와 같이 기록 없는 초는 '확실'(멈춤 없음, 간격 규칙으로만 계산). 녹화 대조용
        on_c = g is not None and (g["amb"] or g["jump"])
        if args.stop_rule == "amb_sensor_ok":
            on_s = g is not None and g["amb"] and not g.get("sensor_bad", False)
        else:
            on_s = g is not None and (g["amb"] or (g["jump"] and g["spread"] >= args.jump_min_spread))
        gm = args.gate_mode
        comp = on_c or gm == "stoponly" or (gm.startswith("every") and t - last_done >= float(gm[5:]) - 1e-6)
        return comp, on_s
    print("받는 중… (Ctrl+C로 끝)", flush=True)
    t_wall = time.time()
    try:
        while src.alive:
            got = src.pop_all()
            if not got:
                time.sleep(0.01)
                continue
            for st, seq, img in got:
                if t_first is None:
                    t_first = st
                    origin = args.t_origin if args.t_origin else st
                    next_scene = (st - origin) + args.warmup
                    if args.t_origin:            # 시각 원점을 주면 장면을 그 원점의 정수 초에 맞춘다(녹화 대조용)
                        next_scene = float(np.ceil(next_scene))
                    if args.loc_url:
                        loc = LocFeed(args.loc_url, origin)
                    if args.fac_map_sem and args.facility:
                        L.fac_map = FacilityMap("", args.fac_map_sem, args.fac_map_ppm, L.fac_kinds, base_png=args.fac_map_base)
                rel = st - origin
                buf.add(seq, rel, rel, img)
                if loc is not None and L.fac_map is not None:
                    for t_l, cands, ppm_l in loc.pop_new():
                        L.fac_map.add(t_l, cands, ppm_l)
                while pending and rel > pending[0][0]:   # 멈춘 뒤: 그 사이 움직임을 반영해 다시 찾고 조준
                    t_re, rec = pending.pop(0)
                    pose2, _ = src.odom(origin + t_re)
                    if pose2 is None:
                        print(f"{t_re:6.1f}s 다시 찾기: 오돔이 없어 건너뜀", flush=True)
                        continue
                    r = L.retarget(buf, t_re, None, pose2, rec["xyz"], rec["picks"][0], rec["z"][0])
                    rec["retarget"] = dict(t=t_re, pose=list(pose2), **r)
                    msg = f"{t_re:6.1f}s 다시 찾기: "
                    if r["box"] is None:
                        msg += "카메라 뒤 — 조준 안 함"
                    else:                               # 화면 밖이어도 팬틸트는 더 넓게 돈다. 범위 밖이면 콘솔이 거절
                        b = r["box"]
                        u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                        msg += (f"(u, v) = ({u:.0f}, {v:.0f}), 거리 {r['z']:.1f} m"
                                + ("" if r["in_view"] else " (RealSense 화면 밖)"))
                        if args.aim:
                            rec["aim"] = src.aim(u, v, r["z"])
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log.flush()
                    print(msg, flush=True)
                L.prefetch(buf, next_scene if gate_now(next_scene)[0] else None)
                if rel <= next_scene:
                    continue
                t = next_scene
                next_scene += 1.0
                dimg, dst = src.depth(origin + t)
                pose, ost = src.odom(origin + t)
                if dimg is None:
                    print(f"{t:6.1f}s 깊이가 없어 건너뜀", flush=True)
                    continue
                if args.far_2x_m:
                    last_depth["d"] = dimg
                if pose is None:
                    print(f"{t:6.1f}s 오돔이 없어 건너뜀(--odom-topic 확인)", flush=True)
                    continue
                if loc is not None and loc.at(t) is None:
                    n_stale += 1
                    if n_stale in (1, 10) or n_stale % 60 == 0:
                        print(f"{t:6.1f}s 위치추정 기록 없음({n_stale}번째, {loc.err or '늦음'}) → "
                              + ("애매로 보고 계속" if args.loc_missing == "amb" else "멈춤 없이 계속"), flush=True)
                comp, on = gate_now(t)
                L.update_floor(dimg)             # replay 와 같이 게이트와 상관없이(정해지면 바로 돌아옴)
                if not comp:
                    skipped += 1
                    continue
                if L.prev is not None and t - L.prev["t"] > 3.5:
                    L.prev = None                # 오래 건너뛰면 트랙은 끊는다(replay 와 같음)
                last_done = t
                ev, regs, thr, tm, base = L.scene(buf, t, dimg, pose, allow_stop=on)
                timing.append(tm)
                lag = time.time() - (t_first + t) if args.clock_synced else None
                line = (f"{t:6.1f}s 구간 {len(regs)}개, 기준 {thr:.4f}, 처리 {tm['합계'] * 1000:.0f} ms, "
                        f"깊이 시차 {1000 * (dst - (origin + t)):+.0f} ms" + ("" if on else ", 위치 확실(멈춤 안 함)"))
                if args.facility:
                    fc = L.facility_check_v2 if L.fac_v2 else L.facility_check
                    fe = fc(t, base["bgr"], dimg, pose, on, ev is not None)
                    if fe:
                        fac_events.append(fe)
                        im = base["bgr"].copy()
                        x0, y0, x1, y1 = (int(round(v)) for v in fe["picks"][0])
                        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 220, 0), 3)
                        path = out / "events" / f"facility_{len(fac_events):03d}_{t:07.1f}s.jpg"
                        cv2.imwrite(str(path), im)
                        flog.write(json.dumps(dict(fe, stamp=origin + t, pose=list(pose), image=str(path)), ensure_ascii=False) + "\n")
                        flog.flush()
                        line += f"  ◆ 시설 관측(멈추지 않음) #{len(fac_events)} {fe['label']} {fe['z'][0]:.1f} m"
                if ev:
                    n_ev += 1
                    events.append(ev)
                    im = base["bgr"].copy()
                    for j, b in enumerate(ev["picks"]):
                        x0, y0, x1, y1 = (int(round(v)) for v in b)
                        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 190, 255), 3 if j == 0 else 2)
                        cv2.putText(im, str(j + 1), (x0 + 3, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2)
                    path = out / "events" / f"event_{n_ev:03d}_{t:07.1f}s.jpg"
                    cv2.imwrite(str(path), im)
                    rec = dict(t=t, stamp=origin + t, picks=ev["picks"], z=ev["z"], xyz=ev.get("xyz"),
                               pose=list(pose), image=str(path))
                    if ev.get("xyz") is not None:       # 판단 순간 (u, v)는 로봇이 서는 동안 어긋나므로 조금 뒤 다시 찾는다
                        pending.append((t + args.retarget_delay, rec))
                    else:                               # event 규칙(좌표 없음): 예전처럼 바로 조준
                        if args.aim:
                            b = ev["picks"][0]
                            rec["aim"] = src.aim((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        log.flush()
                    line += f"  ★ 멈춤 #{n_ev} 확인 {len(ev['picks'])}곳 (거리 {', '.join(f'{z:.1f}' for z in ev['z'])} m)"
                print(line, flush=True)
    except KeyboardInterrupt:
        pass
    src.alive = False
    if loc is not None:
        loc.alive = False
    # replay 와 같은 이름으로도 남긴다(점검·채점 스크립트가 그대로 읽게)
    json.dump(events, open(out / "events_live_replay.json", "w"), ensure_ascii=False)
    if args.facility:
        json.dump(fac_events, open(out / "facility_events.json", "w"), ensure_ascii=False, indent=1)
        json.dump(L.fac_log, open(out / "facility_asks.json", "w"), ensure_ascii=False, indent=1)
        L.fac_vlm.close()
    vlm.close()
    if pool:
        pool.close()
    json.dump(dict(scenes=len(timing), skipped=skipped, n_prop=L.n_prop, events=n_ev, facility=len(fac_events),
                   loc_missing=n_stale, wall_sec=round(time.time() - t_wall, 1)),
              open(out / "summary.json", "w"), ensure_ascii=False)
    print(f"끝: 장면 {len(timing)}개(건너뜀 {skipped}개), 멈춤 {n_ev}번, 시설 멈춤 {len(fac_events)}번, "
          f"위치추정 기록 없던 장면 {n_stale}개, 기록 {out / 'events.jsonl'}")


def compare(ref, got):
    rt, gt = [e["t"] for e in ref], [e["t"] for e in got]
    same_t = rt == gt
    same_p = same_t and all(len(a["picks"]) == len(b["picks"]) and
                            all(np.allclose(x, y, atol=1e-6) for x, y in zip(a["picks"], b["picks"]))
                            for a, b in zip(ref, got))
    print(f"\n기존 결과와 대조: 멈춤 {len(ref)}번 대 {len(got)}번, 시각 {'일치' if same_t else '불일치'}, "
          f"확인 구간 {'일치' if same_p else '불일치'}")
    if not same_t:
        print("  기존에만:", sorted(set(rt) - set(gt)))
        print("  실시간에만:", sorted(set(gt) - set(rt)))


def verify(args):
    """빠른 버전이 기존 함수와 비트 단위로 같은지 앞부분 장면에서 확인한다."""
    root = Path(args.root)
    rows = list(csv.DictReader(open(root / "index.csv")))
    scenes = [float(r[0]) for r in csv.reader(open(args.frames_csv))]
    buf = Frames()
    fast, slow = LiveEvents(None, slow=False), LiveEvents(None, slow=True)
    slow_prop = {}
    n_prop = n_scene = 0
    t_fast = t_slow = 0.0
    si = 0
    for r in rows:
        tb, th = float(r["bag_relative_sec"]), float(r["header_relative_sec"])
        if tb > args.max_t + 1.0:
            break
        if tb < args.t_from - 3.0:               # 앞부분은 읽지 않는다(대조 구간만)
            continue
        buf.add(int(r["frame_index"]), tb, th, cv2.imread(str(root / r["filename"])))
        while si < len(scenes) and tb > scenes[si] and th > scenes[si]:
            t = scenes[si]
            si += 1
            if t < args.t_from:
                continue
            if t > args.max_t:
                break
            a0 = time.perf_counter()
            _, rf = fast.accumulate(buf, t)
            t_fast += time.perf_counter() - a0
            # 기존: 기존 propose + 기존 반복
            for f in buf.items:
                if f["id"] in fast.props and f["id"] not in slow_prop:
                    b0 = time.perf_counter()
                    slow_prop[f["id"]] = small_proposals.propose(f["bgr"])
                    t_slow += time.perf_counter() - b0
                    assert slow_prop[f["id"]] == fast.props[f["id"]], f"후보 다름: 프레임 {f['id']}"
                    n_prop += 1
            slow.props = dict(slow_prop)
            b0 = time.perf_counter()
            _, rs = slow.accumulate(buf, t)
            t_slow += time.perf_counter() - b0
            assert rf == rs, f"누적 결과 다름: {t}s"
            n_scene += 1
    print(f"확인 끝: 후보 {n_prop}장·누적 {n_scene}장면 전부 비트 단위로 같음. "
          f"걸린 시간 빠른 버전 {t_fast:.1f}초 / 기존 {t_slow:.1f}초")


def add_algo_args(p):
    """알고리즘 설정 인자. replay 와 live 가 같은 것을 받는다(2026-10-05, 실시간도 녹화 실험과 같은 설정으로)."""
    p.add_argument("--sim-floor", type=float, default=0.078,
                   help="상대 기준에 더해 이번 장면 SigLIP 점수가 이 값 이상이어야 조준 후보(0 = 안 봄). 2026-10-06 기본 0.078 = 판정 표지판 하위 5%%"
                        "(기본 문구 셋 기준). 4F·7F·G1F·G2F: G2F 천장·배관 멈춤 11 → 4, 7F+G1F 표지판 비율 63 → 62%%, 4층 번호판 5 → 4(56_점수바닥값)")
    # 2026-10-09 기본값 변경(배포판 C): --spot-recent 3, --ask-same-m 2.0, --ask-same-sec 60. 10-08 배포판(A)은 0, 0, 0
    p.add_argument("--spot-recent", type=int, default=3,
                   help="벽 자리 점수를 그 자리 최근 N번 관측 평균으로(0 = 전체 평균, 기존). 2026-10-05 4층 시험")
    p.add_argument("--cand-tile", action="store_true",
                   help="조준 후보를 벽 구간 대신 그 안에서 SigLIP 이 가장 높던 조각(가로·세로 100 px 이하)으로. 2026-10-05 4층 시험")
    p.add_argument("--sig-weight", type=float, default=1.0,
                   help="최종 순위 = 벽 구간 순위 + 이 값 × SigLIP 순위. 1 = 기존(반반), 크게 주면 SigLIP 순서(같으면 벽 구간 순)")
    p.add_argument("--workers", type=int, default=6, help="작은 후보를 병렬로 뽑을 프로세스 수")
    # 2026-10-05: 기본값을 지금 쓰는 설정(13개 층 42_새구조_최종문구와 같음)으로 바꿨다. 인자 없이 돌려도 그 설정이다.
    # 바뀐 기본: --gate-mode every3, --sparse, --sparse-rate 2, --facility, --facility-v2, --sign-prompts roomstd.
    # 끄려면 --no-sparse, --no-facility, --no-facility-v2. 10-05 이전 실행 스크립트를 그대로 다시 돌리면 결과가 달라질 수 있다.
    p.add_argument("--gate-mode", default="every3", choices=["skip", "stoponly", "every2", "every3"],
                   help="확실할 때 계산: skip=안 함, stoponly=매초(멈춤만 막음), everyN=N초마다. 기본 every3")
    p.add_argument("--sign-confusers", action="store_true",
                   help="조준 후보에서 헷갈리는 것(포스터·게시판·조명·창문·문·유리·소화기함) 문구가 이기는 구간을 뺀다")
    p.add_argument("--sign-max-width", type=float, default=0.85,
                   help="조준 후보 실제 폭 상한(m). 4층 43번 분류: 0.85면 게시판 12→1, 번호판 11→9. 0이면 안 봄")
    p.add_argument("--sign-max-height", type=float, default=0.0,
                   help="조준 후보 실제 높이 상한(m). 2026-10-03 8개 층 분석: 0.8이면 문·문틀·게시판 위주로 76/264 빠짐. 0이면 안 봄")
    p.add_argument("--sign-edge-px", type=int, default=0,
                   help="화면 가장자리에서 이 픽셀 안에 닿은(잘린) 구간은 조준 후보에서 뺀다. 0이면 안 봄")
    p.add_argument("--sign-max-group-width", type=float, default=0.0,
                   help="조준 후보의 잘리기 전 원래 묶음 실제 폭 상한(m). 큰 게시판의 한 조각을 거른다. 0이면 안 봄")
    p.add_argument("--board-max", type=float, default=0.02,
                   help="조준 후보 주변을 넓게 잘라 본 '게시판 − 작은 표지' 점수 상한. 3개 층 75번 분류에서 0.02면 "
                        "번호판·표지 0개 잃고 게시판 16→8. 0이면 안 봄")
    p.add_argument("--cam-pitch-deg", type=float, default=11.0,
                   help="RealSense 가 위로 기운 각도(도). 2026-10-04 녹화들의 소실점 중앙값 y≈359 로 추정(Go2 캘리브 파일은 10°)")
    p.add_argument("--ceil-min-elev", type=float, default=14.6, help="천장 조명 판정: 카메라 수평선 위 이 각도(도)보다 높은 상자만")
    p.add_argument("--ceiling-filter", action="store_true",
                   help="천장 조명을 조준 후보에서 뺀다(상자 포화 > 5%% 또는 넓은 조각 '천장 조명 − 작은 표지' > −0.01 이고 카메라 수평선 위 --ceil-min-elev 도 넘음)")
    p.add_argument("--facility", action="store_true", default=True, help="시설확인 멈춤(VLM, 계단)을 함께 낸다(기본 켬)")
    p.add_argument("--no-facility", dest="facility", action="store_false", help="시설확인을 끈다")
    p.add_argument("--facility-v2", action="store_true", default=True,
                   help="시설확인 v2(기본 켬): 번호판 후보가 있어도 지도가 시설을 예측하면 묻고, 5 m dedup 은 찾은 자리만, 질문은 비동기")
    p.add_argument("--no-facility-v2", dest="facility_v2", action="store_false", help="시설확인 v1 로")
    p.add_argument("--search-config", default="",
                   help="층별 찾을 대상 설정(make_search_config.py). 주면 SigLIP 문구·시설 종류·지도 없이 받을 종류를 여기서 받음")
    p.add_argument("--sign-prompts", default="roomstd", choices=["wall", "room", "plate", "roomplate", "roomstd"],
                   help="SigLIP 표지 문구 묶음. 2026-10-05 기본 roomstd = 표준 호실 표지판 두 문구(설정 파일 없이도 13개 층 v2 와 같음). "
                        "wall = 10-05 이전 기본 '벽의 작은 표지'(예전 결과를 다시 낼 때). --search-config 를 주면 그 문구가 우선")
    p.add_argument("--floor-band", default="0.6,2.8",
                   help="2026-10-06 상한 2.2 → 2.8 m(문 위·천장 근처 화장실·돌출 표지가 2.2 m 넘어 빠졌다. G1F 327·330·331초). 바닥 기준 높이 띠 'lo,hi'(m). 바닥면은 주행 앞부분 깊이 20장면으로 잼(게이트와 무관). "
                        "2026-10-04 사람 판정 4개 층: 2배 뺌+띠(V01B) 멈춤 71·번호판 31(44%%) vs 처음 설정 68·26(38%%). 빈 문자열이면 안 봄")
    p.add_argument("--ask-min-scenes", type=int, default=1,
                   help="벽 자리가 서로 다른 장면에서 이만큼 잡혀야 멈춤(1=처음 보자마자, 기존)")
    p.add_argument("--ask-same-m", type=float, default=2.0,
                   help="이미 멈춘 자리에서 수평 이 거리(m) 안의 후보는 같은 대상으로 보고 다시 멈추지 않는다. 0이면 안 봄. "
                        "2026-10-04: 2배 단계를 빼니 상자가 흔들려 같은 번호판에 또 멈춤(G2F 0~420초 1 m 안 재멈춤 3→7)")
    p.add_argument("--ask-same-sec", type=float, default=60.0,
                   help="--ask-same-m 을 최근 이 초 안에 멈춘 자리에만 적용(0 = 시간 제한 없음). 2026-10-08: 재멈춤의 62%%가 10초 안, "
                        "20%%가 60초 넘어 돌아오며 다시 보기(위치추정 60초 창이라 이건 도움)")
    p.add_argument("--prop-scales", default="1",
                   help="작은 후보를 뽑을 배율(쉼표). 기본 1(원본만). 2026-10-04 조정 8개 층: 2배 단계를 빼니 후보 뽑기 "
                        "280~350 → 66~86 ms, 장면 95%% 194~293 ms, 번호판은 잃지 않고 오히려 더 잡힘(사용자 확인). "
                        "옛 동작(원본+2배)은 1,2")
    p.add_argument("--far-2x-m", type=float, default=0.0,
                   help="2배 확대 후보 뽑기를 깊이가 이보다 먼(또는 빈) 칸에서만(m). 7 = 폭 0.14 m 표지가 12 px 아래가 되는 거리. 0이면 전체")
    p.add_argument("--fac-nomap-accept", action="store_true",
                   help="지도에 그 시설 종류가 아예 없을 때 답+5 m 만으로 받음. 2026-10-04 G1 엘베 4건 전부 일반 문(사용자 판정)이라 기본 끔")
    p.add_argument("--fac-free-kinds", default="계단", help="v2: 지도 예측 없이도 받는 시설(쉼표). 나머지는 지도가 앞쪽에 예측할 때만")
    p.add_argument("--fac-map-sem", default="", help="v2 지도 예측용 의미지도 json")
    p.add_argument("--fac-map-base", default="", help="v2 지도 예측을 위상 방식으로(base_map.png, 벽 판정). 비우면 옛 미터 방식(8 m·±60°)")
    p.add_argument("--fac-map-ppm", type=float, default=0.0, help="v2 지도 예측용 지도 px/m")
    p.add_argument("--facility-kinds", default="계단",
                   help="시설확인으로 받을 종류(쉼표). 계단,엘리베이터,화장실. 기본 계단(엘베는 4·5·6층에서 4번 중 1번만 맞음)")
    p.add_argument("--stop-rule", default="amb_sensor_ok", choices=["amb_sensor_ok", "amb_or_jump"],
                   help="멈춤 허용 규칙. amb_sensor_ok(2026-10-07 기본, 실험 지침서 E4): 장소 애매 + 오돔·센서 정상. "
                        "amb_or_jump: 10-06 까지(애매 또는 튐+흩어짐). 녹화 13개 층에서 튐만으로 켜진 초는 거의 없음")
    p.add_argument("--jump-min-spread", type=float, default=5.0,
                   help="튐으로 켤 때 비슷한 후보 퍼짐 최소(m). 0이면 튐만으로 늘 켬(09-30 이전 동작)")
    p.add_argument("--sparse", action="store_true", default=True, help="드물게 찾고 따라가기(기본 켬)")
    p.add_argument("--no-sparse", dest="sparse", action="store_false", help="예전 방식(과거 1.2초 프레임마다 후보)")
    p.add_argument("--sparse-hist", type=int, default=2, help="sparse: 따라올 지난 장면 수")
    p.add_argument("--sparse-rate", type=int, default=2, choices=[1, 2], help="sparse: 1초에 후보 뽑는 장수(기본 2)")
    p.add_argument("--rule", default="ask", choices=["event", "ask"],
                   help="event=5 m 안에 처음 들어오면 상위 3개(12번 줄기), ask=모호할 때 1곳씩 요청(13번 방식)")
    p.add_argument("--ask-dmax", type=float, default=0.0, help="ask: 이 거리(m) 안의 구간만. 0이면 거리 안 봄(13번 그대로)")
    p.add_argument("--ask-near", action="store_true", help="ask: 점수 순 대신 가까운 순으로 1곳")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("replay", help="녹화 데이터를 실시간처럼 흘려 넣는다")
    add_algo_args(r)
    G = "reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/12_4층전체주행"
    r.add_argument("--root", default="/data2/kci_datasets/4F_RGB_FULL_V0_1")
    r.add_argument("--frames-csv", default=f"{G}/frames_4F_full_1s.csv", help="장면 시각 목록(기존과 같은 초)")
    r.add_argument("--depth", default="/data3/kci_datasets/depth_cache_sign_eval/4F_full.npz")
    r.add_argument("--odom", default="/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv")
    r.add_argument("--t0", type=float, default=1783340324.059948032)
    r.add_argument("--reference", default=f"{G}/events_D5_q90_track.json")
    r.add_argument("--max-t", type=float, default=0, help="이 시각까지만(속도 확인용). 0이면 끝까지")
    r.add_argument("--name", default="4F_full", help="구간 이름(기록용)")
    r.add_argument("--save-regions", action="store_true", help="장면마다 구간을 저장(진단용)")
    r.add_argument("--realtime", action="store_true", help="녹화 속도 그대로 흘려 넣어 실제 지연을 잰다")
    r.add_argument("--gate", default="", help="위치추정 모호함 기록(json: t, amb, jump). 멈춤은 모호함·튐일 때만")
    r.add_argument("--fac-map-topk", default="", help="v2 지도 예측용 위치추정 topk_series.csv")
    r.add_argument("--out", required=True)
    lv = sub.add_parser("live", help="Go2 조준 콘솔에서 실시간으로 받아 멈춤 이벤트를 낸다")
    add_algo_args(lv)
    lv.add_argument("--go2", default="http://<GO2_IP>:8088")
    lv.add_argument("--warmup", type=float, default=3.0, help="첫 장면 시각(받기 시작 후 초). 녹화 데이터도 3초부터였다")
    lv.add_argument("--aim", action="store_true", help="멈춤 뒤 다시 찾은 점으로 조준(초점은 안 건드림)")
    lv.add_argument("--retarget-delay", type=float, default=1.0,
                    help="멈춤 판단 뒤 몇 초 지나 다시 찾을지(로봇이 설 시간). 투영이라 계산은 수 ms")
    lv.add_argument("--clock-synced", action="store_true", help="노트북과 Go2 시계가 맞을 때만 실제 지연을 찍는다")
    lv.add_argument("--loc-url", default="",
                    help="위치추정 1초 기록 주소(GET ?after=stamp → JSON 목록). 주면 replay 의 --gate 처럼 애매·튐일 때만 멈춤")
    lv.add_argument("--loc-missing", default="amb", choices=["amb", "quiet"],
                    help="위치추정 기록이 없거나 3초보다 오래된 장면: amb=애매로 보고 멈춤 허용(현장 기본), quiet=replay 처럼 멈춤 없음(녹화 대조)")
    lv.add_argument("--t-origin", type=float, default=0.0,
                    help="장면 시각 원점(로봇 시계 초). 주면 장면을 원점 기준 정수 초에 맞춘다(녹화 대조용). 0이면 첫 프레임")
    lv.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="빠른 버전이 기존 함수와 같은지 확인")
    v.add_argument("--root", default="/data2/kci_datasets/4F_RGB_FULL_V0_1")
    v.add_argument("--frames-csv", default=f"{G}/frames_4F_full_1s.csv")
    v.add_argument("--max-t", type=float, default=40.0)
    v.add_argument("--t-from", type=float, default=0.0, help="이 시각부터 대조(후보가 많은 구간만 볼 때)")
    args = ap.parse_args()
    if args.mode == "replay":
        replay(args)
    elif args.mode == "live":
        live(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
