#!/usr/bin/env python3
"""의미지도 시설 판정 — 위상 방식(2026-10-04).

안내도는 미터 지도가 아니다(축척 불균일, 사용자 지시: 거리·각도 예측 금지, 순서·좌우·인접만). 그래서 미터 반경
(이전 FacilityMap 8 m·±60°, 누적 층의 '관측을 지도에 찍기')을 쓰지 않고 다음만 본다.
  보임   지도 위 후보 자리에서 시설 상자 둘레 어느 점까지 벽(어두운 픽셀)에 안 막히는가. 축척과 무관.
  앞쪽   그 점이 후보 방향(yaw) 앞쪽 반평면인가.
  좌우   관측이 화면 왼쪽/오른쪽(방향 10° 넘을 때만)이면 지도에서도 같은 쪽인가(외적 부호).
  순서   앞쪽 반평면의 방·시설(상자 가까운 점) 중 가까운 순서가 RANK 안인가. 거리 값이 아니라 순서만 쓴다.
yaw 는 지도 m 좌표(오른쪽 x, 위쪽 y) 기준, 픽셀은 y 아래.
"""
import json

import cv2
import numpy as np

TYPES = {"계단": "stair", "엘리베이터": "elevator", "화장실": "restroom"}


class TopoMap:
    def __init__(self, base_png, sem_json, rank=4, wall_thr=100, edge_pts=24):
        g = cv2.imread(base_png, cv2.IMREAD_GRAYSCALE)
        self.wall = g < wall_thr
        self.h, self.w = g.shape
        d = json.load(open(sem_json))
        self.fac = {k: [s["bbox_px"] for s in d["spaces"] if s.get("type") == v] for k, v in TYPES.items()}
        self.marks = [s["bbox_px"] for s in d["spaces"] if s.get("bbox_px")]      # 순서 비교 대상(방·시설 전부)
        self.rank, self.edge_pts = rank, edge_pts

    def _visible(self, x, y, px, py, box):
        """후보 (x,y) → 점 (px,py) 선분이 벽에 막히지 않는가. 시설 상자 안쪽 구간은 보지 않는다(계단실 안 벽)."""
        n = int(max(abs(px - x), abs(py - y)) / 2) + 2
        xs = np.linspace(x, px, n)
        ys = np.linspace(y, py, n)
        bx, by, bw, bh = box
        out = ~((xs >= bx) & (xs <= bx + bw) & (ys >= by) & (ys <= by + bh))
        xi = np.clip(xs[out].astype(int), 0, self.w - 1)
        yi = np.clip(ys[out].astype(int), 0, self.h - 1)
        return not self.wall[yi, xi][2:].any()          # 출발점 근처 2칸은 로봇 자리 잡음이라 뺌

    def _edge(self, box):
        bx, by, bw, bh = box
        k = self.edge_pts // 4
        t = np.linspace(0, 1, k, endpoint=False)
        return np.concatenate([np.c_[bx + bw * t, np.full(k, by)], np.c_[np.full(k, bx + bw), by + bh * t],
                               np.c_[bx + bw * (1 - t), np.full(k, by + bh)], np.c_[np.full(k, bx), by + bh * (1 - t)]])

    def _order(self, x, y, fwd):
        """앞쪽 반평면 표지(방·시설) 가까운 점까지 픽셀 거리 목록(순서 비교용)."""
        ds = []
        for bx, by, bw, bh in self.marks:
            px, py = min(max(x, bx), bx + bw), min(max(y, by), by + bh)
            v = np.array([px - x, -(py - y)])
            if v @ fwd > 0:
                ds.append(float(np.hypot(*v)))
        return sorted(ds)

    def match(self, x, y, yaw, kind, side=0):
        """후보 (x, y, yaw) 에서 kind 시설이 '보이고·앞쪽·(같은 쪽)·가까운 순서 RANK 안'이면 True.
        side: +1 왼쪽, −1 오른쪽, 0 상관없음."""
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        order = None
        for box in self.fac.get(kind, []):
            bx, by, bw, bh = box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return True                              # 시설 안에 서 있음
            for px, py in self._edge(box):
                v = np.array([px - x, -(py - y)])
                if v @ fwd <= 0:
                    continue
                if side and np.sign(fwd[0] * v[1] - fwd[1] * v[0]) != side:
                    continue
                if not self._visible(x, y, px, py, box):
                    continue
                if order is None:
                    order = self._order(x, y, fwd)
                cx, cy = min(max(x, bx), bx + bw), min(max(y, by), by + bh)
                d = float(np.hypot(cx - x, cy - y))
                if sum(o < d - 1e-6 for o in order) < self.rank:
                    return True
                break                                    # 이 상자는 순서 밖
        return False
