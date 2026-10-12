#!/usr/bin/env python3
"""깊이로 '후보가 붙은 면이 세로(벽)인가 가로(천장·바닥)인가'를 잰다(2026-10-04).
번호판은 벽에, 조명은 천장에 붙는다는 물리만 쓴다. 카메라 기울기·높이나 건물 천장 높이를 상수로 두지 않는다.
  1) 바닥면: 화면 아래쪽 깊이 점으로 RANSAC 평면 → 위 방향(중력 반대) 벡터 up 과 카메라 높이.
  2) 면 방향: 깊이에서 점마다 법선(이웃 차분 외적). 후보 상자를 넓힌 영역의 법선이 up 과 이루는 각.
     |n·up| 이 크면 가로면(천장·바닥), 작으면 세로면(벽).
  3) 높이: 후보 깊이 점의 바닥 위 높이를 카메라 높이와 비교(같은 장면 안에서 잰 값끼리만 비교).
조명은 포화돼 깊이가 비기 쉬워 상자 안보다 둘레(넓힌 영역)를 본다.
"""
import numpy as np

FX, FY, CX, CY = 606.80, 606.75, 310.49, 240.52


def backproject(d, step=4):
    """깊이(mm) → 카메라 좌표 점(m), step 간격 격자. x 오른쪽, y 아래, z 앞."""
    z = d[::step, ::step].astype(np.float32) / 1000.0
    vs, us = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
    X = (us - CX) / FX * z
    Y = (vs - CY) / FY * z
    return np.dstack([X, Y, z]), us, vs


def floor_plane(P, rng=np.random.default_rng(0), iters=200, tol=0.03, zmax=6.0):
    """화면 아래 45% 의 점으로 바닥 평면. 돌려줌: up(단위벡터, 카메라 좌표), 카메라 높이(m), 지지 비율. 못 찾으면 None."""
    h = P.shape[0]
    Q = P[int(h * 0.55):].reshape(-1, 3)
    Q = Q[(Q[:, 2] > 0.2) & (Q[:, 2] < zmax)]
    if len(Q) < 200:
        return None
    best = None
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if n[1] > 0:
            n = -n                                   # up 은 화면 위쪽(-y) 성분
        if -n[1] < 0.6:                              # 바닥은 대체로 수평: 위 방향이 -y 쪽이어야
            continue
        dist = Q @ n - a @ n
        inl = np.abs(dist) < tol
        if best is None or inl.sum() > best[0]:
            best = (inl.sum(), n, a)
    if best is None or best[0] < 0.2 * len(Q):
        return None
    _, n, a = best
    Qi = Q[np.abs(Q @ n - a @ n) < tol]               # 지지점으로 다시 맞춤(최소제곱)
    c0 = Qi.mean(0)
    _, _, vt = np.linalg.svd(Qi - c0, full_matrices=False)
    n = vt[2] if vt[2][1] < 0 else -vt[2]
    cam_h = float(-(c0 @ n))                        # 카메라(원점)의 바닥 위 높이 = -(바닥점·up)
    return n, cam_h, best[0] / len(Q)


def normals(P):
    """격자 점 P 의 법선(이웃 차분 외적). 깊이 끊김(이웃 거리 > 0.08·z)은 NaN."""
    dx = P[1:-1, 2:] - P[1:-1, :-2]
    dy = P[2:, 1:-1] - P[:-2, 1:-1]
    n = np.cross(dx, dy)
    z = P[1:-1, 1:-1, 2]
    ok = (z > 0.2) & (np.linalg.norm(dx, axis=2) < 0.08 * z + 0.02) & (np.linalg.norm(dy, axis=2) < 0.08 * z + 0.02) \
         & (P[1:-1, 2:, 2] > 0) & (P[1:-1, :-2, 2] > 0) & (P[2:, 1:-1, 2] > 0) & (P[:-2, 1:-1, 2] > 0)
    nn = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    n[~ok] = np.nan
    out = np.full(P.shape, np.nan, np.float32)
    out[1:-1, 1:-1] = n
    return out


def box_surface(d, box, plane=None, step=4, grow=0.5):
    """후보 상자 하나의 면 특성. 돌려줌 dict(horiz=가로면 비율, vert=세로면 비율, n=법선 수,
    h=상자 깊이 점의 바닥 위 높이 중앙(m), cam_h=카메라 높이, ok=바닥 찾음)."""
    P, us, vs = backproject(d, step)
    if plane is None:
        plane = floor_plane(P)
    N = normals(P)
    x0, y0, x1, y1 = box[:4]
    w, hh = x1 - x0, y1 - y0
    X0, X1, Y0, Y1 = x0 - grow * w, x1 + grow * w, y0 - grow * hh, y1 + grow * hh
    m = (us >= X0) & (us <= X1) & (vs >= Y0) & (vs <= Y1)
    res = dict(ok=plane is not None, horiz=np.nan, vert=np.nan, n=0, h=np.nan, cam_h=np.nan)
    if plane is None:
        return res
    up, cam_h, _ = plane
    nn = N[m]
    nn = nn[~np.isnan(nn[:, 0])]
    res["cam_h"] = cam_h
    if len(nn) >= 5:
        c = np.abs(nn @ up)
        res.update(horiz=float((c > 0.8).mean()), vert=float((c < 0.35).mean()), n=int(len(nn)))
    mi = (us >= x0) & (us <= x1) & (vs >= y0) & (vs <= y1)
    pts = P[m if mi.sum() == 0 else mi].reshape(-1, 3)
    pts = pts[pts[:, 2] > 0.2]
    if len(pts):
        res["h"] = float(np.median(pts @ up) + cam_h)
    return res
