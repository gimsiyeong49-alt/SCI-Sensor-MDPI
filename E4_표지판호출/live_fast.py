#!/usr/bin/env python3
"""실시간 실행기용 빠른 버전. 결과는 기존 함수와 비트 단위로 같아야 한다(verify 로 확인).

small_proposals.merge 와 accumulate_proposals 의 이어 붙이기는 파이썬 이중 반복이라 느렸다
(2026-09-30 측정: 후보 한 장 634 ms 중 merge 384 ms, 누적 한 장면 1,344 ms).
바깥 반복(차례로 넣으며 상자가 커지거나 늘어나는 순서)은 그대로 두고, 안쪽의 '모든 상자와 비교해
처음 걸리는 것 / 가장 가까운 것 찾기'만 numpy 로 한 번에 계산한다.
기존 코드는 float32 점과 파이썬 float 를 섞어 float64 로 계산하므로 여기서도 float64 로 한다.
"""

from __future__ import annotations

import numpy as np

import small_proposals as SP


def merge_fast(boxes, thr=0.3):
    """small_proposals.merge 와 같다. 넓이 큰 순(같으면 입력 순)으로 넣으며, IoU > thr 인 첫 상자에 합친다."""
    order = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    out = np.zeros((len(order), 4), np.float64)
    n = 0
    for b in order:
        if n:
            o = out[:n]
            ix = np.maximum(0, np.minimum(o[:, 2], b[2]) - np.maximum(o[:, 0], b[0]))
            iy = np.maximum(0, np.minimum(o[:, 3], b[3]) - np.maximum(o[:, 1], b[1]))
            inter = ix * iy
            with np.errstate(divide="ignore", invalid="ignore"):
                v = inter / ((o[:, 2] - o[:, 0]) * (o[:, 3] - o[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            v[(ix == 0) | (iy == 0)] = 0.0
            hit = np.flatnonzero(v > thr)
        else:
            hit = []
        if len(hit) == 0:
            out[n] = b
            n += 1
        else:
            h = hit[0]
            o = out[h]
            out[h] = [min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])]
    return [tuple(int(v) for v in r) for r in out[:n]]


def group_fast(boxes, pad=3):
    """small_proposals.group 과 같다. 겹치는(pad 여유) 상자끼리 묶은 연결 성분을, 가장 작은 번호 순으로 낸다.
    기존은 모든 쌍을 파이썬으로 비교했다(후보 2,000개면 200만 쌍, 180 ms). 겹침 판정만 numpy 로 한 번에 한다.
    묶음(연결 성분)과 그 순서(성분의 가장 작은 번호), 합친 상자(최소·최대)는 합치는 순서와 무관하므로 같다."""
    n = len(boxes)
    if n == 0:
        return []
    B = np.array(boxes, np.int64)
    ov = ((B[:, None, 0] - pad <= B[None, :, 2]) & (B[None, :, 0] - pad <= B[:, None, 2])
          & (B[:, None, 1] - pad <= B[None, :, 3]) & (B[None, :, 1] - pad <= B[:, None, 3]))
    ii, jj = np.nonzero(np.triu(ov, 1))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    g = {}
    for i, b in enumerate(boxes):
        r = find(i)
        o = g.get(r)
        g[r] = b if o is None else (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
    out = [b for b in g.values()
           if SP.MIN_W <= b[2] - b[0] <= SP.MAX_W and SP.MIN_H <= b[3] - b[1] <= SP.MAX_H]
    return [tuple(int(v) for v in b) for b in out]


def propose_fast(bgr, scales=(1, 2)):
    """small_proposals.propose 와 같다(merge 만 빠른 것으로)."""
    import cv2
    boxes = []
    for s in scales:
        img = bgr if s == 1 else cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        boxes += [(b[0] / s, b[1] / s, b[2] / s, b[3] / s) for b in SP._propose_one(img)]
    boxes = merge_fast(boxes)
    boxes = boxes + group_fast(boxes)
    return merge_fast(boxes, thr=0.85)


def attach(boxes, nb, pts, st, off, tol, support, added):
    """attach_basic 과 같은 결과를 더 빨리 낸다(2026-09-30: 후보 1,500개 넘는 장면에서 1초 → 수십 ms).

    기존 규칙에서 결과를 바꾸는 새 상자는 그 점에서 거리 tol 안에 있는 것뿐이다.
      - 기준 상자 최소 거리 db ≤ tol 이면: 새 상자가 이기려면 거리 < db ≤ tol.
      - db > tol 이면: 거리 ≤ tol 인 새 상자가 있으면 그중 가장 가까운 것, 없으면 새로 붙인다.
    그래서 새 상자 중심을 tol 크기 칸에 넣어 두고, 점 둘레 3×3 칸만 본다(L1 ≤ tol 이면 가로·세로 모두 ≤ tol).
    같은 거리면 앞 번호(기준 상자 > 먼저 붙은 새 상자)가 이긴다. 거리는 기존과 같은 float64 식으로 잰다."""
    P = pts.reshape(-1, 2).astype(np.float64)
    P32 = pts.reshape(-1, 2)
    S = st.reshape(-1)
    nbx = len(boxes)
    if nbx:
        cb = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], np.float64)
        D = np.abs(cb[None, :, 0] - P[:, None, 0]) + np.abs(cb[None, :, 1] - P[:, None, 1])
        db_arg = D.argmin(1)
        db_min = D[np.arange(len(P)), db_arg]
    cell = float(tol) if tol > 0 else 1.0
    grid = {}
    cen = []

    def put(i):
        a = added[i]["box"]
        c = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
        cen.append(c)
        grid.setdefault((int(c[0] // cell), int(c[1] // cell)), []).append(i)

    for i in range(len(added)):
        put(i)
    for k in range(len(P)):
        px, py = P[k]
        if not S[k] or not (0 <= px < 640 and 0 <= py < 480):
            continue
        best, bd = None, 1e9
        if nbx and db_min[k] < bd:
            best, bd = int(db_arg[k]), float(db_min[k])
        gx, gy = int(px // cell), int(py // cell)
        ba, da_best = None, None
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for i in grid.get((ix, iy), ()):
                    c = cen[i]
                    d = abs(c[0] - px) + abs(c[1] - py)
                    if da_best is None or d < da_best or (d == da_best and i < ba):
                        ba, da_best = i, d
        if ba is not None and da_best < bd:
            best, bd = nbx + ba, da_best
        if best is not None and bd <= tol:
            if best < nbx:
                support[best] += 1
            else:
                added[best - nbx]["support"] += 1
        else:
            w, h = nb[k][2] - nb[k][0], nb[k][3] - nb[k][1]
            p32 = P32[k]
            added.append({"box": (float(p32[0] - w / 2), float(p32[1] - h / 2),
                                  float(p32[0] + w / 2), float(p32[1] + h / 2)), "support": 1})
            put(len(added) - 1)


def attach_basic(boxes, nb, pts, st, off, tol, support, added):
    """accumulate_proposals 의 이어 붙이기 한 이웃 프레임분. support·added 를 제자리에서 고친다.
    점을 차례로 보며 (기준 상자 + 지금까지 새로 붙은 상자) 중 중심 거리(L1)가 가장 가까운 것을 찾는다.
    같은 거리면 앞의 것(기존 반복의 strict < 와 같다)."""
    if boxes:
        cb = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], np.float64)
    else:
        cb = np.zeros((0, 2))
    ca = [((a["box"][0] + a["box"][2]) / 2, (a["box"][1] + a["box"][3]) / 2) for a in added]
    P = pts.reshape(-1, 2).astype(np.float64)
    S = st.reshape(-1)
    db_all = np.abs(cb[None, :, 0] - P[:, None, 0]) + np.abs(cb[None, :, 1] - P[:, None, 1]) if len(cb) else None
    for k in range(len(P)):
        px, py = P[k]
        if not S[k] or not (0 <= px < 640 and 0 <= py < 480):
            continue
        best, bd = None, 1e9
        if db_all is not None:
            j = int(np.argmin(db_all[k]))
            if db_all[k, j] < bd:
                best, bd = j, db_all[k, j]
        if ca:
            A = np.array(ca)
            da = np.abs(A[:, 0] - px) + np.abs(A[:, 1] - py)
            j = int(np.argmin(da))
            if da[j] < bd:                        # 같은 거리면 기준 상자(앞쪽)가 이긴다
                best, bd = len(boxes) + j, da[j]
        if best is not None and bd <= tol:
            if best < len(boxes):
                support[best] += 1
            else:
                added[best - len(boxes)]["support"] += 1
        else:
            w, h = nb[k][2] - nb[k][0], nb[k][3] - nb[k][1]
            p32 = pts.reshape(-1, 2)[k]
            added.append({"box": (float(p32[0] - w / 2), float(p32[1] - h / 2),
                                  float(p32[0] + w / 2), float(p32[1] + h / 2)), "support": 1})
            ca.append(((added[-1]["box"][0] + added[-1]["box"][2]) / 2, (added[-1]["box"][1] + added[-1]["box"][3]) / 2))


def regions_ext(recs, sal, thr, link=40, pad=10, maxw=260, maxh=200, keep=None, topk=0):
    """wall_region_events.regions 와 같은 구간을 내고, 구간마다 '잘리기 전 원래 묶음' 상자도 함께 낸다(2026-10-01).
    regions 는 묶음이 크면 가로 maxw·세로 maxh 로 가운데를 잘라 버려서, 큰 게시판도 잘린 조각만 보면 좁아 보였다.
    반환: [(score, bb, group_bb), ...] (score 내림차순, bb 는 regions 와 같음)."""
    import collections
    idx = [i for i, s in enumerate(sal) if s >= thr and (keep is None or keep[i])]
    parent = {i: i for i in idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cs = {i: ((recs[i]["box"][0] + recs[i]["box"][2]) / 2, (recs[i]["box"][1] + recs[i]["box"][3]) / 2) for i in idx}
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if abs(cs[i][0] - cs[j][0]) < link and abs(cs[i][1] - cs[j][1]) < link:
                parent[find(i)] = find(j)
    groups = collections.defaultdict(list)
    for i in idx:
        groups[find(i)].append(i)
    out = []
    for g in groups.values():
        bx = np.array([recs[i]["box"] for i in g])
        x0, y0 = bx[:, 0].min() - pad, bx[:, 1].min() - pad
        x1, y1 = bx[:, 2].max() + pad, bx[:, 3].max() + pad
        gb = [max(0, x0), max(0, y0), min(640, x1), min(480, y1)]
        if x1 - x0 > maxw or y1 - y0 > maxh:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1 = cx - min(maxw, x1 - x0) / 2, cx + min(maxw, x1 - x0) / 2
            y0, y1 = cy - min(maxh, y1 - y0) / 2, cy + min(maxh, y1 - y0) / 2
        vals = sorted((sal[i] * (1 + min(recs[i].get("support", 0), 2)) for i in g), reverse=True)
        score = sum(vals[:topk] if topk else vals)
        bb = [max(0, x0), max(0, y0), min(640, x1), min(480, y1)]
        out.append((score, bb, [float(v) for v in gb]))
    out.sort(key=lambda x: -x[0])
    return out


def far_rois(depth, far_m=7.0, tile=80, pad=16, min_frac=0.02):
    """2배 확대 후보 뽑기를 할 영역(2026-10-04). 원본에서 놓치는 건 가로 12 px 아래 표지이고, 가장 작은 번호판(폭 약
    0.14 m)이 12 px 아래가 되는 거리는 FX·0.14/12 ≈ 7 m 라 그보다 멀거나 깊이가 빈(0) 픽셀이 min_frac 넘는 칸(tile px)만
    pad 만큼 넓혀 고른다. 붙은 칸은 가로줄로 이어 붙여 상자 수를 줄인다. 처음엔 덩어리 바깥 상자로 했다가 복도 끝·천장·먼
    벽이 L자로 이어져 화면 88% 를 덮었다. 돌려주는 것: [(x0, y0, x1, y1)] 원본 픽셀."""
    h, w = depth.shape[:2]
    far = (depth == 0) | (depth > far_m * 1000)
    rois = []
    for y in range(0, h, tile):
        run = None
        for x in range(0, w, tile):
            on = far[y:y + tile, x:x + tile].mean() > min_frac
            if on and run is None:
                run = x
            if (not on or x + tile >= w) and run is not None:
                x1 = (x + tile) if on else x
                rois.append((max(0, run - pad), max(0, y - pad), min(w, x1 + pad), min(h, y + tile + pad)))
                run = None
    return rois


def propose_far(bgr, depth, far_m=7.0):
    """propose_fast 와 같되 2배 확대 단계는 far_rois 영역에서만(깊이 없으면 전체)."""
    import cv2
    if depth is None:
        return propose_fast(bgr)
    boxes = [tuple(b) for b in SP._propose_one(bgr)]
    for x0, y0, x1, y1 in far_rois(depth, far_m):
        big = cv2.resize(bgr[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        boxes += [(b[0] / 2 + x0, b[1] / 2 + y0, b[2] / 2 + x0, b[3] / 2 + y0) for b in SP._propose_one(big)]
    boxes = merge_fast(boxes)
    boxes = boxes + group_fast(boxes)
    return merge_fast(boxes, thr=0.85)
