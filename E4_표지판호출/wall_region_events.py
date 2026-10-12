#!/usr/bin/env python3
"""멈췄을 때 '확인할 벽 구간'을 몇 개 낼지, 그 안에 번호판이 들어오는지 잰다.

RGB는 번호판 하나를 고르지 못한다(30_관측이벤트: 두드러진 작은 조각이 장당 40~170개).
그래서 조준점 대신 구간을 낸다. 두드러진 후보(아무 벽 조각 상위 10%보다 두드러짐)를
가까운 것끼리 묶어 구간을 만들고, 묶인 후보의 두드러짐 × 관측 이력 합으로 순서를 매긴다.
멈출 때마다 상위 N개 구간을 고화질 카메라가 확인한다고 보고,
  - 번호판(물체 단위)이 상위 N개 구간 안에 들어오는가
  - 그 구간들이 화면을 얼마나 덮는가 (넓게 덮어서 맞히는 건 의미 없다)
  - 같은 개수·같은 크기의 무작위 구간이면 얼마나 들어오는가
를 함께 본다. 각 1초 장면을 '멈춘 순간'으로 가정한다. 정답은 채점에만 쓴다.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from event_criteria import SEGS, D, salience  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920")


def depth_ok(dimg, box, vp, zmax, floor_h=-0.25):
    """깊이로 '읽을 수 있는 거리의 벽 표면'인지 본다. 표지 높이 규칙은 넣지 않는다.
    zmax : 이보다 멀면 제외(리얼센스 깊이와 고화질 판독이 둘 다 약해지는 거리)
    바닥 : 카메라 기울기를 소실점으로 보정한 높이가 카메라보다 floor_h 아래면 바닥면으로 본다.
    깊이가 없으면(0) 먼 곳이거나 반사면이라 제외한다."""
    FY, CY = 606.75, 240.5
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    p = dimg[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    v = p[p > 0]
    if len(v) < 3:
        return False, None
    z = float(np.median(v)) / 1000.0
    th = np.arctan2((vp[1] - CY), FY) if vp is not None else 0.0
    yc = ((y0 + y1) / 2 - CY) / FY * z
    h = -yc * np.cos(th) + z * np.sin(th)
    return (z <= zmax and h > floor_h), z


FX, FY, CX, CY = 606.80, 606.75, 310.49, 240.52


def wall_planes(dimg, vp, zmax=7.0, tol=0.05, iters=200, seed=0):
    """장면 깊이에서 복도 양쪽 벽면을 맞춘다. 카메라 왼쪽 점(x<-0.3 m)과 오른쪽 점(x>0.3 m)을
    따로 모아 각각 세로 평면 하나를 RANSAC으로 찾는다(한쪽 벽이 점이 많아 두 번 잡히는 것을 막는다).
    세로 판정은 소실점으로 잡은 카메라 기울기를 보정한 뒤 법선의 위아래 성분이 작은지로 본다.
    상자 앞·옆면처럼 작은 세로면은 벽보다 점이 적어서 벽으로 잡히지 않는다."""
    th = np.arctan2((vp[1] - CY), FY) if vp is not None else 0.0
    up = np.array([0.0, -np.cos(th), np.sin(th)])          # 카메라 좌표에서 위쪽
    ys, xs = np.mgrid[0:480:4, 0:640:4]
    z = dimg[::4, ::4].astype(np.float32) / 1000.0
    ok = (z > 0.3) & (z <= zmax)
    P = np.stack([(xs[ok] - CX) / FX * z[ok], (ys[ok] - CY) / FY * z[ok], z[ok]], 1)
    rng = np.random.default_rng(seed)
    planes = []
    for side in (P[P[:, 0] < -0.3], P[P[:, 0] > 0.3]):
        if len(side) < 200:
            continue
        best, bn = None, 0
        for _ in range(iters):
            q = side[rng.choice(len(side), 3, replace=False)]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            if np.linalg.norm(n) < 1e-6:
                continue
            n /= np.linalg.norm(n)
            if abs(n @ up) > 0.2:                            # 세로면만
                continue
            d = -n @ q[0]
            k = int((np.abs(side @ n + d) < tol).sum())
            if k > bn:
                best, bn = (n, d), k
        if best is None or bn < 0.15 * len(side):
            continue
        n, d = best
        if d > 0:                                            # 카메라 쪽이 음수
            n, d = -n, -d
        planes.append((n, d))
    return planes


def off_wall(dimg, box, planes):
    """후보가 가장 가까운 벽면보다 카메라 쪽으로 몇 m 나와 있는지. 벽이 없거나 깊이가 없으면 None."""
    if not planes:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    v = dimg[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    v = v[v > 0]
    if len(v) < 3:
        return None
    zc = float(np.median(v)) / 1000.0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    pt = np.array([(cx - CX) / FX * zc, (cy - CY) / FY * zc, zc])
    ds = [-(pt @ n + d) for n, d in planes]
    return min(ds, key=abs)


def context_busy(edges, box, k=1.0):
    """후보 둘레(후보 크기의 k배 폭)에서 가장자리 픽셀 비율. 민벽에 혼자 붙은 판은 낮고,
    게시판의 종이·글자 조각은 둘레도 글자와 종이라 높다. 사각형 판 윤곽 찾기는 판이 합쳐지거나
    놓쳐서 쓸 수 없었다(번호판 7% 대 기타 14%로 안 갈림)."""
    x0, y0, x1, y1 = box
    w, h = max(x1 - x0, 6), max(y1 - y0, 6)
    a0, b0 = int(max(0, x0 - k * w)), int(max(0, y0 - k * h))
    a1, b1 = int(min(640, x1 + k * w)), int(min(480, y1 + k * h))
    sub = edges[b0:b1, a0:a1]
    m = np.ones(sub.shape, bool)
    m[int(max(0, y0 - b0)):int(y1 - b0) + 1, int(max(0, x0 - a0)):int(x1 - a0) + 1] = False
    return float(sub[m].mean()) if m.sum() > 20 else None


def regions(recs, sal, thr, link=40, pad=10, maxw=260, maxh=200, keep=None, wall_w=None, topk=0):
    """wall_w : 후보별 가중치(벽면에서 떨어진 후보는 낮게). 지우지 않고 순서만 뒤로 민다.
    topk   : 0이면 묶인 후보 전부의 합, k면 가장 강한 k개만 합한다(조각이 많은 판이 개수로 이기는 것을 막는다)."""
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
        if x1 - x0 > maxw or y1 - y0 > maxh:        # 너무 크면 가운데 기준으로 자른다(한 번에 볼 크기)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1 = cx - min(maxw, x1 - x0) / 2, cx + min(maxw, x1 - x0) / 2
            y0, y1 = cy - min(maxh, y1 - y0) / 2, cy + min(maxh, y1 - y0) / 2
        vals = sorted((sal[i] * (1 + min(recs[i].get("support", 0), 2)) * (1.0 if wall_w is None else wall_w[i])
                       for i in g), reverse=True)
        score = sum(vals[:topk] if topk else vals)
        bb = [max(0, x0), max(0, y0), min(640, x1), min(480, y1)]
        out.append((score, bb))
    out.sort(key=lambda x: -x[0])
    return out


def inside(g, b):
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def main():
    import os
    from geometry_filter import vanishing_point
    rng = np.random.default_rng(0)
    thr = json.load(open(R / "30_관측이벤트/grid.json"))["threshold"]
    zmax = float(os.environ.get("ZMAX", "0"))       # 0이면 깊이 조건 없음
    setname = os.environ.get("SET", "4F")          # 층 선택(sign_segsets). 4F가 기본이고 기존 결과와 같다
    import sign_segsets
    segs = sign_segsets.get(setname)
    dz = {s["name"]: np.load(s["depth"]) for s in segs}
    roots = {s["name"]: s["root"] for s in segs}
    pz = []
    scenes = []
    for name, pp, gp in [(s["name"], s["proposals"], s["gt"]) for s in segs]:
        runs = {float(json.loads(l)["t"]): json.loads(l) for l in open(pp)}
        for l in open(gp):
            d = json.loads(l)
            t = float(d["t"])
            fpath = roots[name] / runs[t]["frame"]
            bgr = cv2.imread(str(fpath))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            recs = runs[t]["records"]
            sal = [salience(gray, grad, r["box"]) for r in recs]
            plaques = [(g, f"{name}#{o}") for (c, *g), o in zip(d["items"], d["object_id"]) if c == "plaque"]
            keep = None
            dimg = dz[name][f"{t:.1f}"]
            vp = vanishing_point(bgr)
            for g, _ in plaques:
                pz.append(depth_ok(dimg, g, vp, 99)[1])
            if zmax > 0:
                keep = [depth_ok(dimg, r["box"], vp, zmax)[0] for r in recs]
            wall_w = None
            if os.environ.get("WALL") == "1":
                pl = wall_planes(dimg, vp)
                wall_w = []
                for r in recs:
                    o = off_wall(dimg, r["box"], pl)
                    wall_w.append(1.0 if o is None or abs(o) <= 0.25 else 0.3)
            if os.environ.get("BUSY") == "1":
                # 같은 장면 두드러진 후보들의 둘레 빽빽함 중앙값보다 빽빽하면 비례해서 뒤로(최저 0.3배)
                edges = cv2.Canny(gray, 40, 120) > 0
                busy = [context_busy(edges, r["box"]) for r in recs]
                ref = np.median([b for b, s_ in zip(busy, sal) if b is not None and s_ >= thr] or [1.0])
                wall_w = wall_w or [1.0] * len(recs)
                wall_w = [w_ * (1.0 if b is None or b <= ref else max(0.3, ref / b)) for w_, b in zip(wall_w, busy)]
            scenes.append(dict(name=name, t=t, frame=runs[t]["frame"] if setname == "4F" else str(fpath), plaques=plaques,
                               regs=regions(recs, sal, thr, keep=keep, wall_w=wall_w, topk=int(os.environ.get("TOPK", "0")))))
    zz = np.array([z for z in pz if z is not None])
    print(f"번호판 깊이: 유효 {len(zz)}/{len(pz)}, 중앙 {np.median(zz):.1f} m, 5 m 이하 {int((zz <= 5).sum())}, "
          f"6 m 이하 {int((zz <= 6).sum())}, 7 m 이하 {int((zz <= 7).sum())}, 8 m 이하 {int((zz <= 8).sum())}")
    print(f"깊이 조건: {'없음' if zmax <= 0 else f'{zmax} m 이내 + 바닥면 제외'}")
    objs = {o for s in scenes for _, o in s["plaques"]}
    n_inst = sum(len(s["plaques"]) for s in scenes)
    print(f"물리 번호판 {len(objs)}개, 번호판 관측 {n_inst}개, 장면 {len(scenes)}장 (각 장면을 멈춘 순간으로 가정)\n")
    print(f"{'구간 N':>6} | {'덮는 면적':>8} | {'번호판 관측 안에':>14} {'물체':>8} | {'같은 크기 무작위':>14}")
    rows = []
    for N in (1, 2, 3, 5, 8):
        hit_i, hit_o, area, rnd = 0, set(), [], 0.0
        for s in scenes:
            top = [b for _, b in s["regs"][:N]]
            m = np.zeros((480, 640), bool)
            for b in top:
                m[int(b[1]):int(b[3]), int(b[0]):int(b[2])] = True
            area.append(m.mean())
            for g, o in s["plaques"]:
                if any(inside(g, b) for b in top):
                    hit_i += 1
                    hit_o.add(o)
            for _ in range(20):                    # 같은 개수·같은 크기, 위치만 무작위
                rb = []
                for b in top:
                    w, h = b[2] - b[0], b[3] - b[1]
                    x, y = rng.uniform(0, 640 - w), rng.uniform(0, 480 - h)
                    rb.append([x, y, x + w, y + h])
                rnd += sum(any(inside(g, b) for b in rb) for g, _ in s["plaques"]) / 20
        rows.append(dict(N=N, area=float(np.mean(area)), inst=hit_i, obj=len(hit_o), rand=rnd))
        print(f"{N:6d} | {np.mean(area) * 100:7.0f}% | {hit_i:6d}/{n_inst:<7d} {len(hit_o):3d}/{len(objs):<4d} | {rnd:8.1f}/{n_inst}")
    tag = (f"z{zmax:g}" if zmax > 0 else "nodepth") + ("_wall" if os.environ.get("WALL") == "1" else "") \
        + ("_busy" if os.environ.get("BUSY") == "1" else "") \
        + (f"_top{os.environ['TOPK']}" if os.environ.get("TOPK", "0") != "0" else "")
    # OUT 으로 결과 폴더를 직접 줄 수 있다(31_벽구간이벤트 기준, 예: OUT=01_거리제한/4_7m_채택)
    out = R / "31_벽구간이벤트" / (os.environ["OUT"] if os.environ.get("OUT") else (tag if setname == "4F" else f"{setname}_{tag}"))
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "results.json", "w"), ensure_ascii=False, indent=1)
    json.dump({f"{s['name']}_{s['t']}": {"frame": s["frame"], "regions": s["regs"][:8],
                                          "plaques": [g for g, _ in s["plaques"]]} for s in scenes},
              open(out / "regions.json", "w"))


if __name__ == "__main__":
    main()
