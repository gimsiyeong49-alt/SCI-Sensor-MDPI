#!/usr/bin/env python3
"""지금까지 효과가 확인된 것을 묶은 조합. 멈출 때마다 고화질 카메라가 볼 구간을 정한다.

입력은 SigLIP 구간 재정렬 + 겹침 합치기까지 끝난 regions.json(단계 1~6). 이 스크립트가 하는 일:
  7. 순위   : 2D 벽 자리 기억(수평 0.7 m)에서 그 자리를 볼 때마다 나온 SigLIP 값 평균의 순위와
              현재 구간 순위의 평균으로 다시 매긴다.
  9. 벽 거리: 3D 자리마다 '그 장면 깊이로 맞춘 좌·우 벽면까지 거리'의 중앙값을 쌓고, 0.5 m 넘게 먼 자리는
              벽 가까운 자리들 뒤로 보낸다(지우지 않음). 벽 부착형 건물 설정일 때만(--wall_far).
              0.5 m는 설명 그림에서 쓴 값이고 정답으로 고른 값이 아니다.
  8. 건너뜀 : 3D 기억(수평 0.7 m + 높이 차 0.3 m)으로 이미 확인한 자리는 건너뛴다.
 10. 확인   : 남은 구간 중 앞에서 K개를 확인한다.
 11. 차단   : (--gate) 그 자리의 2D 기억 평균이 '이 구간에서 지금까지 본 모든 구간 SigLIP 값의 중앙값'보다
              낮으면 이번 정지에는 확인하지 않는다(과거만 사용, 추가 계산 없음). 남은 게 헛것뿐이면 조준을 아낀다.
과거·현재만 쓴다. 정답은 채점에만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import region_memory as M  # noqa: E402
import sign_segsets  # noqa: E402
from geometry_filter import vanishing_point  # noqa: E402
from wall_region_events import wall_planes  # noqa: E402

FX, FY, CX, CY = M.FX, M.FY, M.CX, M.CY


def wall_dist(box, dimg, planes):
    if not planes:
        return None
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    p = dimg[max(0, int(y0 + h / 4)):int(y1 - h / 4) + 1, max(0, int(x0 + w / 4)):int(x1 - w / 4) + 1]
    v = p[p > 0]
    if len(v) < 5:
        return None
    z = float(np.median(v)) / 1000
    pt = np.array([((x0 + x1) / 2 - CX) / FX * z, ((y0 + y1) / 2 - CY) / FY * z, z])
    return min(abs(pt @ n + d) for n, d in planes)


def match(spots, xyz, radius, vtol, taken):
    best, bd = None, radius
    for i, s in enumerate(spots):
        if i in taken:
            continue
        if vtol is not None and abs(s["xyz"][2] - xyz[2]) > vtol:
            continue
        d = np.linalg.norm(s["xyz"][:2] - xyz[:2])
        if d < bd:
            best, bd = i, d
    if best is None:
        spots.append({"xyz": xyz.copy(), "n": 0, "sims": [], "wd": []})
        best = len(spots) - 1
    s = spots[best]
    s["xyz"] = (s["xyz"] * s["n"] + xyz) / (s["n"] + 1)
    s["n"] += 1
    return best


def run(seg, B, K, use_rank_mem=True, use_wall_far=True, use_skip=True, far_m=0.5, use_gate=False):
    odom = M.Odom(seg["odom"], seg["t0"])
    dz = np.load(seg["depth"])
    keys = sorted([k for k in B if k.startswith(seg["name"])], key=lambda k: float(k.split("_")[-1]))
    s2, s3, checked, seen = [], [], set(), []
    out = {}
    for k in keys:
        t = float(k.split("_")[-1])
        v = B[k]
        regs = v["regions"]
        dimg = dz[f"{t:.1f}"]
        vp = vanishing_point(cv2.imread(str(seg["root"] / v["frame"])))
        planes = wall_planes(dimg, vp) if use_wall_far else []
        pose = odom(t)
        i2, i3, t2, t3 = [], [], set(), set()
        for sim, b in regs:
            xyz, _ = M.region_world(b, dimg, vp, pose)
            if xyz is None:
                i2.append(None)
                i3.append(None)
                continue
            a = match(s2, xyz, 0.7, None, t2)
            s2[a]["sims"].append(float(sim))
            t2.add(a)
            c = match(s3, xyz, 0.7, 0.3, t3)
            wd = wall_dist(b, dimg, planes) if use_wall_far else None
            if wd is not None:
                s3[c]["wd"].append(wd)
            t3.add(c)
            i2.append(a)
            i3.append(c)
        n = len(regs)
        seen.extend(float(sim) for sim, _ in regs)
        base = float(np.median(seen))
        memsim = [regs[j][0] if i2[j] is None else float(np.mean(s2[i2[j]]["sims"])) for j in range(n)]
        if use_rank_mem:                            # 7. 2D 기억 평균 순위
            ms = np.array([regs[j][0] if i2[j] is None else np.mean(s2[i2[j]]["sims"]) for j in range(n)])
            rm = np.argsort(np.argsort(-ms))
            order = sorted(range(n), key=lambda j: (j + rm[j], j))
        else:
            order = list(range(n))
        far = [False] * n
        if use_wall_far:                            # 9. 벽에서 먼 자리는 뒤로
            far = [i3[j] is not None and len(s3[i3[j]]["wd"]) > 0 and np.median(s3[i3[j]]["wd"]) > far_m
                   for j in range(n)]
            order = [j for j in order if not far[j]] + [j for j in order if far[j]]
        picks, skipped = [], []
        for j in order:                             # 8·10. 확인한 자리 건너뛰고 K개
            if len(picks) >= K:
                break
            if use_skip and i3[j] is not None and i3[j] in checked:
                skipped.append(j)
                continue
            if use_gate and memsim[j] < base:
                continue
            picks.append(j)
        for j in picks:
            if i3[j] is not None:
                checked.add(i3[j])
        out[k] = dict(frame=v["frame"], regions=[regs[j] for j in order],
                      picks=[regs[j][1] for j in picks], skipped=[regs[j][1] for j in skipped],
                      far=[regs[j][1] for j in order if far[j]])
    return out


def evaluate(res, gt, n_obj):
    found, n_check, hit_rank = set(), 0, {1: 0, 2: 0, 3: 0}
    for k, v in res.items():
        d = gt[k]
        seg = k.rsplit("_", 1)[0]
        n_check += len(v["picks"])
        for (c, *g), o in zip(d["items"], d["object_id"]):
            if c != "plaque":
                continue
            if any(M.inside(g, b) for b in v["picks"]):
                found.add(f"{seg}#{o}")
            for N in hit_rank:
                hit_rank[N] += any(M.inside(g, b) for _, b in v["regions"][:N])
    return len(found), n_check, hit_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="4F")
    ap.add_argument("--tag", default=None, help="입력(없으면 층별 기본)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.tag or {"4F": "04_SigLIP재정렬/3_겹침합치기_채택", "3F": "07_3층/2_SigLIP재정렬_설명그림"}[args.set]
    B = json.load(open(M.R / tag / "regions.json"))
    segs = sign_segsets.get(args.set)
    gt = {}
    for sg in segs:
        for l in open(sg["gt"]):
            d = json.loads(l)
            gt[f"{sg['name']}_{float(d['t'])}"] = d
    n_inst = sum(c == "plaque" for d in gt.values() for c, *_ in d["items"])
    n_obj = len({f"{k.rsplit('_', 1)[0]}#{o}" for k, d in gt.items()
                 for (c, *_), o in zip(d["items"], d["object_id"]) if c == "plaque"})
    configs = [("기억 없음, 매번 상위 3", 3, False, False, False),
               ("영상 방식(2D 평균), 매번 상위 3", 3, True, False, False),
               ("+ 3D 건너뜀, 상위 3", 3, True, False, True),
               ("+ 3D 건너뜀, 상위 2", 2, True, False, True),
               ("+ 벽에서 먼 자리 뒤로, 상위 3", 3, True, True, True),
               ("+ 벽에서 먼 자리 뒤로, 상위 2 (조합)", 2, True, True, True),
               ("+ 벽에서 먼 자리 뒤로, 상위 1", 1, True, True, True),
               ("+ 차단, 상위 3", 3, True, True, True, True),
               ("+ 차단, 상위 2", 2, True, True, True, True),
               ("최종: 2D기억+차단, 최대3 (건너뜀 없음)", 3, True, False, False, True),
               ("최종 + 벽거리, 최대3 (건너뜀 없음)", 3, True, True, False, True),
               ("벽거리 없이 + 차단, 상위 3", 3, True, False, True, True),
               ("벽거리 없이 + 차단, 상위 2", 2, True, False, True, True)]
    lines = [f"층 {args.set}, 입력 {tag}, 번호판 관측 {n_inst}개 · 실제 {n_obj}개, 장면 {len(gt)}장",
             f"{'방식':34s} | 찾은 번호판 | 확인 횟수 | 확인당 | 순위 1위/2위/3위 안 관측"]
    keep = None
    for lab, K, rm, wf, sk, *g in configs:
        res = {}
        for sg in segs:
            res.update(run(sg, B, K, rm, wf, sk, use_gate=bool(g and g[0])))
        f, c, hr = evaluate(res, gt, n_obj)
        lines.append(f"{lab:34s} | {f:3d}/{n_obj:<3d}    | {c:4d}회   | {f / max(c, 1):.2f}  | "
                     f"{hr[1]}/{hr[2]}/{hr[3]} (/{n_inst})")
        if "(조합)" in lab:
            keep = res
        if lab == "+ 차단, 상위 3":
            keep_gate = res
    print("\n".join(lines))
    out = M.R / (args.out or f"08_조합/{args.set}_상위2_건너뜀_벽거리")
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_output.txt").write_text("\n".join(lines) + "\n")
    json.dump(keep, open(out / "regions.json", "w"))
    g_out = M.R / f"08_조합/{args.set}_상위3_건너뜀_벽거리_차단"
    g_out.mkdir(parents=True, exist_ok=True)
    json.dump(keep_gate, open(g_out / "regions.json", "w"))
    (g_out / "eval_output.txt").write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
