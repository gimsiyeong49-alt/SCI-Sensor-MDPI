#!/usr/bin/env python3
"""매초 멈추는 대신 '표지 같은 자리가 처음 D m 안에 들어올 때'만 멈추는 이벤트 규칙을 시뮬레이션한다.

걷는 동안(1 Hz로 들어오는 장면마다): 벽 구간을 2D 벽 자리 기억에 쌓고 SigLIP 점수를 모은다.
표지 같음 = 그 자리의 기억 평균 SigLIP ≥ 지금까지 본 모든 구간 점수의 중앙값(과거만 사용).
이벤트 = 표지 같은 자리 중 아직 이벤트를 안 낸 자리가 D m 안(구간 가운데 깊이)에 처음 들어오면 멈춘다.
멈추면: 그 장면의 구간을 기억 평균 순위로 정렬하고, 표지 같은 것만 최대 K개 확인한다.
        그 장면에서 D m 안의 표지 같은 자리는 전부 '이벤트 냄'으로 표시해 곧바로 다시 멈추지 않게 한다
        (확인 후보에서 빼는 것이 아니라 멈출지 말지만 정한다).
gap 초: 한 번 멈춘 뒤 다음 이벤트까지 최소 간격(멈춤·조준 시간 가정). 0이면 제한 없음.
abs_thr : '표지 같음'을 SigLIP 절대값으로 판단(사람이 정한 값).
quant   : '표지 같음'을 지금까지 본 모든 구간 점수의 상위 (1-quant) 안에 드는지로 판단(자동, 과거만 사용).
          번호판은 후보 가운데 드물다는 성질만 쓴다. 층·건물마다 점수 높이가 달라도 따라간다.
          처음 warm 개 점수가 모이기 전에는 멈추지 않는다.
track   : 1초 장면 사이를 0.2초 간격 프레임의 광학 흐름으로 이어 같은 구간을 같은 트랙으로 묶고,
          트랙 하나당 한 번만 멈춘다(벽 자리 기억이 쪼개져 같은 물체에 반복해서 멈추는 것을 막는다).
정답은 채점에만 쓴다.
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
from combined_policy import match  # noqa: E402
from geometry_filter import vanishing_point  # noqa: E402

INPUT = {"4F": "04_SigLIP재정렬/3_겹침합치기_채택", "3F": "07_3층/2_SigLIP재정렬_설명그림",
         "3F_eval": "10_고정평가_3층_330_360/2_SigLIP재정렬"}


def _index(root):
    import csv
    rows = list(csv.DictReader(open(Path(root) / "index.csv")))
    col = "header_relative_sec" if "header_relative_sec" in rows[0] else "bag_relative_sec"
    return np.array([float(r[col]) for r in rows]), [r["filename"] for r in rows]


def _propagate(root, times, files, t_a, t_b, pts, step=0.2):
    """t_a 장면의 점들을 step 초 간격 프레임으로 t_b 까지 광학 흐름으로 옮긴다. 잃으면 None."""
    if len(pts) == 0:
        return []
    cur = np.array(pts, np.float32).reshape(-1, 1, 2)
    ok = np.ones(len(pts), bool)
    tt = list(np.arange(t_a, t_b, step)) + [t_b]
    prev = None
    for t in tt:
        i = int(np.argmin(abs(times - t)))
        g = cv2.cvtColor(cv2.imread(str(Path(root) / files[i])), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, cur, None, winSize=(21, 21), maxLevel=3)
            ok &= st.reshape(-1).astype(bool)
            cur = nxt
        prev = g
    return [tuple(p) if o else None for p, o in zip(cur.reshape(-1, 2), ok)]


def prepare(seg, B, quant=None, warm=50, track=False):
    """장면마다 구간의 기억 자리 번호·깊이·기억 평균 점수·기준값을 미리 계산한다(과거만 사용)."""
    odom = M.Odom(seg["odom"], seg["t0"])
    if track:
        times, files = _index(seg["root"])
        next_tid, prev_scene = 0, None
    dz = np.load(seg["depth"])
    keys = sorted([k for k in B if k.startswith(seg["name"])], key=lambda k: float(k.split("_")[-1]))
    spots, seen, scenes = [], [], []
    for k in keys:
        t = float(k.split("_")[-1])
        v = B[k]
        regs = v["regions"]
        dimg = dz[f"{t:.1f}"]
        vp = vanishing_point(cv2.imread(str(seg["root"] / v["frame"])))
        pose = odom(t)
        ids, zs, taken = [], [], set()
        for sim, b in regs:
            xyz, z = M.region_world(b, dimg, vp, pose)
            if xyz is None:
                ids.append(None)
                zs.append(None)
                continue
            a = match(spots, xyz, 0.7, None, taken)
            spots[a]["sims"].append(float(sim))
            taken.add(a)
            ids.append(a)
            zs.append(z)
        seen.extend(float(s) for s, _ in regs)
        if quant is None:
            base = float(np.median(seen))
        else:
            base = float(np.quantile(seen, quant)) if len(seen) >= warm else float("inf")
        tids = [None] * len(regs)
        if track:
            if prev_scene is not None and prev_scene["regs"]:
                cen = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in prev_scene["regs"]]
                moved = _propagate(seg["root"], times, files, prev_scene["t"], t, cen)
                for pj, p in enumerate(moved):
                    if p is None:
                        continue
                    for j, (_, b) in enumerate(regs):
                        if tids[j] is None and b[0] - 10 <= p[0] <= b[2] + 10 and b[1] - 10 <= p[1] <= b[3] + 10:
                            tids[j] = prev_scene["tids"][pj]
                            break
            for j in range(len(regs)):
                if tids[j] is None:
                    tids[j] = next_tid
                    next_tid += 1
        ms = [regs[j][0] if ids[j] is None else float(np.mean(spots[ids[j]]["sims"])) for j in range(len(regs))]
        rm = np.argsort(np.argsort(-np.array(ms))) if regs else []
        order = sorted(range(len(regs)), key=lambda j: (j + rm[j], j))
        scenes.append(dict(k=k, t=t, regs=regs, ids=ids, zs=zs, ms=ms, base=base, order=order, tids=tids))
        if track:
            prev_scene = scenes[-1]
    return scenes


def simulate(scenes, D, K=3, gap=0.0, every=False, abs_thr=None, pick_relative=False):
    """abs_thr: 주면 '표지 같음'을 상대 기준(지금까지 중앙값) 대신 SigLIP 절대값 기준으로 판단한다."""
    fired, fired_t, events, last = set(), set(), [], -1e9
    for s in scenes:
        base = s["base"] if abs_thr is None else abs_thr
        like = [j for j in range(len(s["regs"])) if s["ms"][j] >= base]
        near = [j for j in like if s["zs"][j] is not None and s["zs"][j] <= D]
        new = [j for j in near if s["ids"][j] not in fired
               and (s.get("tids") is None or s["tids"][j] is None or s["tids"][j] not in fired_t)]
        trig = every or (bool(new) and s["t"] - last >= gap)
        if not trig:
            continue
        last = s["t"]
        if abs_thr is not None and pick_relative:       # 멈출지는 절대 기준, 확인할 구간은 기존(상대) 기준으로
            like_pick = [j for j in range(len(s["regs"])) if s["ms"][j] >= s["base"]]
        else:
            like_pick = like
        picks = [s["regs"][j][1] for j in s["order"] if j in like_pick][:K]
        events.append(dict(k=s["k"], t=s["t"], picks=picks))
        for j in near:
            if s["ids"][j] is not None:
                fired.add(s["ids"][j])
            if s.get("tids") and s["tids"][j] is not None:
                fired_t.add(s["tids"][j])
    return events


def score(events, gt, name):
    found, inst, checks = set(), 0, 0
    for e in events:
        d = gt[e["k"]]
        checks += len(e["picks"])
        for (c, *g), o in zip(d["items"], d["object_id"]):
            if c == "plaque" and any(M.inside(g, b) for b in e["picks"]):
                found.add(f"{name}#{o}")
                inst += 1
    return found, inst, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="4F")
    ap.add_argument("--gap", type=float, default=0.0)
    args = ap.parse_args()
    B = json.load(open(M.R / INPUT[args.set] / "regions.json"))
    segs = sign_segsets.get(args.set)
    gt, n_obj = {}, 0
    for sg in segs:
        for l in open(sg["gt"]):
            d = json.loads(l)
            gt[f"{sg['name']}_{float(d['t'])}"] = d
    n_obj = len({f"{k.rsplit('_', 1)[0]}#{o}" for k, d in gt.items()
                 for (c, *_), o in zip(d["items"], d["object_id"]) if c == "plaque"})
    prep = {sg["name"]: prepare(sg, B) for sg in segs}
    secs = sum(len(v) for v in prep.values())
    lines = [f"층 {args.set}: 구간 {len(segs)}개, 장면 {secs}장(1초 간격), 실제 번호판 {n_obj}개, 이벤트 최소 간격 {args.gap:g}초",
             f"{'규칙':22s} | 멈춤 횟수 | 확인 횟수 | 찾은 번호판 | 확인당"]
    rows = []
    for lab, D, every in [("매초 멈춤 (지금 시험)", 7.0, True)] + [(f"처음 {D:g} m 안에 들면", D, False) for D in (4, 5, 6, 7)]:
        f_all, stops, checks = set(), 0, 0
        for sg in segs:
            ev = simulate(prep[sg["name"]], D, gap=args.gap, every=every)
            f, _, c = score(ev, gt, sg["name"])
            f_all |= f
            stops += len(ev)
            checks += c
        rows.append(dict(rule=lab, D=D, stops=stops, checks=checks, found=len(f_all)))
        lines.append(f"{lab:22s} | {stops:4d}번    | {checks:4d}회   | {len(f_all):3d}/{n_obj:<3d}    | {len(f_all) / max(checks, 1):.2f}")
    print("\n".join(lines))
    out = M.R / "11_거리이벤트" / (args.set + (f"_간격{args.gap:g}초" if args.gap else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_output.txt").write_text("\n".join(lines) + "\n")
    json.dump(rows, open(out / "results.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
