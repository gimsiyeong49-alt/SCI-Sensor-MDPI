#!/usr/bin/env python3
"""정답 없는 전체 주행에 지금 파이프라인을 1초 간격으로 돌리고, 멈춤 이벤트가 언제 나는지 계산한다.

처리(기존 스크립트의 함수를 그대로 씀):
  벽 구간(wall_region_events.regions, 7 m·바닥 제외, 조각 1개 점수) → SigLIP 구간 재정렬(겹침 합치기)
  → 2D 벽 자리 기억 + 이벤트 규칙(event_trigger_sim.prepare / simulate).
이벤트 규칙: 표지 같은 자리(기억 평균 ≥ 지금까지 중앙값)가 처음 D m 안에 들어오면 멈춘다. --gap 초 최소 간격.
출력: regions.json(초마다 재정렬된 구간), events_*.json(멈춘 시각과 확인 구간).
단계(--stage): regions(시스템 파이썬, OpenCV 4.5) → siglip(vlm_env, SigLIP) → events(시스템 파이썬).
지금까지 결과와 같은 환경 조합이다. vlm_env의 OpenCV 5.0은 직선 검출 결과 모양이 달라 소실점 함수가 깨진다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import event_trigger_sim as E  # noqa: E402
import region_memory as M  # noqa: E402
from event_criteria import salience  # noqa: E402
from geometry_filter import vanishing_point  # noqa: E402
from wall_region_events import depth_ok, regions  # noqa: E402

THR = json.load(open("reports/_00_SIGN_PIPELINE_20260920/30_관측이벤트/grid.json"))["threshold"]


def nms(order, regs):
    kept = []
    for i in order:
        b = regs[i]
        dup = False
        for j in kept:
            c = regs[j]
            ix = max(0, min(b[2], c[2]) - max(b[0], c[0]))
            iy = max(0, min(b[3], c[3]) - max(b[1], c[1]))
            amin = min((b[2] - b[0]) * (b[3] - b[1]), (c[2] - c[0]) * (c[3] - c[1]))
            dup |= ix * iy >= 0.5 * amin
        if not dup:
            kept.append(i)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="12_4층전체주행")
    ap.add_argument("--root", default="/data2/kci_datasets/4F_RGB_FULL_V0_1")
    ap.add_argument("--depth", default="/data3/kci_datasets/depth_cache_sign_eval/4F_full.npz")
    ap.add_argument("--odom", default="/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv")
    ap.add_argument("--t0", type=float, default=1783340324.059948032)
    ap.add_argument("--stage", required=True, choices=["regions", "siglip", "events"])
    args = ap.parse_args()
    out = M.R / args.run_dir
    root = Path(args.root)
    name = "4F_full"
    if args.stage == "regions":
        stage_regions(args, out, root, name)
    elif args.stage == "siglip":
        stage_siglip(out, root)
    else:
        stage_events(args, out, root, name)


def stage_regions(args, out, root, name):
    dz = np.load(args.depth)
    raw = {}
    for line in open(out / "prior/proposals.jsonl"):
        d = json.loads(line)
        t = float(d["t"])
        if f"{t:.1f}" not in dz:
            continue
        bgr = cv2.imread(str(root / d["frame"]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        recs = d["records"]
        sal = [salience(gray, grad, r["box"]) for r in recs]
        dimg = dz[f"{t:.1f}"]
        vp = vanishing_point(bgr)
        keep = [depth_ok(dimg, r["box"], vp, 7.0)[0] for r in recs]
        regs = [b for _, b in regions(recs, sal, THR, keep=keep, topk=1)[:8]]
        raw[f"{name}_{t}"] = dict(frame=d["frame"], regions=regs)
    json.dump(raw, open(out / "regions_raw.json", "w"))
    print("벽 구간", len(raw))


def stage_siglip(out, root):
    from region_siglip_rerank import RegionSiglip
    vlm = RegionSiglip()
    vlm.score(np.zeros((480, 640, 3), np.uint8), [[0, 0, 200, 200]])
    raw = json.load(open(out / "regions_raw.json"))
    B = {}
    for k, v in raw.items():
        regs = v["regions"]
        if not regs:
            B[k] = dict(frame=v["frame"], regions=[])
            continue
        sim = vlm.score(cv2.imread(str(root / v["frame"])), regs, "tile")
        r_sig = np.argsort(np.argsort(-sim))
        order = nms(sorted(range(len(regs)), key=lambda i: (i + r_sig[i], i)), regs)
        B[k] = dict(frame=v["frame"], regions=[[float(sim[i]), regs[i]] for i in order])
    json.dump(B, open(out / "regions.json", "w"))
    print("SigLIP 재정렬", len(B))


def stage_events(args, out, root, name):
    B = json.load(open(out / "regions.json"))
    seg = dict(name=name, odom=args.odom, t0=args.t0, depth=args.depth, root=root)
    prep = E.prepare(seg, B)
    dur = (max(s["t"] for s in prep) - min(s["t"] for s in prep)) / 60
    lines = [f"4층 전체 주행 {len(prep)}초({dur:.1f}분), 1초 간격 처리"]
    for D, gap, thr in ((5.0, 0.0, None), (5.0, 3.0, None), (5.0, 5.0, None), (7.0, 0.0, None),
                        (5.0, 0.0, 0.06), (5.0, 3.0, 0.06), (5.0, 0.0, 0.05), (5.0, 3.0, 0.05)):
        ev = E.simulate(prep, D, gap=gap, abs_thr=thr)
        tag = f"events_D{D:g}_gap{gap:g}" + (f"_abs{thr:g}" if thr else "")
        json.dump(ev, open(out / f"{tag}.json", "w"))
        in_pl = sum(140 <= e["t"] <= 262 for e in ev)
        lines.append(f"  처음 {D:g} m 안, 최소 간격 {gap:g}초, 기준 {'지금까지 중앙값' if thr is None else f'절대 {thr:g}'}: "
                     f"멈춤 {len(ev)}번 (분당 {len(ev) / dur:.1f}번, 번호판 복도 140~262초 안 {in_pl}번 · 밖 {len(ev) - in_pl}번)")
    print("\n".join(lines))
    (out / "eval_output.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
