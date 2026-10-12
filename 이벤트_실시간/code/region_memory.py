#!/usr/bin/env python3
"""벽 구간을 벽 위 좌표로 기억하고, 다가가는 동안 모인 증거로 순서를 다시 매긴다.

과거와 현재 장면만 쓴다(미래 없음, 딜레이 없음). 장면마다 하는 일은 구간 8개를 벽 좌표로 옮겨
기억과 맞추는 것뿐이다.
  벽 좌표 : 구간 가운데 절반의 깊이 중앙값 → 카메라 기울기(소실점) 보정 → 수평 앞/옆 거리
            → FAST-LIO 자세(odom_link.csv, 이 데이터셋 규약 월드 = R(yaw+228°)·p + t)로 월드 x,y.
            같은 번호판이 여러 초에 걸쳐 0.13 m 안에 모이는 것을 확인했다(정답은 이 확인에만 씀).
  기억    : 0.7 m 안의 기존 자리와 합친다. 자리마다 (시각, 순위, SigLIP 값)을 쌓는다.
  순서    : A persist — 최근 5초 동안 그 자리가 상위 3에 든 횟수(많을수록 앞), 같으면 현재 순위
            B memsim  — 그 자리에서 지금까지 나온 SigLIP 값 평균의 순위와 현재 구간 순위의 평균
            C memmax  — 평균 대신 지금까지 가장 높았던 SigLIP 값(제일 좋았을 때)
            D bestrank— 그 자리가 지금까지 받은 가장 좋은 순위, 같으면 현재 순위
            E memsimz — B에서 4~7 m 거리에서 본 값은 1, 그 밖은 0.3 가중 평균
                        (4~7 m는 번호판 1순위 적중이 높던 거리. 이 개발 구간 정답 분석에서 나온 값이라
                         고정 평가 전까지는 과적합 가능성이 있다)
정답은 채점에만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from event_criteria import SEGS, D  # noqa: E402
from geometry_filter import vanishing_point  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
ODOM = "/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv"
T0 = 1783340324.059948032                      # RGB 첫 장 header 시각 (index.csv header_relative_sec 기준)
YAW_OFF = np.deg2rad(228.0)
FX, FY, CX, CY = 606.80, 606.75, 310.49, 240.52
DEPTH = {"4F_148_178": "/data3/kci_datasets/depth_cache_sign_eval/4F_148.npz",
         "4F_228_258": "/data3/kci_datasets/depth_cache_sign_eval/4F_228.npz"}


LAST_SPOTS = {}                                 # 구간 이름 → 끝났을 때의 기억(그림용)


class Odom:
    def __init__(self, path=ODOM, t0=T0):
        rows = list(csv.DictReader(open(path)))
        self.t = np.array([float(r["t_sec"]) - t0 for r in rows])
        self.x = np.array([float(r["fastlio_x"]) for r in rows])
        self.y = np.array([float(r["fastlio_y"]) for r in rows])
        self.yaw = np.unwrap(np.array([float(r["fastlio_yaw_rad"]) for r in rows]))

    def __call__(self, t):
        return (np.interp(t, self.t, self.x), np.interp(t, self.t, self.y),
                np.interp(t, self.t, self.yaw) + YAW_OFF)


def region_world(box, dimg, vp, pose):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    a0, b0, a1, b1 = int(x0 + w / 4), int(y0 + h / 4), int(x1 - w / 4) + 1, int(y1 - h / 4) + 1
    p = dimg[max(0, b0):b1, max(0, a0):a1]
    v = p[p > 0]
    if len(v) < 5:
        return None, None
    z = float(np.median(v)) / 1000.0
    xc = ((x0 + x1) / 2 - CX) / FX * z
    yc = ((y0 + y1) / 2 - CY) / FY * z
    th = np.arctan2(vp[1] - CY, FY) if vp is not None else 0.0
    fwd, left = z * np.cos(th) + yc * np.sin(th), -xc
    up = -yc * np.cos(th) + z * np.sin(th)          # 카메라 기준 높이(기울기 보정)
    X, Y, a = pose
    return np.array([X + np.cos(a) * fwd - np.sin(a) * left, Y + np.sin(a) * fwd + np.cos(a) * left, up]), z


def inside(g, b):
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def run_segment(seg, B, odom, mode, radius=0.7, window=5.0, topk_mem=3, dim=2, vtol=0.3):
    """dim=2: 바닥 평면 x,y 로만 같은 자리 판단(높이 무시).
    dim=3: 수평 radius 안이면서 높이 차가 vtol 안일 때만 같은 자리(번호판·카드리더·전등을 높이로 가름)."""
    name = seg["name"]
    dz = np.load(seg["depth"])
    root = seg["root"]
    keys = sorted([k for k in B if k.startswith(name)], key=lambda k: float(k.split("_")[-1]))
    spots = []                                  # {"xy": array, "obs": [(t, rank, sim)]}
    out = {}
    for k in keys:
        t = float(k.split("_")[-1])
        v = B[k]
        regs = v["regions"]
        dimg = dz[f"{t:.1f}"]
        vp = vanishing_point(cv2.imread(str(root / v["frame"])))
        pose = odom(t)
        sid = []
        for r, (sim, b) in enumerate(regs):     # 1) 이번 장면 구간을 기억과 맞춘다
            xy, zz = region_world(b, dimg, vp, pose)
            if xy is None:
                sid.append(None)
                continue
            best, bd = None, radius
            for i, s in enumerate(spots):
                d = np.linalg.norm(s["xy"][:2] - xy[:2])
                if dim == 3 and abs(s["xy"][2] - xy[2]) > vtol:
                    continue
                if d < bd and i not in sid:
                    best, bd = i, d
            if best is None:
                spots.append({"xy": xy, "obs": []})
                best = len(spots) - 1
            s = spots[best]
            n = len(s["obs"])
            s["xy"] = (s["xy"] * n + xy) / (n + 1)
            s["obs"].append((t, r, float(sim), zz, [float(x) for x in b]))   # 마지막은 그림용 구간 상자
            sid.append(best)
        n = len(regs)                           # 2) 기억으로 순서를 다시 매긴다(과거+현재만)
        if mode == "persist":
            per = [0 if i is None else sum(1 for tt, rr, *_ in spots[i]["obs"] if t - tt <= window and rr < topk_mem)
                   for i in sid]
            order = sorted(range(n), key=lambda j: (-per[j], j))
        elif mode in ("memsim", "memmax"):
            agg = np.mean if mode == "memsim" else np.max
            ms = np.array([regs[j][0] if sid[j] is None else agg([o[2] for o in spots[sid[j]]["obs"]])
                           for j in range(n)])
            r_mem = np.argsort(np.argsort(-ms))
            order = sorted(range(n), key=lambda j: (j + r_mem[j], j))
        elif mode == "memsimz":
            def wmean(obs):
                w = np.array([1.0 if 4.0 <= o[3] <= 7.0 else 0.3 for o in obs])
                return float(np.sum(w * np.array([o[2] for o in obs])) / np.sum(w))
            ms = np.array([regs[j][0] if sid[j] is None else wmean(spots[sid[j]]["obs"]) for j in range(n)])
            r_mem = np.argsort(np.argsort(-ms))
            order = sorted(range(n), key=lambda j: (j + r_mem[j], j))
        elif mode == "bestrank":
            br = [j if sid[j] is None else min(o[1] for o in spots[sid[j]]["obs"]) for j in range(n)]
            order = sorted(range(n), key=lambda j: (br[j], j))
        else:
            order = list(range(n))
        deps = [region_world(b, dimg, vp, pose)[1] for _, b in regs]
        out[k] = dict(v, regions=[regs[j] for j in order], spot=[sid[j] for j in order],
                      depth=[deps[j] for j in order])
    LAST_SPOTS[name] = spots
    return out


def simulate_checks(src, gt, name, per_stop, skip_checked, zwin=None):
    """멈출 때마다 고화질 카메라가 구간 per_stop개를 확인한다고 보고, 찾은 번호판(물체)과 확인 횟수를 센다.
    skip_checked : 이미 확인한 벽 자리는 건너뛴다(기억 사용).
    zwin         : (zmin, zmax) 이 거리의 구간만 확인한다. 밖이면 다음 정지로 미룬다."""
    checked, found, n_check = set(), set(), 0
    for k in sorted([k for k in src if k.startswith(name)], key=lambda k: float(k.split("_")[-1])):
        v, d = src[k], gt[k]
        picks = []
        for (sim, b), sp, z in zip(v["regions"], v["spot"], v["depth"]):
            if len(picks) >= per_stop:
                break
            if skip_checked and sp is not None and sp in checked:
                continue
            if zwin and (z is None or not zwin[0] <= z <= zwin[1]):
                continue
            picks.append(b)
            if sp is not None:
                checked.add(sp)
        n_check += len(picks)
        for (c, *g), o in zip(d["items"], d["object_id"]):
            if c == "plaque" and any(inside(g, b) for b in picks):
                found.add(f"{name}#{o}")
    return found, n_check


def score(src, gt, N):
    hi, ho, emp, tot = 0, set(), 0, 0
    for k, v in src.items():
        top = [b for _, b in v["regions"][:N]]
        d = gt[k]
        seg = k.rsplit("_", 1)[0]
        for (c, *g), o in zip(d["items"], d["object_id"]):
            if c == "plaque" and any(inside(g, b) for b in top):
                hi += 1
                ho.add(f"{seg}#{o}")
        signs = [g for c, *g in d["items"] if c in ("plaque", "sign")]
        for b in top:
            tot += 1
            emp += not any(inside(g, b) for g in signs)
    return hi, len(ho), emp, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="04_SigLIP재정렬/3_겹침합치기_채택")
    ap.add_argument("--out", default=None, help="결과 폴더(31_벽구간이벤트 기준). 없으면 입력 이름 뒤에 방식을 붙인다")
    ap.add_argument("--set", default="4F", help="층 선택(sign_segsets)")
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3], help="기억 차원: 2=바닥 평면, 3=높이 포함")
    ap.add_argument("--aim", action="store_true",
                    help="구간 대신 구간 안 최고 점수 조각(조준점)으로 기억·채점·확인한다")
    ap.add_argument("--mode", default="memsim", choices=["none", "persist", "memsim", "memmax", "bestrank", "memsimz"])
    args = ap.parse_args()
    B = json.load(open(R / args.tag / "regions.json"))
    if args.aim:
        B = {k: dict(v, regions=[[sim, t] for (sim, _), t in zip(v["regions"], v["tiles"])]) for k, v in B.items()}
        area = [(t[2] - t[0]) * (t[3] - t[1]) for v in B.values() for _, t in v["regions"]]
        print(f"조준 조각 면적 중앙 {np.median(area):.0f} px² (구간 대신)")
    import sign_segsets
    segs = sign_segsets.get(args.set)
    gt = {}
    for sg in segs:
        for l in open(sg["gt"]):
            d = json.loads(l)
            gt[f"{sg['name']}_{float(d['t'])}"] = d
    new = {}
    for sg in segs:
        new.update(run_segment(sg, B, Odom(sg["odom"], sg["t0"]), args.mode, dim=args.dim))
    n_inst = sum(c == "plaque" for d in gt.values() for c, *_ in d["items"])
    n_obj = len({f"{k.rsplit('_', 1)[0]}#{o}" for k, d in gt.items()
                 for (c, *_), o in zip(d["items"], d["object_id"]) if c == "plaque"})
    lines = [f"층 {args.set}, 기억 {args.dim}D, 입력 {args.tag}, 기억 순서 {args.mode}, 단위 {'조준 조각' if args.aim else '구간'}"]
    for N in (1, 2, 3, 5):
        hi, ho, emp, tot = score(new, gt, N)
        lines.append(f"N={N}: 번호판 관측 {hi}/{n_inst}, 물체 {ho}/{n_obj}, 헛구간 {emp}/{tot}")
    lines.append("")
    lines.append("확인 정책 시뮬레이션 (매 초 멈춘다고 가정, 두 구간 합)")
    for per_stop, skip, zwin, lab in ((3, False, None, "매번 상위 3 (기억 없음)"),
                                      (3, True, None, "상위 3, 확인한 자리 건너뜀"),
                                      (1, True, None, "상위 1, 확인한 자리 건너뜀"),
                                      (2, True, None, "상위 2, 확인한 자리 건너뜀"),
                                      (2, True, (4.0, 7.0), "상위 2, 건너뜀, 4~7 m 구간만")):
        f_all, c_all = set(), 0
        for name in [sg["name"] for sg in segs]:
            f, c = simulate_checks(new, gt, name, per_stop, skip, zwin)
            f_all |= f
            c_all += c
        lines.append(f"  {lab:24s}: 번호판 {len(f_all)}/{n_obj}, 확인 {c_all}회, 확인당 {len(f_all) / max(c_all, 1):.2f}개")
    print("\n".join(lines))
    out = R / args.out if args.out else R / (f"{args.tag}_mem_{args.mode}" + ("_aim" if args.aim else "") + ("_3d" if args.dim == 3 else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_output.txt").write_text("\n".join(lines) + "\n")
    json.dump(new, open(out / "regions.json", "w"))


if __name__ == "__main__":
    main()
