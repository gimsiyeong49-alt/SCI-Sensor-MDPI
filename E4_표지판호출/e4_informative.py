#!/usr/bin/env python3
"""E4 호출 정책 P_INFORMATIVE: 멈춤(이벤트)마다 '읽으면 지금 후보들을 가를 수 있나'를 판단한다(2026-10-07, 실험 지침서 14.5).
판독 정답 문자는 쓰지 않는다. 그 순간 알 수 있는 것(위치추정 후보, 의미지도, 이벤트의 3D 위치)만 쓴다.

멈춤 시각 t 의 위치추정 후보 중 1위 점수 −1.5 안의 후보(near)마다, 이벤트 위치(오돔 xy)를 그 후보였다면 지도 어디에 찍히나로 옮긴다
(obs_feedback.py 와 같은 식: 지도점 = 후보 픽셀 + ppm·A(φ)·(이벤트 xy − 지금 오돔 xy)). 그리고
  호실 표지판: 찍힌 자리에서 가장 가까운 호실(경계 plaque_tol + 거리비례 허용 안, 복도 corr_tol 안)의 id, 없으면 '없음'
  계단: 찍힌 자리가 계단 칸 tol 안인지(예/아니오)
near 후보들의 답이 둘 이상으로 갈리면(그리고 호실은 적어도 하나가 호실을 가리키면) informative.
near 후보가 하나뿐이면(이미 하나로 좁혀짐) informative 아님.
정보량(info_bits): near 후보 점수를 가중치(exp(점수 − 1위))로 답별 비중의 엔트로피. 답이 고르게 갈릴수록 크다(정책 기준값은 파일럿에서 정함).
참고: 녹화 실행의 멈춤 목록에서 나중에 거르는 근사다(거른 멈춤 대신 다른 곳에 멈췄을 가능성은 반영하지 않음).

    python3 scripts/e4_informative.py --loc <loc> --run <run> --sem <의미지도> --base <base_map.png> --out <events_informative.json>
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from obs_feedback import A, poly_mask, FX, CX, YAW_OFF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--sem", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--near", type=float, default=1.5, help="1위 점수에서 이만큼 안의 후보만 본다(게이트 spread 정의와 같음)")
    ap.add_argument("--plaque-tol", type=float, default=1.0); ap.add_argument("--corr-tol", type=float, default=1.5)
    ap.add_argument("--tol-per-m", type=float, default=0.08); ap.add_argument("--fac-tol", type=float, default=2.0)
    a = ap.parse_args()
    im = cv2.imread(a.base, cv2.IMREAD_GRAYSCALE); H, W = im.shape
    sem = json.load(open(a.sem))
    rooms = [s for s in sem["spaces"] if s.get("type") == "room" and s.get("polygon_px")]
    lab = np.zeros((H, W), np.int32)                      # 호실 경계 근처를 가장 가까운 호실 번호로 칠한다
    edge_any = np.zeros((H, W), bool)
    for i, r in enumerate(rooms, 1):
        m = poly_mask((H, W), [r["polygon_px"]])
        e = m & cv2.dilate((~m).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        lab[e] = i; edge_any |= e
    DT, (iy, ix) = distance_transform_edt(~edge_any, return_indices=True)
    nearest_room = lab[iy, ix]
    corr = poly_mask((H, W), [c.get("polygon_px") for c in sem.get("corridors", [])])
    DT_corr = distance_transform_edt(~corr) if corr.any() else np.zeros((H, W))
    stair = poly_mask((H, W), [s.get("polygon_px") for s in sem["spaces"] if s.get("type") == "stair"])
    DT_stair = distance_transform_edt(~stair) if stair.any() else None

    def look(img, p, cval):
        x, y = int(round(p[0])), int(round(p[1]))
        return img[y, x] if 0 <= x < W and 0 <= y < H else cval

    tk = {}
    for r in csv.DictReader(open(Path(a.loc) / "topk_series.csv")):
        tk.setdefault(float(r["relative_sec"]), []).append(r)
    tl = {float(r["relative_sec"]): r for r in csv.DictReader(open(Path(a.loc) / "timeline.csv"))}
    ts = np.array(sorted(tk))

    def judge(t, kind, xy, z):
        k = float(ts[np.argmin(np.abs(ts - t))])
        r = tl[k]; ppm = float(r.get("ppm_est") or 0) or 30.0
        o_now = np.array([float(r["odom_x_m"]), float(r["odom_y_m"])])
        c = sorted(tk[k], key=lambda x: -float(x["score"]))
        best = float(c[0]["score"])
        near = [x for x in c if float(x["score"]) >= best - a.near]
        grow = a.tol_per_m * float(np.linalg.norm(xy - o_now))
        answers = []
        for x in near:
            px = np.array([float(x["x_px"]), float(x["y_px"])]); M = A(math.radians(float(x["phi_deg"]))) * ppm
            mp = px + M @ (xy - o_now)
            if kind == "번호판":
                ok = look(DT, mp, 1e9) / ppm <= a.plaque_tol + grow and look(DT_corr, mp, 1e9) / ppm <= a.corr_tol + grow
                answers.append(int(look(nearest_room, mp, 0)) if ok else 0)
            else:
                answers.append(bool(DT_stair is not None and look(DT_stair, mp, 1e9) / ppm <= a.fac_tol + 0.2 * (z or 0) + grow))
        distinct = set(answers)
        # 정보량: near 후보 점수를 가중치(exp(점수 − 1위))로 답별 비중을 내고 그 엔트로피(bit). 답이 고르게 갈릴수록 크다
        wts = np.exp(np.array([float(x["score"]) for x in near]) - best)
        grp = {}
        for v, w_ in zip(answers, wts):
            grp[v] = grp.get(v, 0.0) + w_
        pr = np.array(list(grp.values())) / max(wts.sum(), 1e-12)
        ent = float(-(pr * np.log2(np.maximum(pr, 1e-12))).sum())
        if len(near) < 2:
            info = False
        elif kind == "번호판":
            info = len(distinct) >= 2 and any(v > 0 for v in distinct)
        else:
            info = len(distinct) >= 2
        return dict(t=t, kind=kind, n_near=len(near), answers=[int(v) for v in answers], informative=bool(info), info_bits=ent)

    out = []
    for e in json.load(open(Path(a.run) / "events_live_replay.json")):
        if e.get("xyz"):
            out.append(dict(judge(float(e["t"]), "번호판", np.array(e["xyz"][:2], float), (e.get("z") or [None])[0]), picks=e["picks"][:1]))
    fp = Path(a.run) / "facility_events.json"
    if fp.exists():
        for e in json.load(open(fp)):
            if e.get("label") != "계단" or not e.get("z"):
                continue
            b = e["picks"][0]; z = float(e["z"][0]); t = float(e["t"])
            k = float(ts[np.argmin(np.abs(ts - t))]); r = tl[k]
            yaw = float(r["odom_yaw_rad"]); o = np.array([float(r["odom_x_m"]), float(r["odom_y_m"])])
            fwd, left = z, -(((b[0] + b[2]) / 2 - CX) / FX * z)
            c, s = math.cos(yaw + YAW_OFF), math.sin(yaw + YAW_OFF)
            out.append(dict(judge(t, "계단", o + np.array([c * fwd - s * left, s * fwd + c * left]), z), picks=[b]))
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    n = len(out); ni = sum(x["informative"] for x in out)
    bits = np.array([x["info_bits"] for x in out]) if out else np.zeros(0)
    print(f"{Path(a.run).parent.name}: 이벤트 {n}개 중 informative {ni} ({100 * ni / max(n, 1):.0f}%) | 정보량 중앙 {np.median(bits) if n else 0:.2f} bit,"
          f" ≥0.5 bit {int((bits >= 0.5).sum())}, ≥1.0 bit {int((bits >= 1.0).sum())}")


if __name__ == "__main__":
    main()
