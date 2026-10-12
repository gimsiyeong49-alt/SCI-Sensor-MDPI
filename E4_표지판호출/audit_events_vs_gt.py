#!/usr/bin/env python3
"""이벤트 알고리즘 점검(2026-10-05): 채점용 정답으로 게이트·멈춤을 숫자로 본다.

층마다:
  게이트: 매 초 '켜짐(애매 또는 튐+흩어짐)'과 '위치추정 1등이 실제로 틀림(3 m 밖)'을 맞대어
          켜짐 중 실제로 틀린 비율, 틀린 시간 중 켜진 비율.
  번호판 멈춤: 멈춘 순간 위치가 이미 맞았는지(1등 3 m 안이고 확신), 같은 자리 재멈춤(정답 좌표로 1.5 m 안에 이전 멈춤),
          정답 자세로 멈춘 자리를 지도에 찍었을 때 '호실 경계 1 m 안 + 복도 1.5 m 안'(번호판 있을 법한 자리)인지, 높이.
  계단·엘베 멈춤: 정답 자세로 찍은 자리가 같은 종류 시설 칸(2 m + 깊이 20%) 안인지 → 사람 판정 없는 자동 정밀도.

    python3 scripts/audit_events_vs_gt.py <층> <위치추정·이벤트 폴더(35_/<층>)> <채점용 정답 폴더> <의미지도> <base_map> <오돔 csv>
"""
import csv
import json
import math
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

FX, CX = 606.80, 310.49
YAW_OFF = math.radians(228.0)
KINDS = {"계단": "stair", "엘리베이터": "elevator", "화장실": "restroom"}


def A(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


def mask(shape, polys):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if p and len(p) >= 3:
            cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def main():
    F, R, GD, SEM, BASE, ODOM = sys.argv[1:7]
    S = json.load(open(f"{GD}/summary.json")); ppm = S["ppm"]; Mg = A(math.radians(S["theta_deg"])) * ppm
    g = list(csv.DictReader(open(f"{GD}/warp_clean/gt_warp.csv")))
    gt = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
    rows = list(csv.DictReader(open(ODOM)))
    oT = np.array([float(r["t_sec"]) for r in rows]); oT -= oT[0]
    oXY = np.array([[float(r["fastlio_x"]), float(r["fastlio_y"])] for r in rows]); oY = np.array([float(r["fastlio_yaw_rad"]) for r in rows])
    gi = lambda t: int(np.argmin(abs(gt - t))); oi = lambda t: int(np.argmin(abs(oT - t)))
    def to_map(t, xy):
        return gxy[gi(t)] + Mg @ (np.asarray(xy) - oXY[oi(t)])
    im = cv2.imread(BASE, cv2.IMREAD_GRAYSCALE); H, W = im.shape
    sem = json.load(open(SEM))
    rm = mask((H, W), [s.get("polygon_px") for s in sem["spaces"] if s.get("type") == "room"])
    edge = rm & cv2.dilate((~rm).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    DTe = distance_transform_edt(~edge)
    cm = mask((H, W), [c.get("polygon_px") for c in sem.get("corridors", [])])
    DTc = distance_transform_edt(~cm) if cm.any() else np.zeros((H, W))
    DTf = {k: distance_transform_edt(~mask((H, W), [s.get("polygon_px") for s in sem["spaces"] if s.get("type") == v]))
           for k, v in KINDS.items() if any(s.get("type") == v for s in sem["spaces"])}
    def look(D, p):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(D[y, x]) / ppm if 0 <= x < W and 0 <= y < H else 1e9

    # 1초 위치추정 맞음/틀림
    tk = {}
    for r in csv.DictReader(open(f"{R}/loc/topk_series.csv")):
        if r["rank"] == "1":
            tk[float(r["relative_sec"])] = (float(r["x_px"]), float(r["y_px"]))
    lv = {float(r["relative_sec"]): r["confidence_level"] for r in csv.DictReader(open(f"{R}/loc/timeline.csv"))}
    ok = {t: math.hypot(x - gxy[gi(t)][0], y - gxy[gi(t)][1]) / ppm < 3 for t, (x, y) in tk.items()}
    gate = {round(x["t"]): bool(x["amb"] or (x["jump"] and x["spread"] >= 5)) for x in json.load(open(f"{R}/모호함.json"))}
    ts = [t for t in ok if round(t) in gate]
    on = np.array([gate[round(t)] for t in ts]); wrong = np.array([not ok[t] for t in ts])
    res = dict(F=F, gate_on=on.mean(), wrong=wrong.mean(),
               on_given_wrong=on[wrong].mean() if wrong.any() else np.nan,     # 틀린 시간 중 켜짐
               wrong_given_on=wrong[on].mean() if on.any() else np.nan,        # 켜짐 중 실제로 틀림
               on_given_right=on[~wrong].mean() if (~wrong).any() else np.nan)  # 맞는데 켜짐
    # 번호판 멈춤
    E = json.load(open(f"{R}/run/events_live_replay.json"))
    prev, n_dup, n_ok_loc, n_plaus, hs, n = [], 0, 0, 0, [], 0
    for e in E:
        if not e.get("xyz"):
            continue
        n += 1
        t = float(e["t"]); mp = to_map(t, e["xyz"][:2])
        if any(np.hypot(*(mp - q)) / ppm < 1.5 for q in prev):
            n_dup += 1
        prev.append(mp)
        k = min(ok, key=lambda s: abs(s - t))
        if ok[k] and lv.get(k) == "CONFIDENT":
            n_ok_loc += 1
        if look(DTe, mp) <= 1.0 and look(DTc, mp) <= 1.5:
            n_plaus += 1
        hs.append(e["xyz"][2])
    res.update(n_stop=n, dup=n_dup / max(n, 1), stop_loc_ok=n_ok_loc / max(n, 1), plausible=n_plaus / max(n, 1),
               h_med=float(np.median(hs)) if hs else np.nan)
    # 계단·엘베 멈춤 자동 채점
    fac = {}
    for e in json.load(open(f"{R}/run/facility_events.json")):
        if e.get("label") not in KINDS or not e.get("z"):
            continue
        b = e["picks"][0]; z = float(e["z"][0]); t = float(e["t"])
        fwd, left = z, -(((b[0] + b[2]) / 2 - CX) / FX * z)
        i = oi(t); c, s = math.cos(oY[i] + YAW_OFF), math.sin(oY[i] + YAW_OFF)
        mp = to_map(t, oXY[i] + np.array([c * fwd - s * left, s * fwd + c * left]))
        tol = 2.0 + 0.2 * z
        hit = e["label"] in DTf and look(DTf[e["label"]], mp) <= tol
        other = [k for k, D in DTf.items() if k != e["label"] and look(D, mp) <= tol]
        fac.setdefault(e["label"], []).append((round(t), hit, other))
    res["fac"] = fac
    print(json.dumps(res, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
