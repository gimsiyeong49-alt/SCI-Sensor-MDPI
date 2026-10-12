#!/usr/bin/env python3
"""이벤트 관측(호실 번호판·계단·엘베)을 위치추정 후보 점수에 반영한다(2026-10-05, 작업 순서 ②).
협업자 RTLocalizer 출력(1초 후보 12개) 위에서 돈다. 협업자 방식대로 지난 60초 관측만 쓴다.

매 초, 후보마다:
  1) 지난 window 초 안의 관측 3D 좌표(오돔 좌표계)를 "그 후보 위치였다면 지도 어디에 찍히나"로 옮긴다.
     지도점 = 후보 픽셀 + 추정 축척 · A(φ) · (관측 오돔 xy − 지금 오돔 xy), A(φ) = [[c, −s], [−s, −c]](협업자 _A, 지도 y 아래).
  2) 찍힌 자리가 관측 종류와 맞는지 본다(L = 1 맞음, 아니면 eps).
     계단·엘베·화장실: 의미지도의 같은 종류 칸에서 tol 안(다른 종류 시설 칸이면 other 배).
     호실 번호판(종류 수준): 호실 칸 경계에서 plaque_tol 안이고 복도 칸에서 corr_tol 안.
     호실 번호(--oracle-numbers, 시뮬레이션): 그 번호 호실 칸 경계에서 plaque_tol 안이고 복도 칸 근처.
  3) 새 점수 = 위치추정 점수 + Σ w·ln(max(L, eps)). 다시 줄 세우고 확신도는 같은 정의(1등과 5 m 밖 최고 후보 차 > 1.5).
관측 방향·깊이는 지금 RealSense 상자(번호판은 이벤트 기록의 xyz, 시설은 상자 중심과 깊이). 현장에서는 팬틸트 각도·깊이로 바꾼다.

--oracle-numbers <채점용 정답 폴더>: "고화질 판독이 번호를 읽었다면" 시뮬레이션. 멈춤마다 정답 자세로 관측점을 지도에 찍어
가장 가까운 호실(경계 1.5 m 안)의 번호를 읽은 것으로 친다(가까운 호실이 없으면 번호판이 아니었던 것으로 보고 버림).
정답을 관측 생성에만 쓰고 위치추정·반영 계산에는 쓰지 않는다. 판독이 될 때 얻을 수 있는 효과의 상한을 보는 용도다.

    python3 scripts/obs_feedback.py --loc <위치추정 폴더> --run <이벤트 run 폴더> --sem <의미지도.json> --base <base_map.png> --out <결과>
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

FX, CX = 606.80, 310.49          # RealSense 640×480(region_memory.py 와 같음)
YAW_OFF = math.radians(228.0)    # 이 데이터셋 규약: 센서 방향 = 오돔 yaw + 228°
KINDS = {"계단": "stair", "엘리베이터": "elevator", "화장실": "restroom"}


def A(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


def poly_mask(shape, polys):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        if p is not None and len(p) >= 3:
            cv2.fillPoly(m, [np.array(p, np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--sem", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--use", default="facility,plaque", help="facility, plaque 중 쉼표로")
    ap.add_argument("--fac-kinds", default="계단,엘리베이터,화장실", help="반영할 시설 종류(쉼표)")
    ap.add_argument("--w-fac", type=float, default=1.0); ap.add_argument("--eps-fac", type=float, default=0.2)
    ap.add_argument("--other", type=float, default=0.5)
    ap.add_argument("--fac-tol", type=float, default=2.0, help="m, 시설 칸까지 허용(+깊이의 20%)")
    ap.add_argument("--w-plq", type=float, default=1.0); ap.add_argument("--eps-plq", type=float, default=0.5,
                    help="종류 수준 번호판(멈춤의 절반가량이 번호판 아님 → 약하게)")
    ap.add_argument("--plaque-tol", type=float, default=1.0); ap.add_argument("--corr-tol", type=float, default=1.5)
    ap.add_argument("--oracle-numbers", default="", help="시뮬레이션: 채점용 정답 폴더(번호 판독이 된다고 가정)")
    ap.add_argument("--w-num", type=float, default=1.0); ap.add_argument("--eps-num", type=float, default=0.1)
    ap.add_argument("--max-contra", type=int, default=-1,
                    help="확신하려면 1등이 최근 관측과 모순되는 개수가 이 이하여야 함(-1 이면 조건 없음). "
                         "후보 12개에 정답 근처가 없을 때 '덜 모순되는' 후보가 확신을 얻는 것을 막는다")
    ap.add_argument("--door-tol", type=float, default=0.0,
                    help="번호 관측: 그 호실 문(의미지도 doors) 중심에서 이 거리(m) 안이어야 맞음. 0 이면 호실 벽 전체. 문 정보 없는 호실은 벽 전체")
    ap.add_argument("--dedup-num", action="store_true",
                    help="창 안에서 같은 호실 번호 판독은 가장 최근 것 하나만 센다(2026-10-08). 같은 번호를 몇 초 만에 다시 읽은 것이 "
                         "새 근거로 더해지면 많이 멈추는 정책이 부풀려진다")
    ap.add_argument("--tol-per-m", type=float, default=0.0,
                    help="관측이 지금 위치에서 멀수록 허용을 넓힘(m 당). 축척·방향 오차가 거리에 비례해 커지기 때문(0.08 = 8%%)")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "timeline.csv").exists():
        sys.exit(f"{out} 에 결과가 이미 있다(덮어쓰지 않음)")
    out.mkdir(parents=True, exist_ok=True)
    use = set(a.use.split(","))

    im = cv2.imread(a.base, cv2.IMREAD_GRAYSCALE); H, W = im.shape
    sem = json.load(open(a.sem))
    rooms = [s for s in sem["spaces"] if s.get("type") == "room" and s.get("polygon_px")]
    room_mask = poly_mask((H, W), [r["polygon_px"] for r in rooms])
    edge = room_mask & cv2.dilate((~room_mask).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    DT_redge = distance_transform_edt(~edge)
    corr = poly_mask((H, W), [c.get("polygon_px") for c in sem.get("corridors", [])])
    DT_corr = distance_transform_edt(~corr) if corr.any() else np.zeros((H, W))
    DT_fac = {}
    for k, v in KINDS.items():
        m = poly_mask((H, W), [s.get("polygon_px") for s in sem["spaces"] if s.get("type") == v])
        if m.any():
            DT_fac[k] = distance_transform_edt(~m)
    room_edge_dt = {}
    room_door = {}
    for d in sem.get("doors", []) or []:
        if d.get("room_id"):
            room_door.setdefault(d["room_id"], []).append(np.array(d["center_px"], float))

    def look(img, p, cval):
        x, y = int(round(p[0])), int(round(p[1]))
        return float(img[y, x]) if 0 <= x < W and 0 <= y < H else cval

    tk = {}
    for r in csv.DictReader(open(Path(a.loc) / "topk_series.csv")):
        tk.setdefault(float(r["relative_sec"]), []).append(r)
    tl = {float(r["relative_sec"]): r for r in csv.DictReader(open(Path(a.loc) / "timeline.csv"))}
    ts = sorted(tk)
    od = {t: (np.array([float(tl[t]["odom_x_m"]), float(tl[t]["odom_y_m"])]), float(tl[t]["odom_yaw_rad"])) for t in ts}

    def odom_at(t):
        k = min(ts, key=lambda s: abs(s - t))
        return od[k]

    # 관측 목록: (시각, 종류, 오돔 xy, 깊이, 번호)
    obs = []
    if "plaque" in use:
        for e in json.load(open(Path(a.run) / "events_live_replay.json")):
            if e.get("xyz"):
                obs.append(dict(t=float(e["t"]), kind="번호판", xy=np.array(e["xyz"][:2], float), z=(e.get("z") or [None])[0], num=None))
    if "facility" in use and (Path(a.run) / "facility_events.json").exists():
        for e in json.load(open(Path(a.run) / "facility_events.json")):
            if e.get("label") not in KINDS or e.get("label") not in a.fac_kinds.split(",") or not e.get("z"):
                continue
            b = e["picks"][0]; z = float(e["z"][0])
            fwd, left = z, -(((b[0] + b[2]) / 2 - CX) / FX * z)
            o, yaw = odom_at(float(e["t"]))
            c, s = math.cos(yaw + YAW_OFF), math.sin(yaw + YAW_OFF)
            obs.append(dict(t=float(e["t"]), kind=e["label"], xy=o + np.array([c * fwd - s * left, s * fwd + c * left]), z=z, num=None))
    # 시뮬레이션: 번호 판독
    n_num = 0
    if a.oracle_numbers:
        G = Path(a.oracle_numbers); S = json.load(open(G / "summary.json"))
        gfile = G / "warp_clean" / "gt_warp.csv"
        g = list(csv.DictReader(open(gfile if gfile.exists() else G / "gt.csv")))
        gt_t = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
        Mg = A(math.radians(S["theta_deg"])) * S["ppm"] if S["hand"] > 0 else None
        for o in obs:
            if o["kind"] != "번호판" or Mg is None:
                continue
            i = int(np.argmin(abs(gt_t - o["t"]))); oo, _ = odom_at(o["t"])
            mp = gxy[i] + Mg @ (o["xy"] - oo)
            best = None
            for r in rooms:
                key = r["id"]
                if key not in room_edge_dt:
                    m = poly_mask((H, W), [r["polygon_px"]])
                    e2 = m & cv2.dilate((~m).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    room_edge_dt[key] = distance_transform_edt(~e2)
                d = look(room_edge_dt[key], mp, 1e9) / S["ppm"]
                if d <= 1.5 and (best is None or d < best[0]):
                    best = (d, key, r.get("room_number"))
            if best:
                o["num"] = best[1]; o["room_number"] = best[2]; n_num += 1

    rows_tl, rows_tk = [], []
    stat = dict(n_obs=len(obs), n_num=n_num, kinds={})
    for o in obs:
        stat["kinds"][o["kind"]] = stat["kinds"].get(o["kind"], 0) + 1
    for t in ts:
        r = tl[t]; ppm = float(r.get("ppm_est") or 0) or 30.0
        o_now, _ = od[t]
        recent = [o for o in obs if t - a.window < o["t"] <= t + 1e-6]
        if a.dedup_num:                              # 같은 번호는 가장 최근 판독 하나만
            last = {}
            for o in recent:
                if o["kind"] == "번호판" and o.get("num"):
                    last[o["num"]] = o
            recent = [o for o in recent if not (o["kind"] == "번호판" and o.get("num")) or last[o["num"]] is o]
        cands = []
        for c in tk[t]:
            px = np.array([float(c["x_px"]), float(c["y_px"])]); phi = math.radians(float(c["phi_deg"]))
            M = A(phi) * ppm
            ev = 0.0; contra = 0
            for o in recent:
                mp = px + M @ (o["xy"] - o_now)
                grow = a.tol_per_m * float(np.linalg.norm(o["xy"] - o_now))
                if o["kind"] == "번호판":
                    if a.oracle_numbers and o.get("num"):
                        if a.door_tol > 0 and o["num"] in room_door:
                            ok = min(np.hypot(*(mp - dd)) for dd in room_door[o["num"]]) / ppm <= a.door_tol + grow
                        else:
                            ok = look(room_edge_dt[o["num"]], mp, 1e9) / ppm <= a.plaque_tol + grow and look(DT_corr, mp, 1e9) / ppm <= a.corr_tol + grow
                        ev += a.w_num * math.log(1.0 if ok else a.eps_num); contra += (not ok)
                    elif not a.oracle_numbers:
                        ok = look(DT_redge, mp, 1e9) / ppm <= a.plaque_tol and look(DT_corr, mp, 1e9) / ppm <= a.corr_tol
                        ev += a.w_plq * math.log(1.0 if ok else a.eps_plq)
                else:
                    tol = a.fac_tol + 0.2 * (o["z"] or 0) + grow
                    if o["kind"] in DT_fac and look(DT_fac[o["kind"]], mp, 1e9) / ppm <= tol:
                        L = 1.0
                    elif any(look(D, mp, 1e9) / ppm <= tol for k, D in DT_fac.items() if k != o["kind"]):
                        L = a.other
                    else:
                        L = 0.0
                    ev += a.w_fac * math.log(max(L, a.eps_fac)); contra += (L < 1.0)
            cands.append((float(c["score"]) + ev, c, ev, contra))
        cands.sort(key=lambda x: -x[0])
        top = cands[0]
        margin = None
        for s2, c2, _, _ in cands[1:]:
            if math.hypot(float(c2["x_px"]) - float(top[1]["x_px"]), float(c2["y_px"]) - float(top[1]["y_px"])) / ppm > 5.0:
                margin = top[0] - s2
                break
        rows_tl.append(dict(relative_sec=t, scene=r.get("scene", ""),
                            confidence_level="CONFIDENT" if (margin is not None and margin > 1.5
                                                             and (a.max_contra < 0 or top[3] <= a.max_contra)) else "AMBIGUOUS",
                            confidence_margin=margin if margin is not None else "", ppm_est=r.get("ppm_est", ""),
                            n_obs=len(recent), odom_x_m=r["odom_x_m"], odom_y_m=r["odom_y_m"], odom_yaw_rad=r["odom_yaw_rad"]))
        for k, (s2, c2, ev, _) in enumerate(cands, 1):
            rows_tk.append(dict(relative_sec=t, candidate_id=f"t{t:.3f}_top_{k:02d}", rank=k, x_px=c2["x_px"], y_px=c2["y_px"],
                                yaw_rad=c2["yaw_rad"], score=s2, evidence=ev, phi_deg=c2["phi_deg"],
                                odom_x_m=r["odom_x_m"], odom_y_m=r["odom_y_m"], odom_yaw_rad=r["odom_yaw_rad"]))
    for name, rows in (("timeline.csv", rows_tl), ("topk_series.csv", rows_tk)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stat.update(args=vars(a), ambiguous=float(np.mean([x["confidence_level"] == "AMBIGUOUS" for x in rows_tl])))
    json.dump(stat, open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"{out.name}: 관측 {len(obs)}개 {stat['kinds']}" + (f", 번호 판독(시뮬) {n_num}개" if a.oracle_numbers else "") +
          f", 애매 {stat['ambiguous'] * 100:.0f}%")


if __name__ == "__main__":
    main()
