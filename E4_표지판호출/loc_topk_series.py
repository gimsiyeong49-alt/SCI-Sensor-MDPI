#!/usr/bin/env python3
"""협업자 최종 위치추정(RTLocalizer, Go2 floor_id_pkg/floor_id/localizer/rt_localizer.py)을 1초마다 불러
우리 게이트가 읽는 형식(timeline.csv, topk_series.csv)으로 저장한다(2026-10-04). 협업자 코드는 고치지 않는다.

  - 지도: 협업자 패키지 안 지도(maps/<건물>_maps_perfloor). 우리 의미지도 base_map 과 13개 층 모두 픽셀이 같다(확인함).
  - 축척: --scale-online(2026-10-05, 기본으로 쓸 것) 이면 실측 축척 없이 주행 중 추정한다.
      fit-step 초마다 협업자 metric_fit_v2.fit_window_v2(층 판별과 같은 함수: 축척 목록 15~43 px/m 에서 연속 보정,
      방향·위치 동시 정합)로 그 시점까지의 창을 맞춰 (축척, 방향, 정합 비용 m)을 얻고, 매 초 그때까지 나온 정합만 써서
      방향 = 비용 가중(1/max(비용, 5 cm)) 합이 가장 큰 bootstrap mode, 축척 = 그 방향 정합들의 가중 중앙값.
      건물 축척·방향은 상수라 누적한다. 시작 15초만으로 방향을 정하는 fix_phi 는 1층에서 틀렸다(128° vs 전체 38°).
    --ppm(고정 축척)은 진단·비교용. 실측 복도 길이·주행 전체 층 판별에서 얻은 값이라 로봇은 모른다(쓰면 안 됨).
  - 방향 후보(phi): 시작 boot_s 초로 bootstrap(협업자 방식, 인과).
  - 점수: RTLocalizer.localize(t, W, dt, phis) 의 phi 별 격자 P. 후보 허용 마스크(Z.allow) 밖은 뺀다.
  - top-K 와 확신도: 우리 export_4f_causal_topk_series.py 와 같은 방식(phi 별 7x7 국소 최대점 상위 12개 → 합쳐 상위 K,
    1위와 5 m 밖 최고 후보의 점수 차). 점수는 창 안 프레임 수만큼 더해지므로 '애매함' 기준 1.5 는 우리 기존 창(18초·3초=6프레임)
    기준이라 프레임 수 비례로 환산한다(margin_per6 = margin × 6 / 프레임 수 와 1.5 비교).
  - 후보 방향 yaw = Z.heading(phi, 오돔 yaw) = phi + yaw + BETA(지도 m 좌표, y 위).

    python3 scripts/loc_topk_series.py --building kci --floor 4 --data /data/kci_work/a131_full_v0_4 --ppm 24.14 \
        --start 20 --end 400 --out 결과/loc_4F --workers 12
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PKG = Path("/data/kci_work/floor_id_pkg/floor_id")
_Z = None
_A = None


def setup_env(building):
    import yaml
    cfg = yaml.safe_load((PKG / "config" / "buildings.yaml").read_text(encoding="utf-8"))
    b = cfg["buildings"][building]
    os.environ["SEMLOC_MAPS_ROOT"] = str(PKG / b["maps_root"])
    os.environ["SEMLOC_SEMMAPS_ROOT"] = str(PKG / b["semmaps_root"])
    os.environ["SEMLOC_SCENE"] = cfg.get("scene_source", "lidar")
    os.environ["SEMLOC_NO_DEPTH"] = "1"
    os.environ["SEMLOC_SCORE"] = "max"
    c = cfg.get("consts", {})
    os.environ["SCENE_FWD_DEG"] = str(c.get("scene_fwd_deg", 7.0))
    os.environ["SCENE_SIDE_LO"] = str(c.get("scene_side_lo_deg", 17.5))
    os.environ["SCENE_SIDE_HI"] = str(c.get("scene_side_hi_deg", 35.0))
    near = float(c.get("scene_side_near_m", 0) or 0)
    os.environ["SCENE_SIDE_NEAR_M"] = str(near if near > 0 else 1e9)
    os.environ["GT_REALIGN"] = "0"
    os.environ.setdefault("SEMLOC_VLM_OBS", "/nonexistent")   # lam_sem=0 이면 안 읽음
    sys.path.insert(0, str(PKG / "localizer"))


def make_localizer(a):
    import rt_localizer as RT
    import run_floor_id as R
    Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=a.floor, ppm=a.ppm if a.ppm else R.MID,
                       data_dir=a.data, bag=None)
    T0 = Z.L.T[0]
    if a.scale_online:                           # 층 판별(floor_id.py)과 같은 준비: 시작 15초 bootstrap
        Z.modes, _ = Z.bootstrap(T0, boot_s=a.boot)
    elif a.phi is not None:                        # 진단용: bootstrap 4 mode 중 주어진 각도에 가장 가까운 것으로 고정
        phis, _ = Z.bootstrap(T0, boot_s=a.boot)
        Z.phis = [min(phis, key=lambda p: abs((p - a.phi + 180) % 360 - 180))]
    elif a.fix_phi:                              # 협업자 방식: 시작 창에서 4 mode 를 겨뤄 건물 방향 하나로 확정(인과)
        phi, _ = Z.fix_phi(T0, boot_s=a.boot)    # rt_localizer 주석: 지점마다 재경쟁 9.26 m → 한 번 확정 0.74 m
        Z.phis = [phi]
    else:
        Z.bootstrap(T0, boot_s=a.boot)
    return Z


def init_worker(a):
    global _Z, _A
    setup_env(a.building)
    _A = a
    _Z = make_localizer(a)


def scale_list(a):
    """floor_id.py map_scales 와 같은 축척 목록(정보관·공대 고정 목록, 소융대는 지도 해상도 비율로 옮김)."""
    sys.path.insert(0, str(PKG))
    import floor_id as FI
    import run_floor_id as R
    b = FI.load_cfg()["buildings"][a.building]
    return FI.map_scales(b, R, int(b["n_maps"]))[a.floor]


def fit_one(tf):
    import metric_fit_v2 as V
    Z, a = _Z, _A
    t1 = time.time()
    if any(tf - a.W < j <= tf for j in getattr(a, "jumps", [])):
        return dict(tf=float(tf), ok=False, err="창 안에 오돔 튐")
    try:
        r = V.fit_window_v2(Z, Z.L.T[0] + tf, a.scales, floor=a.floor, W=a.W, dt=a.dt)
    except ValueError as e:
        return dict(tf=float(tf), ok=False, err=str(e))
    # fit_window_v2 는 창 안에서 BETA 를 다시 맞추고 끝나면 되돌린다 → 방향을 전역 BETA 기준으로 옮긴다
    phi_g = (r["phi"] + math.degrees(r["beta"] - Z.BETA)) % 360
    mode = min(Z.modes, key=lambda p: abs((p - phi_g + 180) % 360 - 180))
    return dict(tf=float(tf), ok=True, ppm=r["ppm"], phi=phi_g, mode=float(mode), cost=float(r["cost_m"]),
                wall_cost=float(r["wall_cost_m"]), sec=round(time.time() - t1, 1))


def estimate(fits, t):
    """t 까지 끝난 정합만으로 (축척, 방향 mode, 쓴 정합 수). 없으면 None."""
    f = [x for x in fits if x["ok"] and x["tf"] <= t + 1e-6]
    if not f:
        return None
    w = {}
    for x in f:
        w[x["mode"]] = w.get(x["mode"], 0.0) + 1.0 / max(x["cost"], 0.05)
    mode = max(w, key=w.get)
    g = sorted((x["ppm"], 1.0 / max(x["cost"], 0.05)) for x in f if x["mode"] == mode)
    half, acc = sum(v for _, v in g) / 2, 0.0
    for ppm, v in g:
        acc += v
        if acc >= half:
            return ppm, mode, len(f)


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def one(job):
    from scipy.ndimage import maximum_filter
    rel, ppm, mode = job
    Z, a = _Z, _A
    L = Z.L
    t = L.T[0] + rel
    t1 = time.time()
    phis = Z.phis
    if ppm is not None:
        from metric_fit import set_scale
        set_scale(Z, ppm)
        phis = [mode]
    W = a.W
    past = [j for j in getattr(a, "jumps", []) if j <= rel + 1e-6]
    if past:                                      # 튐 뒤 이동만 쓴다(튐 직후 1초 안은 가장 최근 스캔 하나)
        W = min(W, max(rel - past[-1] - 0.05, 0.5))
    r = Z.localize(t, W=W, dt=a.dt, phis=phis)
    allow = Z.allow.reshape(L.shp)
    i_now = int(np.searchsorted(L.T, r["t"], side="right") - 1)
    yaw_now = float(L.YAW[i_now])
    cands = []
    for phi, field in r["P"].items():
        valid = np.where(allow, field, -np.inf)
        lm = maximum_filter(valid, size=7)
        idx = np.flatnonzero((valid == lm).ravel() & np.isfinite(valid.ravel()))
        idx = idx[np.argsort(-valid.ravel()[idx])]   # 전부 둠: 5 m 밖 경쟁 후보가 상위 12개 밖에 있을 수 있다
        for k in idx:
            x, y = map(float, L.Pf[k])
            cands.append(dict(x_px=x, y_px=y, yaw_rad=wrap_pi(float(Z.heading(phi, yaw_now))),
                              score=float(valid.ravel()[k]), phi_deg=float(phi)))
    cands.sort(key=lambda c: -c["score"])
    sel = cands[:a.top_k]
    nfr = len(r["frames"])
    margin = None
    for o in cands[1:]:                          # 점수순이라 처음 만나는 5 m 밖 후보가 그중 최고
        if math.hypot(o["x_px"] - sel[0]["x_px"], o["y_px"] - sel[0]["y_px"]) / L.PPM > 5.0:
            margin = sel[0]["score"] - o["score"]
            break
    m6 = None if margin is None else margin * 6.0 / max(nfr, 1)
    return dict(rel=float(rel), sel=sel, margin=margin, margin_per6=m6, frames=nfr, scene=r["scene"], ppm=float(L.PPM),
                odom=(float(L.X[i_now]), float(L.Y[i_now]), yaw_now), sec=time.time() - t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--building", required=True, choices=["kci", "gong", "sw"])
    ap.add_argument("--floor", type=int, required=True, help="그 건물 지도 층 번호(패키지 maps/<건물>_maps_perfloor/<n>F)")
    ap.add_argument("--data", required=True, help="a1_3_1 데이터 폴더(odom_link.csv · samples · far_points.npz)")
    ap.add_argument("--ppm", type=float, default=None, help="고정 축척(진단·비교용, 실측값이라 로봇은 모름). --scale-online 과 같이 못 씀")
    ap.add_argument("--scale-online", action="store_true", help="축척·방향을 주행 중 정합 누적으로 추정(인과, 실측값 안 씀)")
    ap.add_argument("--fit-step", type=float, default=30.0, help="--scale-online 정합 간격(초)")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--W", type=float, default=30.0, help="과거 창(초). RTLocalizer 기본 30")
    ap.add_argument("--dt", type=float, default=3.0)
    ap.add_argument("--boot", type=float, default=15.0)
    ap.add_argument("--fix-phi", action="store_true", help="시작 창에서 방향 하나로 확정(협업자 fix_phi). 끄면 4 mode 매번 겨룸")
    ap.add_argument("--phi", type=float, default=None, help="진단용: 방향을 이 각도(도)에 가장 가까운 mode 로 고정(비인과 비교용)")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut-events", default=None, help="sensor_check.py 결과 json. 오돔 튐(순간 이동·회전·자이로 불일치) 시각에서 "
                    "과거 창을 끊는다: 그 뒤 위치추정은 튐 이후 이동만 쓰고, 튐이 낀 축척 정합은 버린다(2026-10-08)")
    a = ap.parse_args()
    a.jumps = []
    if a.cut_events:
        ev = [e["rel_sec"] for e in json.load(open(a.cut_events))["events"] if e["reason"] != "오돔 끊김"]
        for x in sorted(ev):                          # 1초 안에 몰린 것은 하나로
            if not a.jumps or x - a.jumps[-1] > 1.0:
                a.jumps.append(float(x))
        print(f"창 끊기: 튐 {len(a.jumps)}개 {a.jumps}", flush=True)
    if bool(a.ppm) == bool(a.scale_online):
        sys.exit("--ppm(고정, 진단용) 과 --scale-online 중 하나만 준다")
    if a.scale_online:
        setup_env(a.building)
        a.scales = scale_list(a)
    out = Path(a.out)
    if (out / "timeline.csv").exists():
        sys.exit(f"{out} 에 결과가 이미 있다(덮어쓰지 않음)")
    out.mkdir(parents=True, exist_ok=True)
    rels = list(np.arange(a.start, a.end + 1e-6, a.step))
    t0 = time.time()
    fits, jobs = [], [(r, None, None) for r in rels]
    with Pool(a.workers, initializer=init_worker, initargs=(a,)) as pool:
        if a.scale_online:
            fits = pool.map(fit_one, list(np.arange(a.start, a.end + 1e-6, a.fit_step)), chunksize=1)
            with open(out / "fits.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["tf", "ok", "ppm", "phi", "mode", "cost", "wall_cost", "sec", "err"])
                w.writeheader()
                w.writerows(fits)
            est = {r: estimate(fits, r) for r in rels}
            jobs = [(r, est[r][0], est[r][1]) for r in rels if est[r] is not None]
            print(f"정합 {sum(x['ok'] for x in fits)}/{len(fits)}개, 추정 없는 초 {len(rels) - len(jobs)}개(시작 전)", flush=True)
        res = pool.map(one, jobs, chunksize=max(1, len(jobs) // (a.workers * 4)))
    res.sort(key=lambda r: r["rel"])
    tl, tk = [], []
    for r in res:
        lvl = "CONFIDENT" if r["margin_per6"] is not None and r["margin_per6"] > 1.5 else "AMBIGUOUS"
        tl.append(dict(relative_sec=r["rel"], scene=r["scene"], confidence_level=lvl,
                       confidence_margin=r["margin_per6"] if r["margin_per6"] is not None else "",
                       raw_margin=r["margin"] if r["margin"] is not None else "", frames=r["frames"], ppm_est=r["ppm"],
                       odom_x_m=r["odom"][0], odom_y_m=r["odom"][1], odom_yaw_rad=r["odom"][2], sec=round(r["sec"], 2)))
        for i, c in enumerate(r["sel"], 1):
            tk.append(dict(relative_sec=r["rel"], candidate_id=f"t{r['rel']:.3f}_top_{i:02d}", rank=i, x_px=c["x_px"],
                           y_px=c["y_px"], x_m=c["x_px"] / r["ppm"], y_m=c["y_px"] / r["ppm"], yaw_rad=c["yaw_rad"],
                           yaw_deg=math.degrees(c["yaw_rad"]), score=c["score"] * 6.0 / max(r["frames"], 1),
                           phi_deg=c["phi_deg"], odom_x_m=r["odom"][0], odom_y_m=r["odom"][1], odom_yaw_rad=r["odom"][2]))
    for name, rows in (("timeline.csv", tl), ("topk_series.csv", tk)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    amb = np.mean([r["confidence_level"] == "AMBIGUOUS" for r in tl])
    json.dump(dict(source="collab RTLocalizer (floor_id_pkg)", args=vars(a), seconds=len(tl),
                   ambiguous=float(amb), wall_sec=round(time.time() - t0, 1),
                   note="score·margin 은 6프레임 기준으로 환산(우리 게이트 기준 1.5 그대로 쓰려고)"),
              open(out / "manifest.json", "w"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(tl)}초, 애매함 {amb * 100:.0f}%, 걸린 시간 {time.time() - t0:.0f}s, "
          f"시점당 중앙 {np.median([r['sec'] for r in res]):.1f}s")


if __name__ == "__main__":
    main()
