#!/usr/bin/env python3
"""E4 센서 이상 탐지 주입 시험(2026-10-12, PILOT). 65개 출발점마다 구간 30초 지점에 오돔 순간 이동·순간 회전을 넣고
두 판단을 비교한다.
  1초 규칙 : 지금 애매 기록(e4_unknown_gate.py)이 쓰는 것. 1초 전 대비 속도 > 2 m/s 또는 yaw 속도 > 120°/s
  OdomGuard : scripts/sensor_check.py check() 그대로. 0.1 s 오돔 걸음(3 m/s, 150°/s) + 오돔 끊김 + 자이로와 회전 맞춤(9°).
              자이로는 녹화 IMU 시각이 맞는 층(1·2·4층, 공학관 1·2층)만 켜진다(sensor_check 의 gyro_ok 판단 그대로)
주입은 그 시각 이후 오돔 전체를 옮긴다(이동은 진행 방향과 무관하게 x 로, 회전은 yaw 에 더함). 탐지 = 주입 시각 −0.2~+1.0 초 안에 이상 판정.
정상 주행 오탐: 주입 없이 같은 65개 구간 60초 동안 이상 판정이 난 횟수.
    python3 scripts/e4_sensor_inject.py --workers 10
"""
import argparse, csv, json, math, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import sensor_check as SC  # noqa: E402

W = Path("/data2/e4_unknown_20261011"); OUT = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/70_층모름E4/E4_상태판단")
SCJ = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/59_E4_호출정책/paper/sensor_check")
MOVES = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]; TURNS = [5, 10, 20, 30, 45, 90, 150]
_G = {}


def odom_path(F):
    return "/data/kci_work/a131_full_v0_4/odom_link.csv" if F == "4F" else f"/data2/loc_inputs_20261005/{F}/odom_link.csv"


def floor_data(F):
    if F not in _G:
        t, x, y, yaw = SC.load_odom(odom_path(F))
        bag = json.load(open(SCJ / f"{F}.json")).get("bag")
        gyro = SC.load_gyro(bag) if bag else None
        _G[F] = (t, x, y, yaw, gyro)
    return _G[F]


def rule1s(t, x, y, yaw, t_abs):
    i1 = int(np.argmin(abs(t - t_abs))); i0 = int(np.argmin(abs(t - (t_abs - 1.0)))); dt = t[i1] - t[i0]
    if dt <= 0:
        return False
    sp = math.hypot(x[i1] - x[i0], y[i1] - y[i0]) / dt; d = yaw[i1] - yaw[i0]
    return sp > 2.0 or abs(math.degrees(math.atan2(math.sin(d), math.cos(d)))) / dt > 120.0


def job(arg):
    F, tasks = arg
    t, x, y, yaw, gyro = floor_data(F)
    ev0, info0 = SC.check(t, x, y, yaw, gyro)
    gyro_on = info0 is not None and "skipped" not in info0
    rows = []
    for k, ts in tasks:
        t_s = t[0] + ts; t_inj = t_s + 30.0; seg = (t_s, t_s + 60.0)
        base_guard = sum(1 for e in ev0 if seg[0] <= e[0] <= seg[1])
        base_rule = sum(rule1s(t, x, y, yaw, t_s + s) for s in range(1, 61))
        rows.append(dict(task=k, floor=F, kind="정상", size=0, guard=base_guard, rule=base_rule, gyro=int(gyro_on)))
        kidx = int(np.searchsorted(t, t_inj))
        for kind, sizes in (("이동(m)", MOVES), ("회전(°)", TURNS)):
            for sz in sizes:
                X, Y, YW = x.copy(), y.copy(), yaw.copy()
                if kind.startswith("이동"):
                    X[kidx:] += sz
                else:
                    YW[kidx:] += math.radians(sz)
                ev, _ = SC.check(t, X, Y, YW, gyro)
                g = int(any(t_inj - 0.2 <= e[0] <= t_inj + 1.0 for e in ev))
                r = int(any(rule1s(t, X, Y, YW, t[kidx] + d) for d in (0.0, 1.0)))
                rows.append(dict(task=k, floor=F, kind=kind, size=sz, guard=g, rule=r, gyro=int(gyro_on)))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    by = {}
    for s in csv.DictReader(open(W / "segments.csv")):
        by.setdefault(s["floor"], []).append((s["task_id"], float(s["start_sec"])))
    with Pool(a.workers) as p:
        rows = sum(p.map(job, list(by.items())), [])
    with open(OUT / "센서주입_출발점별.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["| 종류 | 크기 | 1초 규칙 | OdomGuard (전체) | OdomGuard (자이로 켜진 층) | OdomGuard (자이로 없는 층) |", "|---|---|---|---|---|---|"]
    for kind, sizes in (("이동(m)", MOVES), ("회전(°)", TURNS)):
        for sz in sizes:
            rr = [r for r in rows if r["kind"] == kind and r["size"] == sz]; gy = [r for r in rr if r["gyro"]]; ng = [r for r in rr if not r["gyro"]]
            f = lambda v: f"{sum(r['guard'] for r in v)}/{len(v)}"
            L.append(f"| {kind} | {sz} | {sum(r['rule'] for r in rr)}/{len(rr)} | {f(rr)} | {f(gy)} | {f(ng)} |")
    base = [r for r in rows if r["kind"] == "정상"]
    L.append(f"\n정상 주행 오탐(주입 없이 65개 구간 × 60초): 1초 규칙 {sum(r['rule'] for r in base)}번, OdomGuard {sum(r['guard'] for r in base)}번 "
             f"(구간 {sum(1 for r in base if r['guard'])}곳)")
    txt = "\n".join(L); print(txt); (OUT / "센서주입_결과.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()
