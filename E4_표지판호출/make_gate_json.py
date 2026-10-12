#!/usr/bin/env python3
"""위치추정 1초 기록(export_4f_causal_topk_series.py 출력: timeline.csv, topk_series.csv)으로 게이트 기록(json)을 만든다.

4층 14_전체주행_모호함/모호함_전체주행.json 과 같은 정의(2026-09-23 대화 기록의 코드를 그대로 옮김):
  amb    = confidence_level == AMBIGUOUS (1위와 5 m 밖 최고 봉우리 점수 차 ≤ 1.5)
  jump   = 1위의 1초 이동량(px / PPM) − 오돔 이동량(timeline 의 odom_x/y) > 3 m
  spread = 1위 점수 − 1.5 안에 드는 후보들 사이의 최대 거리(m), npk = 그 후보 수
  turn   = 오돔 yaw 의 ±0.5초 변화(도). odom_link 와 t0 를 주면 계산, 아니면 timeline 의 yaw 차분.

    python3 scripts/make_gate_json.py --loc 결과/loc_1hz --out 결과/모호함_전체주행.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np


SENSOR_MAX_SPEED = 2.0      # m/s. Go2 걷기(오돔 p99 0.55~1.0 m/s)로는 나올 수 없는 순간 이동 → 오돔·센서 이상
SENSOR_MAX_YAW_RATE = 120.0  # °/s (p99 30~45°/s)


def gate_row(t, level, margin, cands, odom_xy, ppm, prev, turn=0.0, odom_yaw=None):
    """1초 기록 하나로 게이트 한 줄. cands = [(score, x_px, y_px), ...], prev = 지난 줄의 (1위 px, 오돔 m) 또는 None.
    돌려주는 값: (줄, 다음에 넘길 prev). 실시간 이벤트 실행기(live_event_runner.py live)도 이것을 그대로 쓴다.
    2026-10-07: sensor_bad(지난 기록 대비 오돔 속도 > 2 m/s 또는 yaw 속도 > 120°/s) 추가. 실험 지침서 E4 의 '오돔·센서 이상'을
    장소 애매와 따로 표시한다(이 경우 이벤트를 부르지 않는다). 기존 키 값은 그대로."""
    c = sorted(cands, key=lambda x: -x[0])
    top = np.array([c[0][1], c[0][2]])
    o = np.asarray(odom_xy, float)
    s = np.array([x[0] for x in c])
    xy = np.array([[x[1], x[2]] for x in c])
    near = s >= s.max() - 1.5
    spread = float(np.max(np.linalg.norm(xy[near][:, None] - xy[near][None], axis=-1)) / ppm) if near.sum() > 1 else 0.0
    jump = None
    sensor_bad = False
    if prev is not None:
        jump = (np.linalg.norm(top - prev[0]) / ppm, np.linalg.norm(o - prev[1]))
        if len(prev) >= 4 and prev[2] is not None and t > prev[2]:
            dt = t - prev[2]
            sp = np.linalg.norm(o - prev[1]) / dt
            yr = abs(np.degrees(np.angle(np.exp(1j * (odom_yaw - prev[3]))))) / dt if (odom_yaw is not None and prev[3] is not None) else 0.0
            sensor_bad = bool(sp > SENSOR_MAX_SPEED or yr > SENSOR_MAX_YAW_RATE)
    row = dict(t=t, margin=float(margin), amb=level == "AMBIGUOUS",
               jump=bool(jump and jump[0] - jump[1] > 3.0), spread=spread, npk=int(near.sum()), turn=float(turn),
               sensor_bad=sensor_bad)
    return row, (top, o, t, odom_yaw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", required=True, help="timeline.csv, topk_series.csv 가 있는 폴더")
    ap.add_argument("--ppm", type=float, default=24.14, help="지도 px/m. timeline 에 ppm_est(주행 중 추정 축척) 열이 있으면 그 값을 씀")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensor-events", default=None, help="sensor_check.py 결과 json(2026-10-08). 그 초(t−1, t]에 이상이 있으면 sensor_bad 로 더한다")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} 이 이미 있다(덮어쓰지 않음)")
    D = Path(a.loc)
    tl = list(csv.DictReader(open(D / "timeline.csv")))
    tk = collections.defaultdict(list)
    for r in csv.DictReader(open(D / "topk_series.csv")):
        tk[float(r["relative_sec"])].append(r)
    yaw = {float(r["relative_sec"]): float(r["odom_yaw_rad"]) for r in tl}
    rows, prev = [], None
    for r in tl:
        t = float(r["relative_sec"])
        ppm = float(r["ppm_est"]) if r.get("ppm_est") else a.ppm
        cands = [(float(x["score"]), float(x["x_px"]), float(x["y_px"])) for x in tk[t]]
        y0, y1 = yaw.get(t - 1.0, yaw[t]), yaw.get(t + 1.0, yaw[t])
        turn = abs(np.degrees(np.angle(np.exp(1j * (y1 - y0))))) / 2
        row, prev = gate_row(t, r["confidence_level"], r["confidence_margin"], cands,
                             (float(r["odom_x_m"]), float(r["odom_y_m"])), ppm, prev, turn, odom_yaw=float(r["odom_yaw_rad"]))
        rows.append(row)
    if a.sensor_events:
        ev = [e["rel_sec"] for e in json.load(open(a.sensor_events))["events"]]
        for row in rows:
            if any(row["t"] - 1.0 < x <= row["t"] + 1e-6 for x in ev):
                row["sensor_bad"] = True; row["sensor_src"] = "sensor_check"
    json.dump(rows, open(out, "w"), ensure_ascii=False)
    on = [x["amb"] or (x["jump"] and x["spread"] >= 5) for x in rows]
    print(f"{out}: {len(rows)}초, 모호함 {np.mean([x['amb'] for x in rows]) * 100:.0f}%, 튐 {np.mean([x['jump'] for x in rows]) * 100:.0f}%, "
          f"게이트 켜짐 {np.mean(on) * 100:.0f}%")


if __name__ == "__main__":
    main()
