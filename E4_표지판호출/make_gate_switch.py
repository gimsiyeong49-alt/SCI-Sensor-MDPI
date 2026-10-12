#!/usr/bin/env python3
"""모호함 기록(모호함_v2.json)에 '1위 순위 급변 뒤 K초는 애매'를 더한 새 기록을 만든다(2026-10-11, 모호함 기준 재정립).
근거·정의: docs/모호함_기준_재정립_논의용_20261011.md 6절.
  1위 이동(급변) = jump(1위가 1초에 오돔보다 3 m 넘게 더 움직임) 그리고 spread ≥ 5 m(5 m 넘게 떨어진 다른 후보로 넘어감) 그리고 센서 정상
  새 amb = 기존 amb 또는 (마지막 1위 이동 뒤 K초 미만)
다른 키는 그대로 두고, 바뀐 줄에 amb_src="switch" 를 단다. 이벤트 실행기는 고치지 않는다(--gate 로 이 파일을 준다).
    python3 scripts/make_gate_switch.py --src .../35_위치추정_축척추정/4F/모호함_v2.json --out .../4F/모호함_v3_급변10초.json --k 10
"""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=float, default=10.0, help="1위 이동 뒤 애매로 둘 시간(초)")
ap.add_argument("--min-spread", type=float, default=5.0)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} 이 이미 있다(덮어쓰지 않음)")
rows = json.load(open(a.src)); last = None; added = 0
for x in rows:
    t = float(x["t"])
    if x["jump"] and float(x.get("spread", 0)) >= a.min_spread and not x.get("sensor_bad", False):
        last = t
    if not x["amb"] and last is not None and t - last < a.k:
        x["amb"] = True; x["amb_src"] = "switch"; added += 1
json.dump(rows, open(out, "w"), ensure_ascii=False)
n = len(rows); amb = sum(x["amb"] for x in rows)
print(f"{out.parent.name}: {n}초, 급변 뒤 애매로 바뀐 초 {added}, 애매 {amb / n * 100:.1f}%")
