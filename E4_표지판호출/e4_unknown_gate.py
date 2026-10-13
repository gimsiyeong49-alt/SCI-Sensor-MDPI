#!/usr/bin/env python3
"""층 모름 E4(지침서판, 2026-10-11): 출발점 구간의 매 장면(초) 애매 기록을 E3 연결판 후보(건물 전체 층)로 만든다.
기준: reports/_02_E4_표지판호출/모호함_판단기준_재정립.md 5절. 구간·시각 맞춤은 e4_unknown_segments.py(segments.csv) 를 따른다.
  가설 = 연결판 남은 후보(3초마다, e3_seeded.score 와 같은 벽 관통 거름·같은 자세 합치기). 1위 = merit 가장 낮은 것
  다른 가설 = 층이 다르거나 같은 층에서 5 m 넘게 떨어진 것(거리 m = px / 1위 축척)
  점수 경쟁 = 다른 가설 중 최고 merit − 1위 merit ≤ Δ (다른 가설이 없으면 점수로는 확신)
  1위 이동 = 직전 1위(처음은 E2 정지 Top-1)와 층이 다르거나 같은 층 5 m 넘게 옮김, 센서 정상일 때만. 그때부터 K초 동안 애매
  센서 이상 = 위치추정 오돔으로 매초 직접 계산(1초 전 대비 속도 > 2 m/s 또는 yaw 속도 > 120°/s, make_gate_json.py 와 같은 기준)
  T 계산은 장면 시각 start_sec + T + off 부터 쓴다(인과). 그 전(출발 직후)은 E2 후보뿐이라 애매
정답 파일은 읽지 않는다(과제 표의 층은 어느 녹화·오돔을 쓸지 고르는 데만 씀).
출력: <out>/<출발점>.json(이벤트 실행기 --gate 형식, 키 = 장면 시각) + <out>/<출발점>_가설.json(P_INFORMATIVE 용, 계산마다 애매 가설)
2026-10-11 독립 검토 반영: 시각 off 맞춤, E2 1위부터 이동 셈, 센서 정상일 때만 이동, 경쟁 없음 = 확신, 센서 이상 직접 계산, 층 지도 다 있는지 확인.
    python3 scripts/e4_unknown_gate.py --seg /data2/e4_unknown_20261011 --out /data2/e4_unknown_20261011/gate_d0.2 --delta 0.2 --k 10
"""
import argparse, csv, json, math, os, sys
from pathlib import Path
import numpy as np

VAR = os.environ.get("E4_VAR", "")              # "merged" = E3 합치기(E2 후보 + 노드 재탐색) 3초 후보(2026-10-12)
SEEDED = {"merged": Path("/data2/e3_merged_3s_20261012"), "v8": Path("/data2/e3_merged_3s_v8_20261012")}.get(VAR, Path("/data2/e3_seeded_3s_20261011"))   # v8 = 지도 8차(2026-10-12)
SFX = {"merged": "_merged", "v8": "_v8"}.get(VAR, "")
E2TOP = Path("reports/_01_E2_초기위치/_작업자료/E2_dense_T0_nospace_정밀검증/E2_top100_candidates.csv")
SENSOR_MAX_SPEED, SENSOR_MAX_YAW_RATE = 2.0, 120.0
wrap = lambda a: (a + 180) % 360 - 180


def data_dir(F):
    return Path("/data/kci_work/a131_full_v0_4") if F == "4F" else Path("/data2/loc_inputs_20261005") / F


def hypotheses(J, T):
    """e3_seeded.score 와 같은 거름: 층마다 벽 관통 0.3 m 안(전부 넘으면 가장 덜 뚫은 것), merit 순, 같은 자세(1 m·10°·1.1배) 합치기."""
    feas = []
    for j in J:
        o = j["out"].get(str(T))
        if not o:
            continue
        C = [r for r in o["rows"] if r["ok"]]
        if C:
            feas += [c for c in C if c["X"] <= 0.3] or [c for c in C if c["X"] <= min(x["X"] for x in C) + 1e-9]
    feas.sort(key=lambda c: c["merit"]); P = []
    for c in feas:
        if any(c["m"] == p["m"] and math.hypot(c["x"] - p["x"], c["y"] - p["y"]) / math.sqrt(c["s"] * p["s"]) < 1.0 and abs(wrap(c["hd"] - p["hd"])) < 10
               and abs(math.log(c["s"] / p["s"])) < math.log(1.1) for p in P):
            continue
        P.append(c)
    return P


def distinct(a, b):
    return a["m"] != b["m"] or math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / a["s"] > 5.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seg", required=True, help="e4_unknown_segments.py 출력 폴더")
    ap.add_argument("--out", required=True); ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--k", type=float, default=10.0); ap.add_argument("--dur", type=int, default=60); ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); seg = Path(a.seg)
    segs = [s for s in csv.DictReader(open(seg / "segments.csv")) if not a.tasks or s["task_id"] in a.tasks.split(",")]
    e2top = {}
    for r in csv.DictReader(open(E2TOP)):
        e2top.setdefault(r["trial_id"], []).append(r)
    odom_cache, summ = {}, []
    for sg in segs:
        k, F, ts, off = sg["task_id"], sg["floor"], float(sg["start_sec"]), float(sg["off"])
        f = out / f"{k}.json"
        if f.exists():
            sys.exit(f"{f} 이 이미 있다(덮어쓰지 않음)")
        need = {int(r["floor_id"]) for r in e2top[k]}
        files = {int(p.stem): p for p in (SEEDED / k).glob("*.json")}
        assert (set(files) >= need) if VAR in ("merged", "v8") else (set(files) == need), f"{k}: 연결판 층 지도 결과 {sorted(files)} / E2 후보 층 {sorted(need)}"
        J = [json.load(open(p)) for p in files.values()]
        if F not in odom_cache:
            rows_o = list(csv.DictReader(open(data_dir(F) / "odom_link.csv")))
            odom_cache[F] = (np.array([float(r["t_sec"]) for r in rows_o]), np.array([float(r["fastlio_x"]) for r in rows_o]),
                             np.array([float(r["fastlio_y"]) for r in rows_o]), np.array([float(r["fastlio_yaw_rad"]) for r in rows_o]))
        OT, OX, OY, OYAW = odom_cache[F]; org = float(sg["odom_t0"]) - off      # 장면 시각 0 의 절대 시각

        def sensor_bad(tt):
            i1 = int(np.argmin(np.abs(OT - (org + tt)))); i0 = int(np.argmin(np.abs(OT - (org + tt - 1.0))))
            dt = OT[i1] - OT[i0]
            if dt <= 0:
                return False
            sp = math.hypot(OX[i1] - OX[i0], OY[i1] - OY[i0]) / dt
            yr = abs(math.degrees(math.atan2(math.sin(OYAW[i1] - OYAW[i0]), math.cos(OYAW[i1] - OYAW[i0])))) / dt
            return bool(sp > SENSOR_MAX_SPEED or yr > SENSOR_MAX_YAW_RATE)

        r1 = min(e2top[k], key=lambda r: int(r["rank"]))
        prev = dict(m=int(r1["floor_id"]), x=float(r1["x_px"]), y=float(r1["y_px"]), hd=float(r1["heading_deg"]), s=float(r1["ppm"]), merit=None, src="E2")
        comp, t_sw = {}, None
        for T in range(a.step, a.dur + 1, a.step):
            P = hypotheses(J, T)
            if not P:
                continue
            avail = ts + T + off
            top = P[0]; alt = next((c for c in P[1:] if distinct(top, c)), None)
            gap = alt["merit"] - top["merit"] if alt else None
            sw = distinct(prev, top) and not sensor_bad(avail)
            if sw:
                t_sw = avail
            near = [dict(m=c["m"], x=c["x"], y=c["y"], hd=c["hd"], s=c["s"], merit=c["merit"]) for c in P if c["merit"] <= top["merit"] + a.delta]
            if sw and not any(not distinct(prev, c) for c in near):   # 이동 직후면 직전 1위도 가설에 넣는다
                near.append(dict(m=prev["m"], x=prev["x"], y=prev["y"], hd=prev["hd"], s=prev["s"], merit=prev.get("merit"), prev_top=True))
            comp[avail] = dict(T=T, gap=gap, sw=sw, t_sw=t_sw, near=near, top=dict(m=top["m"], x=top["x"], y=top["y"], hd=top["hd"], s=top["s"]))
            prev = dict(m=top["m"], x=top["x"], y=top["y"], hd=top["hd"], s=top["s"], merit=top["merit"])
        scenes = [float(r[0]) for r in csv.reader(open(seg / "frames" / f"{k}.csv"))]
        keys = sorted(comp); rows, hyp, used = [], [], set()
        for tt in scenes:
            sb = sensor_bad(tt)
            last = max((x for x in keys if x <= tt + 1e-6), default=None)
            if last is None:                                          # 첫 계산 전: E2 후보뿐 → 애매
                rows.append(dict(t=tt, margin=0.0, amb=True, jump=False, spread=0.0, npk=0, turn=0.0, sensor_bad=sb, amb_src="start")); continue
            c = comp[last]
            by_score = c["gap"] is not None and c["gap"] <= a.delta
            by_sw = c["t_sw"] is not None and tt - c["t_sw"] < a.k
            first = last not in used; used.add(last)
            rows.append(dict(t=tt, margin=round(c["gap"], 4) if c["gap"] is not None else 99.0, amb=bool(by_score or by_sw),
                             jump=bool(c["sw"] and first), spread=0.0, npk=len(c["near"]), turn=0.0, sensor_bad=sb,
                             amb_src="score" if by_score else ("switch" if by_sw else ""), floors=sorted({h["m"] for h in c["near"]}), top_m=c["top"]["m"], T=c["T"]))
            if first:
                hyp.append(dict(t=tt, avail=last, T=c["T"], near=c["near"], top=c["top"]))
        json.dump(rows, open(f, "w"), ensure_ascii=False)
        json.dump(hyp, open(out / f"{k}_가설.json", "w"), ensure_ascii=False)
        n = len(rows); summ.append((k, sum(r["amb"] for r in rows) / max(n, 1), sum(r["amb_src"] == "switch" for r in rows), sum(r["sensor_bad"] for r in rows), n))
    print(f"출발점 {len(summ)}곳, Δ={a.delta}, K={a.k}초 | 애매 비율 평균 {np.mean([s[1] for s in summ]) * 100:.1f}% "
          f"(중앙 {np.median([s[1] for s in summ]) * 100:.0f}%), 급변 뒤로만 애매 {sum(s[2] for s in summ)}초, 센서 이상 {sum(s[3] for s in summ)}초, "
          f"장면 {min(s[4] for s in summ)}~{max(s[4] for s in summ)}개")
    print("애매 비율 낮은 5곳:", [(s[0], round(s[1], 2)) for s in sorted(summ, key=lambda s: s[1])[:5]])


if __name__ == "__main__":
    main()
