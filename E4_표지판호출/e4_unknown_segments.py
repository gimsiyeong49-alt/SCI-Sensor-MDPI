#!/usr/bin/env python3
"""층 모름 E4 출발점 구간 정의(2026-10-11). 애매 기록(e4_unknown_gate.py)과 구간 실행(run_e4_unknown_20261011.sh)이 같이 쓴다.
E2·E3 시각(start_sec, T)은 위치추정 오돔 첫 샘플(L.T[0]) 기준이고, 이벤트 실행기 장면 시각은 녹화 기준(4층은 헤더 첫 시각, 나머지는 bag 시작)이다.
층마다 차이 off = L.T[0] − 녹화 기준(0.3~0.8초)을 재서
  구간(장면 시각) = [start_sec + off, start_sec + off + 60], 장면 = 그 안의 정수 초
  연결판 T 계산은 장면 시각 start_sec + T + off 부터 쓸 수 있다(라이다를 미리 쓰지 않음)
로 맞춘다. 정답은 읽지 않는다. 결과: <out>/segments.csv, <out>/frames/<출발점>.csv(그 구간 장면만)
    python3 scripts/e4_unknown_segments.py --out /data2/e4_unknown_20261011
"""
import argparse, csv
from pathlib import Path

TASKS = Path("reports/_01_E2_초기위치/_작업자료/tasks/E2_tasks.csv")
C = Path("/data3/kci_datasets/depth_cache_sign_eval")
FR4 = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/12_4층전체주행/frames_4F_full_1s.csv")


def data_dir(F):
    return Path("/data/kci_work/a131_full_v0_4") if F == "4F" else Path("/data2/loc_inputs_20261005") / F


def origin(F):
    if F == "4F":                                    # 4층 실행기는 헤더 시각 기준(실시간 대조 run_live_parity_C 의 T0 와 같음)
        r = next(csv.DictReader(open("/data2/kci_datasets/4F_RGB_FULL_V0_1/index.csv")))
        return int(r["header_timestamp_ns"]) / 1e9 - float(r["header_relative_sec"])
    r = next(csv.DictReader(open(f"/data3/kci_datasets/{F}_RGB_FULL_5HZ_V0_1/index.csv")))
    return int(r["bag_timestamp_ns"]) / 1e9 - float(r["bag_relative_sec"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--dur", type=float, default=60.0)
    a = ap.parse_args(); out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    rows, offs = [], {}
    for t in csv.DictReader(open(TASKS)):
        k, F, ts = t["task_id"], t["floor"], float(t["start_sec"])
        if F not in offs:
            t0_odom = float(next(csv.DictReader(open(data_dir(F) / "odom_link.csv")))["t_sec"])
            offs[F] = (t0_odom - origin(F), t0_odom)
        off = offs[F][0]; lo, hi = ts + off, ts + off + a.dur
        fr = FR4 if F == "4F" else C / f"{F}_full_frames.csv"
        sc = [r for r in csv.reader(open(fr)) if lo - 1e-6 <= float(r[0]) <= hi + 1e-6]
        f = out / "frames" / f"{k}.csv"
        if not f.exists():
            csv.writer(open(f, "w", newline="")).writerows(sc)
        rows.append(dict(task_id=k, floor=F, building=t["building"], start_sec=ts, off=round(off, 4), odom_t0=offs[F][1],
                         seg_lo=round(lo, 4), seg_hi=round(hi, 4), n_scenes=len(sc), first_scene=sc[0][0] if sc else "", last_scene=sc[-1][0] if sc else ""))
    with open(out / "segments.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"출발점 {len(rows)}곳 | 층별 off(초): " + ", ".join(f"{F} {v[0]:.2f}" for F, v in offs.items()))
    print("장면 수 최소·최대:", min(r["n_scenes"] for r in rows), max(r["n_scenes"] for r in rows))


if __name__ == "__main__":
    main()
