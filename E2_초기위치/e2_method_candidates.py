#!/usr/bin/env python3
"""E2 새 구성(2026-10-10): 거리 비교·정밀 검증으로 다시 매긴 Top-100 후보를 그림 스크립트가 읽는 형식으로 쓴다(평가 결과 변환만, 계산 없음).
입력: _작업자료/미세정합시험/<이름>/<과제>.json (e2_refine_test.py --save-top: rank_tw0·rank_merit, top100.tw0·top100.merit)
출력: _작업자료/E2_dense_T0<접미사>_거리비교/, _정밀검증/ 에 E2_top100_per_trial.csv(true_rank, floor_rank, top1_floor)·E2_top100_candidates.csv
  점수(score): 거리비교 = 양방향 점수(클수록 좋음), 정밀검증 = merit(작을수록 좋음)
    python3 scripts/e2_method_candidates.py --name T0_nospace_후보 --suffix _nospace
"""
import argparse, csv, json
from pathlib import Path

W = Path("reports/_01_E2_초기위치/_작업자료")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--suffix", required=True); a = ap.parse_args()
    J = {json.load(open(f))["task"]: json.load(open(f)) for f in sorted((W / "미세정합시험" / a.name).glob("*.json"))}
    for key, label in (("tw0", "거리비교"), ("merit", "정밀검증")):
        od = W / f"E2_dense_T0{a.suffix}_{label}"; od.mkdir(parents=True, exist_ok=True)
        per, cand = [], []
        for k, j in J.items():
            tops = j["top100"][key]
            per.append(dict(trial_id=k, true_rank=j[f"rank_{key}"] if j[f"rank_{key}"] else ">1000", floor_rank=j[f"floor_rank_{key}"] if j[f"floor_rank_{key}"] else ">1000",
                            top1_floor=tops[0][0] if tops else "", n_eff="", n_eff_floors=""))
            for i, q in enumerate(tops, 1):
                m, x, y, hd, s = q[:5]
                cand.append(dict(trial_id=k, rank=i, floor_id=m, x_px=x, y_px=y, heading_deg=hd, ppm=s, score=q[5] if len(q) > 5 else ""))
        for name, rows in (("E2_top100_per_trial.csv", per), ("E2_top100_candidates.csv", cand)):
            with open(od / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(od, len(per), "과제", len(cand), "후보")


if __name__ == "__main__":
    main()
