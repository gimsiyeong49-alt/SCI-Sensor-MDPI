#!/usr/bin/env python3
"""E4 '기하 부족' 판정용: 층마다 a1_3_1 표본(위치추정 입력, /data2/loc_inputs_20261005/<층>)의 쓰는 칸 수(role != 4)와
벽 방향 다양성(normal_diversity)을 위치추정 시각축(첫 표본 기준 초, loc_topk_series.py relative_sec 와 같음)으로 센다(2026-10-07).
normal_diversity = 쓰는 칸 벽 법선(normal_axial_angle_rad, 축 방향)의 2D 구조 텐서 작은/큰 고유값 비(가중치 normal_concentration).
0 이면 벽이 한 방향뿐(긴 직선 복도: 복도 방향 위치가 정해지지 않음), 1 에 가까우면 여러 방향.
    python3 scripts/e4_geom_cells.py --out <csv 폴더>
"""
import argparse, csv
from multiprocessing import Pool
from pathlib import Path
import numpy as np
B = Path("/data2/loc_inputs_20261005")
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
DIR = {F: B / F for F in FL}
DIR["4F"] = Path("/data/kci_work/a131_full_v0_4")    # 4층 위치추정 입력(35_ 실행과 같음)


def count(args):
    F, rows = args
    out = []
    t0 = float(rows[0]["t_sec"])
    for r in rows:
        p = DIR[F] / "samples" / r["sample_id"] / "data_short.npz"
        try:
            with np.load(p) as z:
                role = z["a1_3_1_all_candidate_role"]; ids = z["a1_3_1_all_candidate_flat_cell_id"]
                th = z["normal_axial_angle_rad"].reshape(-1)[ids]; wc = z["normal_concentration"].reshape(-1)[ids]
            n_all, n_use = int(len(role)), int(np.sum(role != 4))
            m = (role != 4) & np.isfinite(th) & np.isfinite(wc) & (wc > 0)
            if m.sum() >= 10:
                c, s_, w = np.cos(th[m]), np.sin(th[m]), wc[m]
                T = np.array([[np.sum(w * c * c), np.sum(w * c * s_)], [np.sum(w * c * s_), np.sum(w * s_ * s_)]]) / np.sum(w)
                ev = np.linalg.eigvalsh(T); div = float(ev[0] / max(ev[1], 1e-12))
            else:
                div = float("nan")
        except Exception:
            n_all = n_use = -1; div = float("nan")
        out.append((round(float(r["t_sec"]) - t0, 3), r["sample_id"], n_all, n_use, round(div, 4)))
    return F, out


ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
jobs = [(F, list(csv.DictReader(open(DIR[F] / "odom_link.csv")))) for F in FL]
with Pool(8) as pool:
    for F, out in pool.imap_unordered(count, jobs):
        with open(Path(a.out) / f"geom_cells_{F}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["relative_sec", "sample_id", "cells_all", "cells_used", "normal_diversity"]); w.writerows(out)
        u = np.array([x[3] for x in out if x[3] >= 0]); dv = np.array([x[4] for x in out if np.isfinite(x[4])])
        print(F, len(out), "쓰는 칸 p1/중앙", np.percentile(u, [1, 50]).round().tolist(), "| 벽 방향 다양성 p5/p25/중앙", np.percentile(dv, [5, 25, 50]).round(3).tolist(),
              "| <0.1 비율", round(float(np.mean(dv < 0.1)), 3), flush=True)
