#!/usr/bin/env python3
"""E2 Top-100 채점(평가 전용, 2026-10-09; 2026-10-10 정답 후보·정답 층 최고 후보의 위치·방향·축척 오차, 1위 층·축척, 계산 시간 열 추가). e2_dense.py 가 저장한 전 후보 점수판(모든 층 × 칸 × 방향 × 축척)에서
서로 다른 후보를 점수순으로 뽑는다: 같은 층에서 이미 뽑힌 후보와 위치 R_CL m·방향 YAW_CL° 안이면 같은 후보로 보고 건너뛴다(축척만 다른 것도 같은 후보).
정답 후보 = 정답 층 + 위치 r m + 방향 θ° 안(기본 3 m, 30°). 정답은 이 스크립트만 읽는다(tasks_gt_EVAL_ONLY.csv, 36_채점용정답).
  True Rank(서로 다른 후보 기준, 1000 위까지), Hit@1·3·5·10·20·50·100, 층 맞힘@K, 정답 후보의 축척 오차
    python3 scripts/e2_dense_score.py [--T 20] [--scale-mode list|fixed|oracle]
결과: _01_E2_초기위치/_작업자료/E2_dense_T20/E2_top100_per_trial.csv, E2_top100_candidates.csv(과제마다 상위 100), E2_top100_summary.md
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); G = Path("reports/_01_E2_초기위치/_작업자료"); SRC = Path("/data2/e2_dense_20261009")
KS = (1, 3, 5, 10, 20, 50, 100); R_CL, YAW_CL, MAXR = 2.0, 30.0, 1000
TYPE = {"S1": "긴 직선 복도", "S2": "꺾이는 곳", "S3": "갈림길", "S4": "계단·엘리베이터 앞", "S5": "막다른 곳"}
wrap = lambda d: (d + 180) % 360 - 180


def distinct(d, meta, limit):
    """점수순 서로 다른 후보 limit 개(층·위치·방향). 위치 거리는 그 후보의 축척으로 m 환산."""
    o = np.argsort(-d["score"]); acc = []
    for i in o:
        m = int(d["floor"][i]); mm = meta["maps"][str(m)]; s = mm["scales"][int(d["scale_idx"][i])]
        px = d[f"cellpx_{m}"][int(d["cell"][i])]; hd = mm["modes"][int(d["phi_idx"][i])] + math.degrees(mm["beta"])
        if any(a["m"] == m and np.hypot(*(a["px"] - px)) / min(a["s"], s) <= R_CL and abs(wrap(a["hd"] - hd)) <= YAW_CL for a in acc):
            continue
        acc.append(dict(m=m, px=px.astype(float), s=s, hd=hd, score=float(d["score"][i])))
        if len(acc) >= limit:
            break
    return acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--T", type=float, default=20.0); ap.add_argument("--r", type=float, default=3.0); ap.add_argument("--yaw", type=float, default=30.0)
    ap.add_argument("--scale-mode", default="list", help="e2_dense.py --scale-mode 와 같게(list/fixed/oracle)")
    ap.add_argument("--scene", default="lidar", help="e2_dense.py --scene 와 같게(lidar/depth)")
    ap.add_argument("--space", default="on", help="e2_dense.py --space 와 같게(on/off)")
    ap.add_argument("--src-dir", default="", help="점수판 폴더를 직접 지정(E3: /data2/e3_dense_20261010/<조건>_T<T>)")
    ap.add_argument("--res-dir", default="", help="결과 폴더를 직접 지정")
    ap.add_argument("--eff-margin", type=float, default=1.5, help="남은 후보 = 1위와 점수 차 × 6 / 관측 장수가 이 값 이하(기존 애매함 기준 1.5, 관측 6장 환산)")
    a = ap.parse_args(); sfx = ("" if a.scale_mode == "list" else f"_{a.scale_mode}") + ("_depth" if a.scene == "depth" else "") + ("_nospace" if a.space == "off" else "")
    src = Path(a.src_dir) if a.src_dir else SRC / f"T{a.T:g}{sfx}"; res = Path(a.res_dir) if a.res_dir else G / f"E2_dense_T{a.T:g}{sfx}"; res.mkdir(parents=True, exist_ok=True)
    tasks = list(csv.DictReader(open(G / "tasks/E2_tasks.csv"))); gtt = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    rows, top = [], []
    for t in tasks:
        p = src / f"{t['task_id']}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p)); meta = json.loads(str(d["meta"])); F = t["floor"]; m_true = int(gtt[t["task_id"]]["true_map"])
        S = json.load(open(R / "36_채점용정답" / F / "summary.json")); ppm_gt = S["ppm"]; yaw_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
        gp = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gtime = np.array([float(x["relative_sec"]) for x in gp]); gxy = np.array([[float(x["x_px"]), float(x["y_px"])] for x in gp])
        p_gt = gxy[int(np.argmin(abs(gtime - (float(t["start_sec"]) + a.T))))]
        acc = distinct(d, meta, MAXR)
        ok = lambda c: c["m"] == m_true and np.hypot(*(c["px"] - p_gt)) / ppm_gt <= a.r and abs(wrap(c["hd"] - yaw_gt)) <= a.yaw
        tr = next((i for i, c in enumerate(acc, 1) if ok(c)), None); fr = next((i for i, c in enumerate(acc, 1) if c["m"] == m_true), None)
        se = abs(acc[tr - 1]["s"] - ppm_gt) / ppm_gt * 100 if tr else None
        pe = lambda c: round(float(np.hypot(*(c["px"] - p_gt)) / ppm_gt), 2)            # 위치 오차(m, 정답 축척)
        ye = lambda c: round(abs(wrap(c["hd"] - yaw_gt)), 1)                             # 방향 오차(°)
        sp = lambda c: round(100 * (c["s"] - ppm_gt) / ppm_gt, 1)                       # 축척 오차(%, 부호)
        tc = acc[tr - 1] if tr else None; fc = acc[fr - 1] if fr else None
        nf = int(next(iter(meta["maps"].values())).get("n_frames", 1))             # 쌓인 관측 장수(E2 제자리 = 1)
        eff_m = a.eff_margin * nf / 6.0                                            # 기존 애매함 기준과 같게 관측 6장 기준으로 환산(loc_topk_series margin_per6)
        eff = [c for c in acc if c["score"] >= acc[0]["score"] - eff_m]           # 남은 후보(서로 다른 후보 중 1위와 점수 차가 기준 이하)
        n_fin = int(np.isfinite(d["score"]).sum())
        flags = f"scale={meta.get('scale_mode', 'list')};scene={meta.get('scene', 'lidar')};space={meta.get('space', 'on')}"
        sec = sum(float(v.get("sec", 0)) for v in meta["maps"].values())
        for i, c in enumerate(acc[:100], 1):
            top.append(dict(trial_id=t["task_id"], rank=i, floor_id=c["m"], x_px=round(c["px"][0], 1), y_px=round(c["px"][1], 1), heading_deg=round(c["hd"] % 360, 1),
                            ppm=c["s"], score=round(c["score"], 4), x_m=round(c["px"][0] / c["s"], 2), y_m=round(c["px"][1] / c["s"], 2),
                            yaw_rad=round(np.deg2rad(c["hd"] % 360), 4), cluster_id=i, source_flags=flags))
        rows.append(dict(experiment_id="E2", trial_id=t["task_id"], floor=F, start_type=t["start_type"], T_init_s=a.T, raw_candidates=len(d["score"]),
                         true_rank=tr if tr else f">{MAXR}", floor_rank=fr if fr else f">{MAXR}", **{f"hit_{k}": int(bool(tr) and tr <= k) for k in KS},
                         **{f"floor_recall_{k}": int(bool(fr) and fr <= k) for k in KS}, true_cand_scale_err_pct=round(se, 1) if se is not None else "",
                         true_cand_pos_err_m=pe(tc) if tc else "", true_cand_yaw_err_deg=ye(tc) if tc else "",
                         truefloor_best_pos_err_m=pe(fc) if fc else "", truefloor_best_yaw_err_deg=ye(fc) if fc else "",
                         truefloor_best_scale=fc["s"] if fc else "", truefloor_best_scale_err_pct=sp(fc) if fc else "", gt_scale=round(ppm_gt, 2),
                         top1_floor=acc[0]["m"], top1_scale=acc[0]["s"], compute_sec=round(sec, 1), source_flags=flags,
                         top1_pos_err_m=pe(acc[0]) if acc[0]["m"] == m_true else "", top1_yaw_err_deg=ye(acc[0]), top1_scale_err_pct=sp(acc[0]) if acc[0]["m"] == m_true else "",
                         n_eff=len(eff), n_eff_floors=len({c["m"] for c in eff}), true_in_eff=int(bool(tc) and any(c is tc for c in eff)),
                         truefloor_in_eff=int(any(c["m"] == m_true for c in eff)), n_raw_finite=n_fin, n_raw_removed=len(d["score"]) - n_fin, n_frames=nf, eff_margin_used=round(eff_m, 3),
                         split="PILOT", gt="36_채점용정답(LiDAR, 독립 아님)"))
        print(t["task_id"], "정답 순위", rows[-1]["true_rank"], "층 순위", rows[-1]["floor_rank"], flush=True)
    for name, rr in (("E2_top100_per_trial.csv", rows), ("E2_top100_candidates.csv", top)):
        with open(res / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    n = len(rows); avg = lambda k, rr=rows: 100 * np.mean([r[k] for r in rr])
    L = [f"E2 Top-100 (서로 다른 후보, 같은 층 {R_CL:g} m·{YAW_CL:g}° 안은 하나): 과제 {n}개, 출발 뒤 {a.T:g}초. 과제마다 모든 층 × 칸 × 방향 4 × 축척 7 원후보 중앙 "
         f"{np.median([r['raw_candidates'] for r in rows]):,.0f}개. 정답 후보 = 정답 층 + {a.r:g} m + {a.yaw:g}°. PILOT, 정답=LiDAR 채점용 정답.", "",
         "| K | " + " | ".join(str(k) for k in KS) + " |", "|---" * (len(KS) + 1) + "|",
         "| 층 맞힘@K | " + " | ".join(f"{avg(f'floor_recall_{k}'):.0f}%" for k in KS) + " |",
         "| Hit@K | " + " | ".join(f"{avg(f'hit_{k}'):.0f}%" for k in KS) + " |", ""]
    trs = [r["true_rank"] for r in rows if isinstance(r["true_rank"], int)]
    L.append(f"정답이 {MAXR}위 안에 있는 과제 {len(trs)}/{n}, 그 순위 중앙 {np.median(trs):.0f}" if trs else "")
    L += ["", "| 출발 유형 | 과제 | Hit@1 | Hit@10 | Hit@100 | 층 맞힘@1 |", "|---|---|---|---|---|---|"]
    for ty, nm in TYPE.items():
        rr = [r for r in rows if r["start_type"] == ty]
        if rr:
            L.append(f"| {nm} | {len(rr)} | {avg('hit_1', rr):.0f}% | {avg('hit_10', rr):.0f}% | {avg('hit_100', rr):.0f}% | {avg('floor_recall_1', rr):.0f}% |")
    L += ["", "| 층 | 정답 후보 순위(과제 5개) |", "|---|---|"]
    for F in dict.fromkeys(r["floor"] for r in rows):
        L.append(f"| {F} | " + ", ".join(str(r["true_rank"]) for r in rows if r["floor"] == F) + " |")
    (res / "E2_top100_summary.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
