#!/usr/bin/env python3
"""E2 가벼운 시험: 기존 상위 후보를 협업자 양방향 점수(twoway.twoway_score)로 다시 매기면 Top-K 가 오르나(2026-10-10).
협업자 floor_id.py 가 층 순위를 정할 때 쓰는 그 점수다: 후보 자세에서 36방향으로 지도상 첫 벽 거리와 실제 스캔 거리를 비교하고,
지도엔 벽이 있는데 스캔엔 없거나(지도 벽이 설명 안 됨) 그 반대면 감점. 협업자 코드는 고치지 않고 부른다. 정답은 채점에만 쓴다.
  순위 1: 양방향 점수만(협업자 층 판별 방식)    순위 2: 기존 점수 + 양방향 점수(같은 가중, 조정 안 함)
    python3 scripts/e2_twoway_test.py [--src T0_depth_nospace] [--top 1000]
결과: reports/_01_E2_초기위치/_작업자료/양방향시험/<src>.csv, 요약 출력
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_dense as ED  # noqa: E402
from e2_dense_score import distinct  # noqa: E402

G = Path("reports/_01_E2_초기위치/_작업자료"); R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/36_채점용정답")
OUT = G / "양방향시험"; wrap = lambda a: (a + 180) % 360 - 180
_W = {}


def walls(building, m, F):
    """그 건물 m층 지도의 벽(협업자 FloorplanLocalizer 가 정리한 wall_real). 데이터 폴더는 벽과 무관해 아무 층이나."""
    if (building, m) not in _W:
        import floor_id as FI
        if ED._ST.get("building") != building:
            cfg = FI.load_cfg(); b = FI.setup_env(cfg, building); ED._ST.update(building=building, b=b, cache={})
        import rt_localizer as RT, run_floor_id as Rr
        Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=Rr.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
        _W[(building, m)] = getattr(Z.L, "wall_real", Z.L.wall)
    return _W[(building, m)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default="T0_depth_nospace"); ap.add_argument("--top", type=int, default=1000); ap.add_argument("--tasks", default="")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    tasks = list(csv.DictReader(open(G / "tasks/E2_tasks.csv"))); gtt = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in a.tasks.split(",")]
    tasks.sort(key=lambda t: (t["building"], t["floor"]))
    rows = []
    for t in tasks:
        k, F, b = t["task_id"], t["floor"], t["building"]; m_true = int(gtt[k]["true_map"]); ts = float(t["start_sec"])
        d = dict(np.load(Path("/data2/e2_dense_20261009") / a.src / f"{k}.npz")); meta = json.loads(str(d["meta"]))
        acc = distinct(d, meta, a.top)
        # 그 순간 스캔(위치추정이 쓰는 그 표본, 전부)
        import floor_id as FI
        if ED._ST.get("building") != b:
            cfg = FI.load_cfg(); bb = FI.setup_env(cfg, b); ED._ST.update(building=b, b=bb, cache={})
        import rt_localizer as RT, run_floor_id as Rr, twoway as TW
        if (F, "scan") not in ED._ST["cache"]:
            ED._ST["cache"][(F, "scan")] = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=1, ppm=Rr.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
        L = ED._ST["cache"][(F, "scan")].L; pts, yaw = L._scan(L.T[0] + ts); pts = np.asarray(pts, float)
        tw = []
        for c in acc:
            d_obs = TW.sector_obs(pts, math.radians(c["hd"]) + yaw)        # 거울 1, 방향 = 후보 방향(장착각 포함) + 오돔 yaw
            tw.append(float(TW.twoway_score(walls(b, c["m"], F), c["px"][0], c["px"][1], c["s"], d_obs)[0]))
        tw = np.array(tw); sc = np.array([c["score"] for c in acc])
        # 채점(정답은 여기서만)
        S = json.load(open(R / F / "summary.json")); ppm = S["ppm"]; yaw_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
        gp = list(csv.DictReader(open(R / F / "warp_clean" / "gt_warp.csv"))); gt = np.array([float(x["relative_sec"]) for x in gp]); gx = np.array([[float(x["x_px"]), float(x["y_px"])] for x in gp])
        pg = gx[int(np.argmin(abs(gt - ts)))]
        ok = np.array([c["m"] == m_true and np.hypot(*(c["px"] - pg)) / ppm <= 3.0 and abs(wrap(c["hd"] - yaw_gt)) <= 30 for c in acc])
        fl = np.array([c["m"] == m_true for c in acc])
        def first(order, mask):
            hit = np.where(mask[order])[0]
            return int(hit[0]) + 1 if len(hit) else None
        o0 = np.arange(len(acc)); o1 = np.argsort(-tw, kind="stable"); o2 = np.argsort(-(sc + tw), kind="stable")
        rows.append(dict(trial_id=k, floor=F, start_type=t["start_type"], rank_orig=first(o0, ok), rank_twoway=first(o1, ok), rank_sum=first(o2, ok),
                         floor_rank_orig=first(o0, fl), floor_rank_twoway=first(o1, fl), floor_rank_sum=first(o2, fl), n=len(acc)))
        print(k, rows[-1]["rank_orig"], rows[-1]["rank_twoway"], rows[-1]["rank_sum"], flush=True)
    with open(OUT / f"{a.src}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    hit = lambda key, K: sum(1 for r in rows if r[key] and r[key] <= K)
    print(f"\n{a.src}, 과제 {len(rows)}개, 기존 상위 {a.top}개를 다시 매김")
    for nm, key, fk in (("기존 점수", "rank_orig", "floor_rank_orig"), ("양방향 점수만", "rank_twoway", "floor_rank_twoway"), ("기존 + 양방향", "rank_sum", "floor_rank_sum")):
        print(f"  {nm:12s} 정답 위치 Top-1 {hit(key, 1)} · Top-10 {hit(key, 10)} · Top-100 {hit(key, 100)} · 정답 층 Top-1 {hit(fk, 1)} · Top-100 {hit(fk, 100)}")


if __name__ == "__main__":
    main()
