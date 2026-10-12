#!/usr/bin/env python3
"""E3 최종 노드 그대로(2026-10-11): 로봇 실시간 노드(realtime/rt_live_node → floor_id 와 같은 알고리즘)를 출발점마다 막 켠 상태로 흉내낸다.
노드 실험 모드는 [시작] → 걷기 → [멈춤] 뒤 그동안의 창으로 floor_id 를 한 번 돌려 판정한다(인수인계 내용, 로봇 노드 코드는 로컬 사본이 없어 미대조).
출발점 t_s 마다 T = 15·30·45·60초 창으로 floor_id.py 본문과 같은 계산: metric_fit_v2.fit_window_v2(t_s+T, 축척 목록, W=T) → twoway, 층 순위 = twoway 내림차순.
실제 차이는 W=T 하나로, `floor_id.py --at t_s+T --window T` 와 같다. 출발 시각 bootstrap 줄은 결과에 들어가지 않는다
(fit_window_v2 가 창 [t−W, t] 에서 fit_beta·bootstrap 을 새로 하고 끝나면 되돌림, metric_fit_v2.py:66-70·174-176; 2026-10-11 교차 검증).
협업자 패키지 코드는 고치지 않고 부른다. 정답은 채점(--score)에만 쓴다.
저장: /data2/e3_node_20261011/<과제>/<층지도>.json, 채점 reports/_03_E3_걸으며좁히기/_작업자료/노드/
    python3 scripts/e3_node.py --tasks 4F_T1,S1F_T4 --workers 8        # 계산
    python3 scripts/e3_node.py --score                                  # 채점
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("/data2/e3_node_20261011"); G = Path("reports/_01_E2_초기위치/_작업자료"); TS = (15, 30, 45, 60)
RES = Path("reports/_03_E3_걸으며좁히기/_작업자료/노드")
NMAP = {"kci": 9, "gong": 7, "sw": 8}


def job(arg):
    tid, F, building, t_s, m, times = arg
    od = OUT / tid; f = od / f"{m}.json"
    old = json.load(open(f)) if f.exists() else None
    todo = [T for T in times if not old or str(T) not in old["rows"]]       # 이미 계산한 시점은 건너뜀(60초 먼저, 나머지 나중)
    if not todo:
        return tid, m, "있음"
    import e2_dense as ED
    sys.path.insert(0, "/data/kci_work/floor_id_pkg/floor_id")
    import floor_id as FI
    cfg = FI.load_cfg(); b = FI.setup_env(cfg, building)
    import rt_localizer as RT, run_floor_id as R, twoway as TW, metric_fit_v2 as V, metric_fit_negev as MN
    MN._door.clear()                                  # 문 지도 캐시가 층 번호로만 묶여 건물이 섞이지 않게(2026-10-10 겪은 문제)
    t0 = time.time(); DT = float(cfg["consts"]["dt_s"])
    Z = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
    L = Z.L; Z.bootstrap(L.T[0] + t_s, boot_s=15.0)                 # 노드: [시작] 시각에서 방향·센서 각도
    SC = FI.map_scales(b, R, int(b["n_maps"]))[m]; rows = dict(old["rows"]) if old else {}
    for T in todo:
        t = L.T[0] + t_s + T
        try:
            r = V.fit_window_v2(Z, t, SC, floor=m, W=float(T), dt=DT)
            pts, yaw = L._scan(t)
            d_obs = TW.sector_obs(np.asarray(pts, float) * [1, r["mirror"]], np.deg2rad(r["phi"]) + yaw + r["beta"])
            tw = float(TW.twoway_score(getattr(L, "wall_real", L.wall), *r["pos"], r["ppm"], d_obs)[0])
            rows[str(T)] = dict(map=m, twoway=tw, ppm=float(r["ppm"]), phi=float(r["phi"]), beta_deg=float(math.degrees(r["beta"])), mirror=int(r["mirror"]),
                           pos=[float(v) for v in r["pos"]], wall_cost_m=float(r["wall_cost_m"]), scene=r["scene"])
        except Exception as e:
            rows[str(T)] = dict(map=m, err=str(e)[:300])
    od.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(dict(task=tid, map=m, t_s=t_s, rows=rows, sec=round(time.time() - t0, 1) + (old["sec"] if old else 0))))
    return tid, m, round(time.time() - t0, 1)


def run(tasks, workers, times):
    T_ = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    jobs = [(k, T_[k]["floor"], T_[k]["building"], float(T_[k]["start_sec"]), m, times) for k in tasks for m in range(1, NMAP[T_[k]["building"]] + 1)]
    jobs.sort(key=lambda j: (j[2], j[1], j[0], j[4]))
    print("과제", len(tasks), "작업", len(jobs), flush=True); t0 = time.time()
    from multiprocessing import Pool
    with Pool(workers, maxtasksperchild=4) as pool:
        for tid, m, s in pool.imap_unordered(job, jobs):
            print(f"  {tid} {m}층 {s}s  ({time.time() - t0:.0f}s)", flush=True)


def score():
    """정답(채점용): 층 순위, 1위 층 후보가 정답 위치(정답 층 + 3 m + 30°)인가, 정답 층 후보가 3 m·30° 안인가."""
    import e2_visual_ko as EV
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    wrap = lambda a: (a + 180) % 360 - 180; rows = []
    for k, t in EV.TASK.items():
        d = OUT / k
        if not d.exists() or len(list(d.glob("*.json"))) < NMAP[t["building"]]:
            continue
        J = {int(f.stem): json.load(open(f)) for f in d.glob("*.json")}; m_true = int(EV.GT[k]["true_map"]); ts = float(t["start_sec"])
        _, ppm, hd_gt, gt, gx = EV.gt_of(k)
        for T in TS:
            if not all(str(T) in J[m]["rows"] for m in J):
                continue
            c = [J[m]["rows"][str(T)] for m in J if "twoway" in J[m]["rows"][str(T)]]
            order = sorted(c, key=lambda x: -x["twoway"]); pg = gx[int(np.argmin(abs(gt - (ts + T))))]
            def ok(x):
                return x["map"] == m_true and math.hypot(x["pos"][0] - pg[0], x["pos"][1] - pg[1]) / ppm <= 3.0 and abs(wrap(x["phi"] + x["beta_deg"] - hd_gt)) <= 30
            fr = next((i + 1 for i, x in enumerate(order) if x["map"] == m_true), None); tc = next((x for x in order if x["map"] == m_true), None)
            rows.append(dict(trial_id=k, floor=t["floor"], building=t["building"], start_type=t["start_type"], T=T, n_maps=len(order),
                             verdict=order[0]["map"] if order else "", floor_rank=fr if fr else "", top1_ok=int(bool(order) and ok(order[0])),
                             top3_ok=int(any(ok(x) for x in order[:3])), truefloor_ok=int(bool(tc) and ok(tc)),
                             truefloor_pos_err_m=round(math.hypot(tc["pos"][0] - pg[0], tc["pos"][1] - pg[1]) / ppm, 2) if tc else "",
                             truefloor_yaw_err_deg=round(abs(wrap(tc["phi"] + tc["beta_deg"] - hd_gt)), 1) if tc else "",
                             truefloor_scale_err_pct=round(100 * (tc["ppm"] - ppm) / ppm, 1) if tc else "",
                             truefloor_wall_cost_m=round(tc["wall_cost_m"], 3) if tc else "", top1_wall_cost_m=round(order[0]["wall_cost_m"], 3) if order else ""))
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / "노드_채점.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len({r["trial_id"] for r in rows}); print("출발점", n)
    for T in TS:
        rr = [r for r in rows if r["T"] == T]
        if not rr:
            continue
        print(f"  ({len(rr)}곳)", end="")
        print(f"  {T}초: 층 1위 {sum(r['floor_rank'] == 1 for r in rr)} · 층 상위3 {sum(1 for r in rr if r['floor_rank'] and r['floor_rank'] <= 3)} · "
              f"1위 위치 맞음 {sum(r['top1_ok'] for r in rr)} · 상위3 안 위치 {sum(r['top3_ok'] for r in rr)} · 정답 층 후보 위치 맞음 {sum(r['truefloor_ok'] for r in rr)}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--score", action="store_true")
    ap.add_argument("--times", default="15,30,45,60", help="계산할 시점(초). 이미 있는 시점은 건너뜀")
    a = ap.parse_args()
    if a.score:
        return score()
    tasks = a.tasks.split(",") if a.tasks else [t["task_id"] for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))]
    run(tasks, a.workers, [int(x) for x in a.times.split(",")])


if __name__ == "__main__":
    main()
