#!/usr/bin/env python3
"""E2 전 후보 점수판(2026-10-09). 출발점(과제) 하나에서 그 건물 모든 층 × 갈 수 있는 모든 격자 칸 × 방향 후보 × 축척 목록의 점수를 낸다.
협업자 위치추정(로봇 패키지 rt_localizer.RTLocalizer, lam_sem 0, w_nogo 3)을 고치지 않고 부른다. 정답은 읽지 않는다(채점은 e2_dense_score.py).
  과제 출발 t_s: 층 지도마다 새로 bootstrap(t_s, 15초)으로 방향 후보 4개와 센서 각도(BETA)를 정한다(출발 전 기록 안 씀)
  축척: 협업자 축척 목록(run_floor_id.SCALES 15~43 px/m 7개, 소융대는 지도 해상도 비율로 옮김) 각각 set_scale
  점수: localize(t_s + T, W=T, dt=3) 의 방향별 점수판 P(갈 수 있는 칸만, Z.allow).
  T = 0: 제자리 1회 관측(E2, 2026-10-09 사용자 정의) — 출발 시각 구조점 표본 1개만, 이동 누적 없음. 방향 후보·센서 각도도 그 표본으로.
  T > 0: 출발 뒤 T초 걸으며 누적(E3 조건)
  저장(/data2): 과제마다 npz — floor, scale_idx, phi_idx, cell(격자 번호), score(float32) 와 층별 격자 좌표·BETA·방향 목록
  축척 조건(--scale-mode, 2026-10-10): list = 목록 7개(기본) · fixed = 협업자 기본 축척 하나(run_floor_id.MID 24 px/m, 소융대는 지도 비율로 옮김)
    · oracle = 정답 축척 하나(참고 상한, 정답 ppm 을 읽으므로 실제 방법이 아님; 다른 층은 목록 비율로 옮김 → 정답 층이 유리)
    · oracle_all = 층마다 자기 정답 축척(공정한 참고 상한, 정보관 45과제만). 저장 폴더 T{T}_{mode}
  장면 조건(--scene, 2026-10-10): lidar = 지금 협업자 기본(SEMLOC_SCENE=lidar) · depth = 깊이 장면(B), tasks/E2_scene_labels.csv 의 depth_scene
    (e2_scene_labels.py 가 협업자 깊이 분류식을 미리 뽑아둔 깊이 영상에 적용한 값)을 L.depth_scene 자리에 넣는다. 협업자 코드는 고치지 않는다. 저장 폴더 끝에 _depth
  공간 조건(--space, 2026-10-10): on = 협업자 기본(갈 수 없는 칸 TRAV_W·빈 공간 FREE_W·금지 구역 w_nogo 감점) · off = 셋 다 0(벽 맞추기+장면 사전만).
    실행 중 모듈 상수·인스턴스 값만 바꾼다(원본 코드 그대로). 후보 칸(allow: 지도 안 갈 수 있는 칸, 계단 제외)은 두 조건 공통. 저장 폴더 끝에 _nospace
    지침서 조건: A = lidar·off, B = depth·off, C = depth·on (참고: lidar·on = 협업자 실제 설정)
    python3 scripts/e2_dense.py [--tasks 4F_T1,...] [--T 20] [--workers 20] [--scale-mode list|fixed|oracle] [--scene lidar|depth] [--space on|off]
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

PKG = Path("/data/kci_work/floor_id_pkg/floor_id")
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "localizer")); sys.path.insert(0, str(Path(__file__).parent))
G = Path("reports/_01_E2_초기위치/_작업자료"); GTDIR = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/36_채점용정답")
OUT = Path("/data2/e2_dense_20261009")
DATA = {"4F": "/data/kci_work/a131_full_v0_4"}
_ST = {}
BETA_MOUNT_DEG = 228.0   # 라이다 장착 각도(odom yaw 기준). 채점용 정답 summary 의 beta_deg 와 같은 로봇 보정값, 위치 정보 아님


def init():
    _ST.update(cache={})


def oracle_ppm(F):
    """참고 상한 조건 전용: 채점용 정답의 축척(px/m). 후보 생성에 정답을 쓰는 유일한 곳."""
    return float(json.load(open(GTDIR / F / "summary.json"))["ppm"])


def job(arg):
    tid, F, building, t_s, m, T, mode, m_true, scene, space = arg[:10]; vlm = arg[10] if len(arg) > 10 else None
    import floor_id as FI
    if _ST.get("building") != building:
        cfg = FI.load_cfg(); b = FI.setup_env(cfg, building); _ST.update(building=building, b=b, cache={})
    import rt_localizer as RT, run_floor_id as R
    from metric_fit import set_scale
    key = (F, m)
    if key not in _ST["cache"]:
        if len(_ST["cache"]) > 3:
            _ST["cache"].clear()
        _ST["cache"][key] = RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=m, ppm=R.MID, data_dir=DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)
    Z = _ST["cache"][key]; L = Z.L
    import node_local as NL
    if "TRAV_W0" not in _ST:
        _ST.update(TRAV_W0=NL.TRAV_W, FREE_W0=NL.FREE_W)
    NL.TRAV_W, NL.FREE_W = (_ST["TRAV_W0"], _ST["FREE_W0"]) if space == "on" else (0.0, 0.0)
    Z.w_nogo = 3.0 if space == "on" else 0.0
    if vlm is not None:                              # 이미지(VLM) 점수: 협업자 lam_sem 경로를 그대로 탄다(관측·가중치만 넣음)
        lam, ob, cal = vlm
        Z.lam_sem = lam; Z.obs = {f"{t_s:g}": ob}; Z.okeys = np.array([t_s]); Z.cal_past = lambda *aa, **kk: cal
    else:
        Z.lam_sem = 0.0
    if scene is not None:                            # 깊이 장면(B): 이 과제의 깊이 라벨을 장면 자리에 (인스턴스 속성만, 원본 코드 그대로)
        L.depth_scene = lambda tsec, _s=scene: _s
    elif "depth_scene" in L.__dict__:
        del L.depth_scene
    allsc = FI.map_scales(_ST["b"], R, int(_ST["b"]["n_maps"])); scales = allsc[m]
    k_mid = [float(v) for v in R.SCALES].index(float(R.MID))
    if mode == "fixed":
        scales = [scales[k_mid]]
    elif mode == "oracle":
        scales = [oracle_ppm(F) * allsc[m][k_mid] / allsc[m_true][k_mid]]
    elif mode == "oracle_all":                       # 공정한 참고 상한: 층마다 그 층 자기 정답 축척(정보관만, 9개 층 모두 녹화·정답 있음)
        if building != "kci":
            raise RuntimeError("oracle_all 은 층마다 정답 축척이 있는 정보관만")
        scales = [oracle_ppm(f"{m}F")]
    t0 = time.time()
    static = T <= 0                                  # 제자리 1회 관측(E2): 출발 시각 표본 하나로 방향 후보·점수
    if static:                                       # 움직임이 없으면 센서 장착 각도를 추정할 수 없다 → 로봇 장착 보정값(13개 층 공통 228°, 미러 1)
        Z.BETA, Z.MIRROR = math.radians(BETA_MOUNT_DEG), 1
    else:
        Z.BETA = None                                # 걸을 때는 과제마다 출발 뒤 15초로 새로(인과)
    modes, _ = Z.bootstrap(L.T[0] + t_s, boot_s=0.5 if static else 15.0)
    allow = Z.allow.reshape(L.shp); idx = np.flatnonzero(allow.ravel())
    out = dict(m=m, beta=float(Z.BETA), mirror=int(Z.MIRROR), modes=[float(p) for p in modes], scales=[float(s) for s in scales],
               cells=idx.astype(np.int32), cell_px=L.Pf[idx].astype(np.float32), parts=[])
    for si, s in enumerate(scales):
        set_scale(Z, s)
        r = Z.localize(L.T[0] + t_s + T, W=0.5 if static else T, dt=3.0, phis=modes)
        if len(r["frames"]) != (1 if static else len(r["frames"])):
            raise RuntimeError(f"제자리 조건인데 표본 {len(r['frames'])}개")
        for pi, p in enumerate(modes):
            out["parts"].append((si, pi, np.asarray(r["P"][p]).ravel()[idx].astype(np.float32)))
    set_scale(Z, R.MID)
    out["sec"] = round(time.time() - t0, 1); out["tid"] = tid
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=""); ap.add_argument("--T", type=float, default=20.0); ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--scale-mode", default="list", choices=("list", "fixed", "oracle", "oracle_all"))
    ap.add_argument("--scene", default="lidar", choices=("lidar", "depth"))
    ap.add_argument("--space", default="on", choices=("on", "off"))
    ap.add_argument("--vlm", type=float, default=0.0, help="이미지(VLM) 점수 가중치 lam_sem(협업자 기본 0.5). 0 이면 끔. 저장 폴더 끝에 _vlm<값>")
    ap.add_argument("--vlm-obs", default="reports/_01_E2_초기위치/_작업자료/vlm/obs_starts.json")
    ap.add_argument("--vlm-cal", default="reports/_01_E2_초기위치/_작업자료/vlm/obs_cal.json", help="단서별 가중치를 정할 화면(13개 층 고르게, 정답 안 씀)")
    a = ap.parse_args()
    tasks = list(csv.DictReader(open(G / "tasks/E2_tasks.csv")))
    if a.tasks:
        tasks = [t for t in tasks if t["task_id"] in a.tasks.split(",")]
    od = OUT / (f"T{a.T:g}" + ("" if a.scale_mode == "list" else f"_{a.scale_mode}") + ("_depth" if a.scene == "depth" else "") + ("_nospace" if a.space == "off" else "") + (f"_vlm{a.vlm:g}" if a.vlm else "")); od.mkdir(parents=True, exist_ok=True)
    vlmd = {}
    if a.vlm:
        from semloc import CueCalibration
        obs = json.load(open(a.vlm_obs)); calo = json.load(open(a.vlm_cal)); cal = CueCalibration(calo, list(calo), mode="raw")
        print("단서 가중치", {k: round(v, 2) for k, v in cal.w.items()}, flush=True)
        vlmd = {t["task_id"]: (a.vlm, obs[t["task_id"]], cal) for t in tasks}
    dscn = {r["task_id"]: r["depth_scene"] for r in csv.DictReader(open(G / "tasks/E2_scene_labels.csv"))} if a.scene == "depth" else {}
    mtrue = {}
    if a.scale_mode == "oracle":
        mtrue = {r["task_id"]: int(r["true_map"]) for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    nmap = {"kci": 9, "gong": 7, "sw": 8}
    jobs = [(t["task_id"], t["floor"], t["building"], float(t["start_sec"]), m, a.T, a.scale_mode, mtrue.get(t["task_id"], m), dscn.get(t["task_id"]), a.space, vlmd.get(t["task_id"])) for t in tasks if not (od / f"{t['task_id']}.npz").exists()
            for m in range(1, nmap[t["building"]] + 1)]
    jobs.sort(key=lambda j: (j[2], j[1], j[4]))          # 같은 지도끼리 이어서(캐시)
    print(f"과제 {len({j[0] for j in jobs})}개, 작업 {len(jobs)}개", flush=True)
    t0 = time.time(); got = {}
    with Pool(a.workers, initializer=init) as pool:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            got.setdefault(r["tid"], []).append(r)
            t = [x for x in tasks if x["task_id"] == r["tid"]][0]
            if len(got[r["tid"]]) == nmap[t["building"]]:
                R_ = sorted(got.pop(r["tid"]), key=lambda x: x["m"])
                fl, sc, ph, ce, scs = [], [], [], [], []
                for x in R_:
                    for si, pi, v in x["parts"]:
                        fl.append(np.full(len(v), x["m"], np.uint8)); sc.append(np.full(len(v), si, np.uint8)); ph.append(np.full(len(v), pi, np.uint8))
                        ce.append(np.arange(len(v), dtype=np.int32)); scs.append(v)
                meta = dict(task=r["tid"], T=a.T, scale_mode=a.scale_mode, scene=a.scene, space=a.space, vlm=a.vlm, maps={x["m"]: dict(beta=x["beta"], mirror=x["mirror"], modes=x["modes"], scales=x["scales"], sec=x["sec"]) for x in R_})
                np.savez_compressed(od / f"{r['tid']}.npz", floor=np.concatenate(fl), scale_idx=np.concatenate(sc), phi_idx=np.concatenate(ph),
                                    cell=np.concatenate(ce), score=np.concatenate(scs), meta=json.dumps(meta),
                                    **{f"cellpx_{x['m']}": x["cell_px"] for x in R_})
                print(f"  {r['tid']} 저장 후보 {sum(len(v) for v in scs):,}개 ({time.time() - t0:.0f}s)", flush=True)
    print(f"끝 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
