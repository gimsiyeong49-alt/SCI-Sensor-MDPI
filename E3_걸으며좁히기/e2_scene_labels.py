#!/usr/bin/env python3
"""E2 출발점 65곳의 장면 라벨(2026-10-10). 정답은 읽지 않는다.
  lidar: 지금 협업자 위치추정이 쓰는 라벨(node_local.lidar_scene, SEMLOC_SCENE=lidar)
  depth: 협업자 깊이 장면 분류(node_local.depth_scene 의 깊이 경로)와 같은 식을 미리 뽑아둔 깊이 영상에 적용
         - 깊이 영상: /data3/kci_datasets/depth_cache_sign_eval/<층>_full.npz (aligned_depth_to_color, 1초 간격, mm)
         - 협업자 코드는 과거 1.5초(0, 0.5, 1.0, 1.5초 전) 프레임 중앙값 → 여기서는 1초 간격이라 [t-1.5, t] 안 프레임 중앙값
         - 한 프레임: 가운데 1/3 높이 띠에서 앞(가운데 1/5 폭)·왼쪽(1/4)·오른쪽(1/4) 거리 중앙값
         - 앞 < 4.5 m 막다른 곳, (앞 > 12 이고 옆 < 7.5) 또는 옆 < 4.5 복도, 나머지 교차로
    python3 scripts/e2_scene_labels.py
결과: _01_E2_초기위치/_작업자료/tasks/E2_scene_labels.csv
"""
import csv, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_dense as ED  # noqa: E402

G = Path("reports/_01_E2_초기위치/_작업자료")
DEP = Path("/data3/kci_datasets/depth_cache_sign_eval")


def depth_dist(dep_mm):
    dep = dep_mm.astype(float) / 1000.0; dep[dep == 0] = np.nan; Hh, Wd = dep.shape; mid = dep[Hh // 3:2 * Hh // 3]
    return (np.nanmedian(mid[:, 2 * Wd // 5:3 * Wd // 5]), np.nanmedian(mid[:, :Wd // 4]), np.nanmedian(mid[:, 3 * Wd // 4:]))


def classify(fwd, L, Rr):
    wide = max(L, Rr)
    if fwd < 4.5:
        return "dead-end"
    if (fwd > 12 and wide < 7.5) or wide < 4.5:
        return "corridor"
    return "junction"


def depth_scene(F, t):
    d = np.load(DEP / f"{F}_full.npz"); keys = np.array(sorted(float(k) for k in d.files))
    use = keys[(keys >= t - 1.5) & (keys <= t)]
    if len(use) == 0:
        use = keys[[int(np.argmin(abs(keys - t)))]]
    s = [depth_dist(d[f"{k:.1f}"]) for k in use]
    f, l, r = (float(np.nanmedian([x[i] for x in s])) for i in range(3))
    return classify(f, l, r), f, l, r, ",".join(f"{k:g}" for k in use)


def lidar_scene(t):
    import floor_id as FI
    if ED._ST.get("building") != t["building"]:
        cfg = FI.load_cfg(); b = FI.setup_env(cfg, t["building"]); ED._ST.update(building=t["building"], b=b, cache={})
    import rt_localizer as RT, run_floor_id as Rr
    F = t["floor"]
    if F not in ED._ST["cache"]:
        ED._ST["cache"] = {F: RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=1, ppm=Rr.MID,
                                             data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)}
    return ED._ST["cache"][F].L.depth_scene(float(t["start_sec"]))


def main():
    rows = []
    for t in csv.DictReader(open(G / "tasks/E2_tasks.csv")):
        lab, f, l, r, used = depth_scene(t["floor"], float(t["start_sec"]))
        rows.append(dict(task_id=t["task_id"], floor=t["floor"], start_type=t["start_type"], start_sec=t["start_sec"], lidar_scene=lidar_scene(t),
                         depth_scene=lab, depth_fwd_m=round(f, 2), depth_left_m=round(l, 2), depth_right_m=round(r, 2), depth_keys=used))
        print(rows[-1], flush=True)
    with open(G / "tasks/E2_scene_labels.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    same = sum(r["lidar_scene"] == r["depth_scene"] for r in rows)
    print(f"라이다·깊이 라벨 같음 {same}/{len(rows)}")
    for k in ("corridor", "junction", "dead-end"):
        print(k, "라이다", sum(r["lidar_scene"] == k for r in rows), "깊이", sum(r["depth_scene"] == k for r in rows))


if __name__ == "__main__":
    main()
