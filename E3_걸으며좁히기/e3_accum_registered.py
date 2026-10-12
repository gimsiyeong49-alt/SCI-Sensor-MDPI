#!/usr/bin/env python3
"""E3 쌓인 라이다를 '그 시점 후보 자세'로 지도에 얹기(2026-10-11, 협업자 위치추정 그림의 스캔 정합과 같은 식).
위치추정 코드(rt_localizer.localize_buf)와 같은 변환: 프레임 i 의 지도 위치 = 후보 위치(지금) + ppm·_A(φ)·(오돔_i − 오돔_지금),
점 방향 = 후보 방향(φ + 장착각) + 그 프레임 오돔 yaw, 점은 400개 표본(같은 씨앗)·거울 적용. 관측 고르기는 window_scans(3초 간격) 그대로.
    registered(task, T, m, px, py, hd_deg, ppm) -> (지도 점 Nx2, 프레임 위치 Fx2, 관측 장수, 점 수)
    python3 scripts/e3_accum_registered.py --check 4F_T1    # 정답 자세로 만든 경로가 채점용 정답 경로와 겹치는지
"""
import argparse, csv, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402


def _loc(task):
    import e2_dense as ED
    t = EV.TASK[task]; EV.obs(task); return ED._ST["cache"][(t["floor"], 1)]


def registered(task, T, px, py, hd_deg, ppm, beta_deg=228.0):
    from rt_localizer import _A
    Z = _loc(task); L = Z.L; t = EV.TASK[task]; ts = float(t["start_sec"])
    buf = Z.window_scans(L.T[0] + ts + T, T, 3.0) if T > 0 else Z.window_scans(L.T[0] + ts + 3.0, 3.0, 3.0)[:1]   # 0초 = 출발 관측 한 장(E2 와 같은 프레임)
    beta, mirror = beta_deg, 1                      # 기본 228°(E2·E3), 노드 후보는 그 창에서 추정한 값
    phi = math.radians(hd_deg - beta); anchor = np.asarray(buf[-1]["xy"], float)
    P, pos, n = [], [], 0
    for fr in buf:
        pts = np.asarray(fr["pts"], float); n += len(pts)
        if len(pts) > 400:
            rs = np.random.RandomState(int(fr.get("sample_seed_t", fr.get("t", 0))) % 97); pts = pts[rs.choice(len(pts), 400, replace=False)]
        pts = pts * [1, mirror]
        off = ppm * (_A(phi) @ (np.asarray(fr["xy"], float) - anchor)); p = np.array([px, py]) + off
        P.append(EV.project(pts, p, math.radians(hd_deg) + float(fr["yaw"]), ppm)); pos.append(p)
    return np.vstack(P), np.array(pos), len(buf), n, [fr["t"] - L.T[0] for fr in buf]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", default="4F_T1"); a = ap.parse_args()
    G = EV.G; EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    for k in a.check.split(","):
        ts = float(EV.TASK[k]["start_sec"]); _, ppm, hd, gt, gx = EV.gt_of(k)
        for T in (15, 30, 60):
            pT = gx[int(np.argmin(abs(gt - (ts + T))))]
            P, pos, nf, npt, tt = registered(k, T, pT[0], pT[1], hd, ppm)
            g = np.array([gx[int(np.argmin(abs(gt - x)))] for x in tt])
            e = np.hypot(*(pos - g).T) / ppm
            print(f"{k} {T}초: 관측 {nf}장 · 프레임 위치 대 정답 경로 오차 중앙 {np.median(e):.2f} m · 최대 {e.max():.2f} m")


if __name__ == "__main__":
    main()
