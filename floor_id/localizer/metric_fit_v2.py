"""metric_fit_v2.py — 감점 강화판 정합 (2026-09-18 사용자 요청). metric_fit.py·metric_fit_negev.py 는 그대로 둔다.

바뀐 점 (후보 생성·축척 스윕·국소 정합의 경계는 fit_window 와 같다)
  1. 벽 속 점: 기존 거리장은 벽 픽셀 안에서 0 이라 두꺼운 벽이 점을 빨아들였다.
     그 지도의 보통 벽 반두께 h(벽 픽셀 안쪽 깊이의 90번째 백분위)보다 깊이 들어간 점은 (깊이 − h) 만큼 떨어진 것으로 본다.
     → 국소 정합의 잔차와 정합 비용 모두 이 거리장으로 계산.
  2. 빈 공간 점 T2: 벽에서 0.6m 넘게 떨어진 점의 비율, 가중 1.5 (이전 0.6).
  3. 건물 밖 점 T3: building_footprint(안내도 의미 분석의 건물 내부) 밖으로 0.5m 넘게 나간 점의 비율, 가중 3.0.
  4. 벽 관통 경로 X: 창 안의 주행 거리계 경로를 0.2초 간격으로 지도에 올려 벽(문 자리 제외)을 지난 길이(m).
     0.3m 넘으면 후보에서 제외(모두 넘으면 가장 적게 뚫은 쪽을 남김), 그 아래는 1.0/m 로 감점.
  5. 안 보인 벽 T4 (문 틈 너머 제외): 가중 0.6 그대로.
  merit_v2 = 비용 + 0.5·궤적 통행불가 + 0.3·no-go + 1.5·T2 + 3.0·T3 + 1.0·X + 0.6·T4
  후보: 국소 정합한 전 후보(서로 다른 자세)를 다시 채점 — T4 는 계산량 때문에 T4 제외 점수 상위 12개에만.
가중치는 결과를 보고 고른 값이 아니라 요청 취지(빈 공간 강하게, 건물 밖 더 강하게, 벽 관통은 불가)에 맞춰 정한 값이다. GT 미사용.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, distance_transform_edt
from scipy.optimize import least_squares
from node_local import GRID_STEP
from metric_fit import set_scale
from metric_fit_negev import door_mask, unexplained_walls

W_FREE, W_OUT, W_MISS, W_CROSS = 1.5, 3.0, 0.6, 1.0
FAR_M, OUT_M, CROSS_MAX_M, TRAJ_DT = 0.6, 0.5, 0.3, 0.2
N_T4 = 12
DUP_M, DUP_DEG, DUP_SCALE = 1.0, 10.0, 1.1


def _maps(Z, floor):
    """층 지도마다 한 번: 벽 속 초과 깊이를 더한 거리장, 건물 밖 거리, 경로용 벽."""
    if getattr(Z, '_v2', None) is None:
        L = Z.L
        wall = getattr(L, 'wall_real', L.wall)
        depth = distance_transform_edt(wall)
        h = float(np.percentile(depth[wall], 90)) if wall.any() else 1.0
        F = L.DFpx + np.maximum(depth - h, 0.0)
        fp = np.asarray(Z.sm.footprint, bool)
        out_d = distance_transform_edt(~fp)
        wall_tr = wall & ~door_mask(floor, wall.shape)
        Z._v2 = dict(F=F, half_px=h, out_d=out_d, wall_tr=wall_tr, door=door_mask(floor, wall.shape), wall=wall)
    return Z._v2


def _sample(field, px, cval):
    return map_coordinates(field, [px[:, 1], px[:, 0]], order=1, mode='constant', cval=cval)


def _crossing_m(wall_tr, track_px, ppm):
    """경로를 1px 간격으로 조밀화해 벽 픽셀에 든 길이(m). 지도 밖도 관통으로 센다."""
    H, W = wall_tr.shape
    seg = np.diff(track_px, axis=0)
    n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1])).astype(int), 1)
    pts = np.vstack([track_px[i] + seg[i] * (np.arange(n[i])[:, None] / n[i]) for i in range(len(seg))] + [track_px[-1:]])
    x, y = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
    inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    bad = ~inb
    bad[inb] = wall_tr[y[inb], x[inb]]
    return float(bad.sum() / ppm)


def fit_window_v2(localizer, t_now, scales, floor, W=60., dt=3., topk=3):
    from rt_localizer import _A, _R
    Z, L = localizer, localizer.L
    M = _maps(Z, floor)
    saved = (L.PPM, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis)
    try:
        start = max(L.T[0], t_now - W)
        Z.fit_beta(start, t_now)
        modes, _ = Z.bootstrap(start, boot_s=t_now - start)
        buf = Z.window_scans(t_now, W, dt)
        frames = np.array([fr['t'] for fr in buf])
        anchor = np.asarray(buf[-1]['xy'])
        cloud, trajectory, fidx = [], [], []
        for n, fr in enumerate(buf):
            pts, yaw = fr['pts'], fr['yaw']
            if len(pts) > 400:
                pts = pts[np.random.RandomState(int(fr['sample_seed_t']) % 97).choice(len(pts), 400, replace=False)]
            delta = np.asarray(fr['xy']) - anchor
            cloud.append(delta + (pts * [1, Z.MIRROR]) @ _R(yaw + Z.BETA).T)
            trajectory.append(delta); fidx.append(np.full(len(pts), n, np.int16))
        cloud = np.vstack(cloud); fidx = np.concatenate(fidx); trajectory = np.asarray(trajectory)
        if len(cloud) < 50:
            raise ValueError('Too few scan points for metric fitting')
        # 조밀 경로: 창 안 주행 거리계 전 표본을 0.2초 간격으로 (마지막 스캔 시각까지)
        sel = np.where((L.T >= start) & (L.T <= frames[-1]))[0]
        keep, last = [], -1e9
        for i in sel:
            if L.T[i] - last >= TRAJ_DT:
                keep.append(i); last = L.T[i]
        dense = np.c_[np.asarray(L.X)[keep], np.asarray(L.Y)[keep]] - anchor
        if len(dense) < 2:
            dense = trajectory
        sample = cloud[np.linspace(0, len(cloud) - 1, min(1000, len(cloud))).astype(int)]
        CV = float(max(L.H, L.W))

        def project(x, pts):
            return pts @ _A(x[2]).T * np.exp(x[3]) + x[:2]

        def dist(x, pts):
            return _sample(M['F'], project(x, pts), CV) / np.exp(x[3])

        def residual(x):
            return np.minimum(dist(x, sample), 5.)

        cands = []
        for ppm in scales:
            set_scale(Z, ppm)
            result = Z.localize(t_now, W=W, dt=dt, phis=modes)
            for phi in modes:
                values = np.where(Z.allow.reshape(L.shp), result['P'][phi], -np.inf).ravel()
                for k in np.argsort(-values)[:topk]:
                    if not np.isfinite(values[k]): continue
                    x0 = np.r_[L.Pf[k], np.deg2rad(phi), np.log(ppm)]
                    shift = 1.5 * GRID_STEP
                    low = x0 + [-shift, -shift, -np.deg2rad(8), np.log(.8)]
                    high = x0 + [shift, shift, np.deg2rad(8), np.log(1.25)]
                    x = least_squares(residual, x0, bounds=(low, high), loss='soft_l1', f_scale=.2,
                                      diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60).x
                    px, py = np.rint(x[:2]).astype(int)
                    if not (0 <= px < L.W and 0 <= py < L.H and Z.ok_px[py, px]):
                        continue
                    s = float(np.exp(x[3]))
                    err = dist(x, cloud)
                    cost = float(np.mean(np.sort(err)[:int(.7 * len(err))]))
                    track = project(x, trajectory)
                    ix, iy = np.rint(track).astype(int).T
                    inside = (ix >= 0) & (ix < L.W) & (iy >= 0) & (iy < L.H)
                    trav = np.zeros(len(ix), bool); trav[inside] = Z.ok_px[iy[inside], ix[inside]]
                    blocked = float(1 - trav.mean())
                    nogo = float(Z.pen_px[py, px])
                    cpx = project(x, cloud)
                    T2 = float(np.mean(err > FAR_M))
                    T3 = float(np.mean(_sample(M['out_d'], cpx, CV) / s > OUT_M))
                    dpx = project(x, dense)
                    X = _crossing_m(M['wall_tr'], dpx, s)
                    part = cost + .5 * blocked + .1 * Z.w_nogo * nogo + W_FREE * T2 + W_OUT * T3 + W_CROSS * min(X, CROSS_MAX_M)
                    cands.append(dict(x=x.copy(), err=err, cost=cost, blocked=blocked, nogo=nogo, T2=T2, T3=T3, X=X, part=part,
                                      scene=result['scene'], cpx=cpx, tpx=track, dpx=dpx))
        if not cands:
            raise ValueError('No traversable metric fit candidates')
        feas = [c for c in cands if c['X'] <= CROSS_MAX_M]
        n_feas = len(feas)
        if not feas:                                  # 전부 벽을 뚫으면 가장 적게 뚫은 쪽들만
            xmin = min(c['X'] for c in cands)
            feas = [c for c in cands if c['X'] <= xmin + 1e-9]
        feas.sort(key=lambda c: c['part'])
        picked = []
        for c in feas:
            x = c['x']
            if any(np.hypot(*(x[:2] - p['x'][:2])) / np.exp(0.5 * (x[3] + p['x'][3])) < DUP_M
                   and abs((np.degrees(x[2] - p['x'][2]) + 180) % 360 - 180) < DUP_DEG
                   and abs(x[3] - p['x'][3]) < np.log(DUP_SCALE) for p in picked):
                continue
            picked.append(c)
            if len(picked) == N_T4: break
        for c in picked:
            s = float(np.exp(c['x'][3]))
            t4, t4_raw, n_hits, n_door, miss = unexplained_walls(M['wall'], M['door'], c['cpx'], c['tpx'], s)
            c.update(T4=t4, T4_with_doors=t4_raw, n_wall_hits=n_hits, n_door_rays=n_door, miss_xy=miss,
                     merit=c['part'] + W_MISS * t4)
        bi = min(range(len(picked)), key=lambda i: picked[i]['merit'])
        best = picked[bi]
        x = best['x']; e = best['err']
        return dict(pos=x[:2].copy(), phi=float(np.degrees(x[2]) % 360), ppm=float(np.exp(x[3])),
                    beta=float(Z.BETA), mirror=int(Z.MIRROR), cost_m=best['merit'], wall_cost_m=best['cost'],
                    blocked_fraction=best['blocked'], nogo=best['nogo'], T2=best['T2'], T3=best['T3'], X_cross_m=best['X'],
                    T4=best['T4'], T4_with_doors=best['T4_with_doors'], n_wall_hits=best['n_wall_hits'],
                    n_door_rays=best['n_door_rays'], rank_in_part=bi, n_candidates=len(cands),
                    n_feasible=n_feas, half_wall_px=M['half_px'], anchor=anchor, frames=frames, t=float(frames[-1]),
                    cloud_px=best['cpx'], trajectory_px=best['dpx'], frame_px=best['tpx'], fidx=fidx, distances_m=e,
                    miss_wall_px=best['miss_xy'], median_m=float(np.median(e)), inlier_10cm=float(np.mean(e < .1)),
                    scene=best['scene'])
    finally:
        ppm, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis = saved
        set_scale(Z, ppm)
