"""metric_fit_negev.py — metric_fit.fit_window 사본 + 빈 공간 감점으로 후보 재선택(옵션). metric_fit.py 는 그대로 둔다.

현재 정합(fit_window)은 벽까지 거리의 가까운 70% 평균만 보므로, 점이 벽 가까이만 가면 빈 공간·안 보인 벽을 따지지 않는다
(2026-09-14 진단 out_LiDAR/update0914/diag/README.md). 여기서는 후보 생성·국소 정합·merit 계산을 한 줄도 바꾸지 않고,
기존 merit 상위의 서로 다른 자세 RERANK 개에 두 감점을 더해 다시 고른다.

  T2 먼 점        : 벽에서 FAR_M 넘게 떨어진 점의 비율 (fit 의 outlier_60cm 과 같은 정의)
  T4 안 보인 벽    : 궤적 프레임마다 360° 광선의 첫 지도 벽 중 TOL_M 안에 점이 없는 비율(사거리 twoway.RMAX).
                   광선이 첫 벽에 닿기 전에 지도 문 마스크(문 틈을 메운 막대)를 지나면 문 너머 벽이라 뺀다.
  merit_neg = merit + FAR_M·(T2 + T4)   — 비율에 이미 쓰는 0.6m 를 곱해 거리 단위로. 이 데이터로 조정한 값 없음.

base(= 기존 fit_window 결과와 같아야 함)와 neg(재선택 결과)를 함께 돌려준다. GT 미사용.
"""
import os
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, binary_dilation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from node_local import GRID_STEP
from metric_fit import set_scale
import twoway as TW

RERANK = 8                    # 기존 merit 상위의 서로 다른 자세 수
FAR_M, EDGE_M, TOL_M = 0.6, 0.3, 0.3
N_DIR = 180                   # 광선 2°
DOOR_GROW_PX = 3              # 문 막대 래스터 반올림 여유(지도 px)
DUP_M, DUP_DEG, DUP_SCALE = 1.0, 10.0, 1.1   # 이보다 가까우면 같은 자세로 본다(이웃 축척·격자에서 같은 해로 수렴)

_door = {}


def door_mask(floor, shape):
    if floor not in _door:
        root = os.environ.get('SEMLOC_MAPS_ROOT', '/home/user/kci_maps_perfloor')
        m = np.array(Image.open(f'{root}/{floor}F/maps/door_mask.png').convert('L')) > 127
        _door[floor] = binary_dilation(m, iterations=DOOR_GROW_PX)
    assert _door[floor].shape == shape, (floor, _door[floor].shape, shape)
    return _door[floor]


def unexplained_walls(wall, door, cloud, traj, ppm, rmax_m=TW.RMAX):
    """(T4 문 제외, T4 문 포함, 문 제외 전 첫 벽 수, 문 너머로 뺀 수, 안 보인 벽 좌표)."""
    H, Wd = wall.shape
    ang = (np.arange(N_DIR) + 0.5) / N_DIR * 2 * np.pi
    rr = np.arange(EDGE_M * ppm, rmax_m * ppm, 1.0)
    cx, sy = np.outer(np.cos(ang), rr), np.outer(np.sin(ang), rr)
    idx = np.arange(len(rr))[None, :]
    hits, thru = [], []
    for S in traj:
        xs, ys = S[0] + cx, S[1] - sy
        inb = (xs >= 0) & (xs < Wd) & (ys >= 0) & (ys < H)
        xi, yi = np.clip(np.rint(xs).astype(int), 0, Wd - 1), np.clip(np.rint(ys).astype(int), 0, H - 1)
        w = wall[yi, xi] & inb
        has = w.any(1); j = np.argmax(w, 1); a = np.where(has)[0]
        d = (door[yi, xi] & inb & (idx <= j[:, None])).any(1)
        hits.append(np.c_[xs[a, j[a]], ys[a, j[a]]]); thru.append(d[a])
    hits, thru = np.vstack(hits), np.concatenate(thru)
    if len(hits) == 0:
        return 1.0, 1.0, 0, 0, hits
    miss = cKDTree(cloud).query(hits, k=1)[0] > TOL_M * ppm
    keep = ~thru
    t4 = float(miss[keep].mean()) if keep.any() else 1.0
    return t4, float(miss.mean()), int(len(hits)), int(thru.sum()), hits[keep & miss]


def fit_window_negev(localizer, t_now, scales, floor, W=60., dt=3., topk=3, rerank=RERANK):
    """fit_window 과 같은 후보·같은 국소 정합. 반환 dict(base=기존 최선, neg=감점 재선택, cands=재채점 후보 요약)."""
    from rt_localizer import _A, _R
    Z, L = localizer, localizer.L
    wall = getattr(L, 'wall_real', L.wall)
    door = door_mask(floor, wall.shape)
    saved = (L.PPM, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis)
    try:
        start = max(L.T[0], t_now-W)
        Z.fit_beta(start, t_now)
        modes, _ = Z.bootstrap(start, boot_s=t_now-start)
        buf = Z.window_scans(t_now, W, dt)
        frames = np.array([fr['t'] for fr in buf])
        sampling_times = np.array([fr['sample_seed_t'] for fr in buf])
        anchor = np.asarray(buf[-1]['xy'])
        cloud, trajectory, fidx = [], [], []
        for n, fr in enumerate(buf):
            pts, yaw = fr['pts'], fr['yaw']
            if len(pts) > 400:
                pts = pts[np.random.RandomState(int(fr['sample_seed_t'])%97).choice(len(pts), 400, replace=False)]
            delta = np.asarray(fr['xy']) - anchor
            cloud.append(delta + (pts * [1, Z.MIRROR]) @ _R(yaw+Z.BETA).T)
            trajectory.append(delta); fidx.append(np.full(len(pts), n, np.int16))
        cloud = np.vstack(cloud); fidx = np.concatenate(fidx)
        if len(cloud) < 50:
            raise ValueError('Too few scan points for metric fitting')
        sample = cloud[np.linspace(0, len(cloud)-1, min(1000, len(cloud))).astype(int)]
        trajectory = np.asarray(trajectory)

        def project(x, pts):
            return pts @ _A(x[2]).T * np.exp(x[3]) + x[:2]

        def distances(x, pts):
            px = project(x, pts)
            return map_coordinates(L.DFpx, [px[:, 1], px[:, 0]], order=1,
                                   mode='constant', cval=max(L.H, L.W)) / np.exp(x[3])

        def residual(x):
            return np.minimum(distances(x, sample), 5.)

        cands = []
        for ppm in scales:
            set_scale(Z, ppm)
            result = Z.localize(t_now, W=W, dt=dt, phis=modes)
            for phi in modes:
                values = np.where(Z.allow.reshape(L.shp), result['P'][phi], -np.inf).ravel()
                indices = np.argsort(-values)[:topk]
                for k in indices:
                    if not np.isfinite(values[k]): continue
                    x0 = np.r_[L.Pf[k], np.deg2rad(phi), np.log(ppm)]
                    shift = 1.5 * GRID_STEP
                    low = x0 + [-shift, -shift, -np.deg2rad(8), np.log(.8)]
                    high = x0 + [shift, shift, np.deg2rad(8), np.log(1.25)]
                    opt = least_squares(residual, x0, bounds=(low, high),
                                        loss='soft_l1', f_scale=.2,
                                        diff_step=[1e-3, 1e-3, 1e-4, 1e-4], max_nfev=60)
                    x = opt.x
                    px, py = np.rint(x[:2]).astype(int)
                    if not (0 <= px < L.W and 0 <= py < L.H and Z.ok_px[py, px]):
                        continue
                    error = distances(x, cloud)
                    cost = float(np.mean(np.sort(error)[:int(.7*len(error))]))
                    track = project(x, trajectory)
                    ix, iy = np.rint(track).astype(int).T
                    inside = (ix >= 0) & (ix < L.W) & (iy >= 0) & (iy < L.H)
                    traversable = np.zeros(len(ix), bool)
                    traversable[inside] = Z.ok_px[iy[inside], ix[inside]]
                    blocked = float(1-traversable.mean())
                    nogo = float(Z.pen_px[py, px])
                    merit = cost + .5 * blocked + .1 * Z.w_nogo * nogo
                    cands.append(dict(x=x.copy(), merit=merit, cost=cost, blocked=blocked, nogo=nogo,
                                      error=error, scene=result['scene'], seed_ppm=float(ppm), seed_phi=float(phi)))
        if not cands:
            raise ValueError('No traversable metric fit candidates')
        order = sorted(range(len(cands)), key=lambda i: cands[i]['merit'])   # 안정 정렬 = 원본의 '처음 나온 최소' 규칙

        # 서로 다른 자세만 상위 rerank 개
        picked = []
        for i in order:
            x = cands[i]['x']
            dup = False
            for j in picked:
                y = cands[j]['x']
                dpos = np.hypot(*(x[:2]-y[:2])) / np.exp(0.5*(x[3]+y[3]))
                dphi = abs((np.degrees(x[2]-y[2]) + 180) % 360 - 180)
                if dpos < DUP_M and dphi < DUP_DEG and abs(x[3]-y[3]) < np.log(DUP_SCALE):
                    dup = True; break
            if not dup:
                picked.append(i)
            if len(picked) == rerank: break

        def pack(i, rank):
            c = cands[i]; x = c['x']
            cpx, tpx = project(x, cloud), project(x, trajectory)
            t4, t4_raw, n_hits, n_door, miss_xy = unexplained_walls(wall, door, cpx, tpx, float(np.exp(x[3])))
            t2 = float(np.mean(c['error'] > FAR_M))
            e = c['error']
            return dict(pos=x[:2].copy(), phi=float(np.degrees(x[2]) % 360), ppm=float(np.exp(x[3])),
                        beta=float(Z.BETA), mirror=int(Z.MIRROR),
                        score=-c['merit'], cost_m=c['merit'], wall_cost_m=c['cost'], blocked_fraction=c['blocked'], nogo=c['nogo'],
                        T2=t2, T4=t4, T4_with_doors=t4_raw, n_wall_hits=n_hits, n_door_rays=n_door,
                        merit_neg=float(c['merit'] + FAR_M*(t2 + t4)), base_rank=rank, n_candidates=len(cands),
                        anchor=anchor, frames=frames, frame_times=frames, sampling_times=sampling_times, t=float(frames[-1]),
                        requested_t=float(t_now), age_s=float(t_now-frames[-1]),
                        cloud_px=cpx, trajectory_px=tpx, fidx=fidx, distances_m=e, miss_wall_px=miss_xy,
                        median_m=float(np.median(e)), inlier_10cm=float(np.mean(e < .1)), outlier_60cm=t2,
                        beta_quality=dict(Z.beta_quality), scene=c['scene'])

        packed = [pack(i, r) for r, i in enumerate(picked)]
        best = min(range(len(packed)), key=lambda r: packed[r]['merit_neg'])   # 동점은 기존 순위가 앞선 쪽
        summary = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()
                    if k in ('pos', 'phi', 'ppm', 'cost_m', 'wall_cost_m', 'blocked_fraction', 'nogo',
                             'T2', 'T4', 'T4_with_doors', 'n_wall_hits', 'n_door_rays', 'merit_neg', 'base_rank')}
                   for p in packed]
        return dict(base=packed[0], neg=packed[best], cands=summary)
    finally:
        ppm, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis = saved
        set_scale(Z, ppm)
