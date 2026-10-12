"""Joint position/orientation/scale fitting with distances measured in metres.

The RT grid supplies hypotheses. Its pixel-clipped likelihood is appropriate
for a fixed scale, but is not comparable across scales: shrinking a cloud onto
one thick wall can win. Refine each scale's hypotheses before comparing their
unclipped, robust metric residuals.
"""
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from node_local import GRID_STEP


def set_scale(localizer, ppm):
    L = localizer.L
    L.PPM = float(ppm)
    L.freeD_grid = L.freeD[np.clip(L.GYY.astype(int), 0, L.H-1),
                         np.clip(L.GXX.astype(int), 0, L.W-1)] / L.PPM
    localizer.sm.PPM = L.PPM


def fit_window(localizer, t_now, scales, W=60., dt=3., topk=3):
    """Use only [t_now-W, t_now]; restore the caller's tracking calibration.

    Returned pose includes its own beta and ppm. Use returned cloud_px or pass
    both values to scan_to_map when displaying this result. No GT is accessed.
    """
    from rt_localizer import _A, _R
    Z, L = localizer, localizer.L
    saved = (L.PPM, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis)
    try:
        start = max(L.T[0], t_now-W)
        Z.fit_beta(start, t_now)
        modes, _ = Z.bootstrap(start, boot_s=t_now-start)
        buf = Z.window_scans(t_now, W, dt)
        frames = np.array([fr['t'] for fr in buf])
        sampling_times = np.array([fr['sample_seed_t'] for fr in buf])
        anchor = np.asarray(buf[-1]['xy'])
        cloud, trajectory = [], []
        for fr in buf:
            pts, yaw = fr['pts'], fr['yaw']
            if len(pts) > 400:
                pts = pts[np.random.RandomState(int(fr['sample_seed_t'])%97).choice(len(pts), 400, replace=False)]
            delta = np.asarray(fr['xy']) - anchor
            cloud.append(delta + (pts * [1, Z.MIRROR]) @ _R(yaw+Z.BETA).T)
            trajectory.append(delta)
        cloud = np.vstack(cloud)
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

        best = None
        for ppm in scales:
            set_scale(Z, ppm)
            result = Z.localize(t_now, W=W, dt=dt, phis=modes)
            for phi in modes:
                values = np.where(Z.allow.reshape(L.shp), result['P'][phi], -np.inf).ravel()
                indices = np.argsort(-values)[:topk]
                for k in indices:
                    if not np.isfinite(values[k]): continue
                    x0 = np.r_[L.Pf[k], np.deg2rad(phi), np.log(ppm)]
                    # A grid cell and neighbouring yaw/scale hypotheses overlap.
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
                    # Keep the existing traversal constraint after sub-grid refinement.
                    blocked = float(1-traversable.mean())
                    # Preserve the original no-go/restroom penalty: geometric
                    # refinement must not undo candidate constraints.
                    nogo = float(Z.pen_px[py, px])
                    merit = cost + .5 * blocked + .1 * Z.w_nogo * nogo
                    if best is None or merit < best['cost_m']:
                        best = dict(pos=x[:2].copy(), phi=float(np.degrees(x[2])%360),
                                    ppm=float(np.exp(x[3])), beta=float(Z.BETA), mirror=int(Z.MIRROR),
                                    score=-merit, cost_m=merit, wall_cost_m=cost,
                                    blocked_fraction=blocked,
                                    nogo=nogo,
                                    anchor=anchor, frames=frames, frame_times=frames,
                                    sampling_times=sampling_times, t=float(frames[-1]),
                                    requested_t=float(t_now), age_s=float(t_now-frames[-1]),
                                    cloud_px=project(x, cloud),
                                    trajectory_px=track, distances_m=error,
                                    median_m=float(np.median(error)),
                                    inlier_10cm=float(np.mean(error<.1)),
                                    outlier_60cm=float(np.mean(error>.6)),
                                    beta_quality=dict(Z.beta_quality), scene=result['scene'])
        if best is None:
            raise ValueError('No traversable metric fit candidates')
        return best
    finally:
        ppm, Z.BETA, Z.MIRROR, Z.beta_quality, Z.phis = saved
        set_scale(Z, ppm)
