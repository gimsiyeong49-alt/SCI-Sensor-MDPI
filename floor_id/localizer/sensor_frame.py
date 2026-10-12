"""Estimate the planar scan-to-odometry rotation from overlapping observations.

No map, floor identifier, trajectory PCA, or ground truth is used. All frames
are supplied by the caller, which controls the causal time boundary.
"""
import numpy as np
from scipy.spatial import cKDTree


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def estimate_beta(frames, pair_s=3.0, step_deg=2.0, min_disp=.3,
                  max_pair_error=.2, cap=500):
    """Robust nearest-neighbour registration, including odometry translation.

    A pair whose best possible alignment still exceeds max_pair_error metres
    cannot calibrate the frame transform. This rejects distorted startup scans
    rather than interpreting their unrelated points as a sensor rotation.
    """
    times = np.array([f['t'] for f in frames])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Calibration needs at least two time-ordered scans')
    angles = np.deg2rad(np.arange(0., 360., step_deg))
    curves = {1: [], -1: []}
    intervals, turns = [], []
    for i, first in enumerate(frames):
        j = int(np.argmin(abs(times - (times[i] + pair_s))))
        if j <= i or abs(times[j] - times[i] - pair_s) > pair_s / 4:
            continue
        second = frames[j]
        delta = np.asarray(second['xy']) - first['xy']
        if np.linalg.norm(delta) < min_disp:
            continue
        p = np.asarray(first.get('near_pts', first['pts']), float)
        q = np.asarray(second.get('near_pts', second['pts']), float)
        p, q = p[np.isfinite(p).all(1)], q[np.isfinite(q).all(1)]
        if min(len(p), len(q)) < 50:
            continue
        if len(p) > cap: p = p[np.linspace(0, len(p)-1, cap).astype(int)]
        if len(q) > cap: q = q[np.linspace(0, len(q)-1, cap).astype(int)]
        tree = cKDTree(p)
        dyaw = second['yaw'] - first['yaw']
        for mirror in (1, -1):
            # In the original scan frame: M R(dyaw) M = R(mirror*dyaw).
            rotated = q @ rotation(mirror * dyaw).T
            costs = []
            for beta in angles:
                translation = (delta @ rotation(-first['yaw'] - beta).T) * [1, mirror]
                costs.append(np.median(tree.query(rotated + translation)[0]))
            curves[mirror].append(costs)
        intervals.append((float(times[i]), float(times[j])))
        turns.append(abs(np.arctan2(np.sin(dyaw), np.cos(dyaw))) >= np.deg2rad(10))
    if not intervals:
        raise ValueError('Not enough overlapping scans with translation to calibrate')
    hypotheses = []
    for mirror, rows in curves.items():
        costs = np.asarray(rows)
        good = costs.min(axis=1) <= max_pair_error
        if not good.any(): continue
        objective = np.median(costs[good], axis=0)
        best = int(np.argmin(objective))
        # Straight travel cannot identify handedness. Evaluate turn pairs even
        # when a wrong handedness would reject them as bad observations.
        selection = np.asarray(turns) if sum(turns) >= 2 else np.ones(len(rows), bool)
        support = float(np.median(costs[selection, best]))
        hypotheses.append((support, mirror, best, objective, good))
    if not hypotheses:
        raise ValueError('No geometrically consistent scan pairs; collect more observations')
    support, mirror, best, objective, good = min(hypotheses, key=lambda h: h[0])
    return float(angles[best]), dict(
        mirror=mirror, handedness_error_m=support, turn_pairs=int(sum(turns)),
        error_m=float(objective[best]), pairs=int(good.sum()),
        rejected_pairs=int((~good).sum()),
        interval=[float(times[0]), float(times[-1])],
        pair_intervals=[intervals[i] for i in np.flatnonzero(good)],
        step_deg=float(step_deg))
