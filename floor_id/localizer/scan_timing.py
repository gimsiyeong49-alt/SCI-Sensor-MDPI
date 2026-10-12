"""Scan-time association, shared by replay and live accumulation (no ROS dependency)."""
from bisect import bisect_left
import numpy as np


def window_indices(times, end, span, step):
    """Nearest samples to a time grid, restricted to the closed past window.

    Return (requested time, index). Never repeat a scan or borrow a future scan
    at the end / a pre-window scan at the beginning. Do not bridge missing data.
    """
    times = np.asarray(times, float)
    if step <= 0 or span < 0:
        raise ValueError('step must be positive and span nonnegative')
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('Scan timestamps must be strictly increasing')
    lo = int(np.searchsorted(times, end-span, side='left'))
    hi = int(np.searchsorted(times, end, side='right'))
    out = []
    for requested in end - np.arange(0, span+1e-6, step)[::-1]:
        if requested < times[0] or lo == hi:
            continue
        j = int(np.searchsorted(times, requested))
        candidates = [i for i in (j-1, j) if lo <= i < hi]
        if not candidates:
            continue
        i = min(candidates, key=lambda k: (abs(times[k]-requested), k))
        if abs(times[i]-requested) > step or (out and i == out[-1][1]):
            continue
        out.append((float(requested), i))
    return out


class ScanOdomSync:
    """Buffer scans until odometry brackets their acquisition timestamp.

    Yaw follows the shortest arc across +/-pi. Never extrapolate stale odometry.
    Output is time ordered; delayed scans older than an emitted scan are dropped.
    Both callbacks must use the caller's common lock.
    """
    def __init__(self, history_s=65., max_gap_s=.5, max_wait_s=1.):
        self.history_s = float(history_s)
        self.max_gap_s = float(max_gap_s)
        self.max_wait_s = float(max_wait_s)
        self.odom = []
        self.pending = []
        self.last_emitted = -np.inf
        self.dropped = 0

    def add_odom(self, t, xy, yaw):
        t = float(t); xy = np.asarray(xy, float).copy(); yaw = float(yaw)
        if not np.isfinite(t) or not np.isfinite(xy).all() or not np.isfinite(yaw):
            return []
        k = bisect_left([p[0] for p in self.odom], t)
        item = (t, xy, yaw)
        if k < len(self.odom) and self.odom[k][0] == t:
            self.odom[k] = item
        else:
            self.odom.insert(k, item)
        cutoff = self.odom[-1][0] - self.history_s
        # Keep one pose preceding the history boundary for interpolation.
        while len(self.odom) > 2 and self.odom[1][0] < cutoff:
            self.odom.pop(0)
        return self._drain()

    def add_scan(self, t, pts):
        t = float(t)
        if not np.isfinite(t) or t <= self.last_emitted:
            self.dropped += 1
            return []
        k = bisect_left([p[0] for p in self.pending], t)
        if k < len(self.pending) and self.pending[k][0] == t:
            return []
        self.pending.insert(k, (t, np.asarray(pts, float).copy()))
        cutoff = self.pending[-1][0] - self.history_s
        while self.pending and self.pending[0][0] < cutoff:
            self.pending.pop(0); self.dropped += 1
        return self._drain()

    def _drain(self):
        out = []
        while self.pending and self.odom:
            t, pts = self.pending[0]
            ts = [p[0] for p in self.odom]
            k = bisect_left(ts, t)
            if k == len(ts):
                break  # Await the odometry sample after this scan.
            if ts[k] == t:
                _, xy, yaw = self.odom[k]
            elif k == 0:
                # An earlier odometry callback may still arrive; only expire
                # after the retained history has advanced past this scan.
                if max(ts[-1], self.pending[-1][0])-t <= self.max_wait_s:
                    break
                self.pending.pop(0); self.dropped += 1; continue
            else:
                a, b = self.odom[k-1], self.odom[k]
                if b[0]-a[0] > self.max_gap_s:
                    # Allow a delayed intermediate pose to fill this gap.
                    if max(ts[-1], self.pending[-1][0])-t <= self.max_wait_s:
                        break
                    self.pending.pop(0); self.dropped += 1; continue
                w = (t-a[0])/(b[0]-a[0])
                xy = a[1] + w*(b[1]-a[1])
                dyaw = np.arctan2(np.sin(b[2]-a[2]), np.cos(b[2]-a[2]))
                yaw = a[2] + w*dyaw
            self.pending.pop(0)
            if t <= self.last_emitted:
                self.dropped += 1; continue
            self.last_emitted = t
            out.append((t, pts, xy.copy(), float(yaw)))
        return out
