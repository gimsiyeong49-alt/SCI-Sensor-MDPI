#!/usr/bin/env python3
"""오돔·센서 이상 판단(2026-10-08, 실험 지침서 14.1 ODOMETRY_OR_SENSOR_UNRELIABLE). make_gate_json 의 1초 속도 기준(2 m/s, 120°/s)이
1 m 이하 이동 튐과 10~45° 회전 튐을 못 잡아(E4 주입 시험 3/8) 아래로 바꾼다. 기준은 13개 층 정상 주행 실측에서 정했다(층·건물 맞춤 없음).
  A. 오돔 한 걸음(≈0.1 s)씩: 속도 > 3.0 m/s 또는 회전 > 150°/s  (13개 층 정상 최대 2.3 m/s, 104°/s)
  B. 자이로와 회전 맞춤(IMU 가 있을 때): 0.3 s 동안 오돔 회전 − 자이로 적분, 앞 30 s 중앙값(자이로 바이어스)을 뺀 차이 > 9°
     (13개 층 정상 최대 7.6°). 자이로 축·부호·시각 차이는 처음 60 s 로 정한다(인과). 축이 확실할 때만 켠다(gyro_ok).
  C. 오돔 끊김: 다음 자세까지 > 1.0 s
결과는 이상 시각 목록(초, 이유). 게이트는 그 초를 SENSOR 로 본다.
    python3 scripts/sensor_check.py --odom <odom_link.csv> [--bag <IMU 든 rosbag 폴더>] --out <sensor_events.json>
"""
import argparse, csv, json, sqlite3, struct
from pathlib import Path
import numpy as np

V_MAX, W_MAX, GYRO_TOL, GYRO_WIN, BIAS_WIN, GAP_MAX = 3.0, 150.0, 9.0, 0.3, 30.0, 1.0


def load_odom(p):
    r = list(csv.DictReader(open(p)))
    return (np.array([float(x["t_sec"]) for x in r]), np.array([float(x["fastlio_x"]) for x in r]), np.array([float(x["fastlio_y"]) for x in r]),
            np.unwrap(np.array([float(x["fastlio_yaw_rad"]) for x in r])))


def load_gyro(bag, topic="/utlidar/imu"):
    """sensor_msgs/Imu 를 rosbag2 sqlite 에서 직접 읽는다(ROS 없이). 헤더 시각이 수신 시각과 하루 넘게 다르면(Go2 L1 시계) 중앙 차이로 옮긴다."""
    T, H, G = [], [], []
    for f in sorted(Path(bag).glob("*.db3")):            # 여러 조각으로 나뉜 녹화본도 전부
        c = sqlite3.connect(f"file:{f}?mode=ro&immutable=1", uri=True)
        tid = [r[0] for r in c.execute("select id from topics where name=?", (topic,))]
        if not tid:
            continue
        for ts, d in c.execute(f"select timestamp,data from messages where topic_id={tid[0]}"):   # 정렬은 numpy 로(인덱스 없는 녹화본에서 sqlite 임시 정렬 과열 방지)
            d = bytes(d); sec, nsec = struct.unpack_from("<iI", d, 4); L = struct.unpack_from("<I", d, 12)[0]; o = 16 + L; o = (o - 4 + 7) // 8 * 8 + 4
            o += 8 * 4 + 8 * 9; G.append(struct.unpack_from("<3d", d, o)); T.append(ts * 1e-9); H.append(sec + nsec * 1e-9)
    if not T:
        return None
    o = np.argsort(T); T, H, G = list(np.array(T)[o]), list(np.array(H)[o]), list(np.array(G)[o])
    T, H, G = np.array(T), np.array(H), np.array(G)
    t = H + np.median(T - H) if np.median(np.abs(T - H)) > 1.0 else H
    return t, G


def fit_gyro(t, yaw, tg, G, span=60.0):
    """처음 span 초로 자이로 축·부호·시각 차이를 고른다(오돔 회전 속도와 상관이 가장 큰 것)."""
    m = t <= t[0] + span; tt = np.arange(t[0], t[m][-1], 0.05); wr = np.gradient(np.interp(tt, t, yaw), tt)
    best, per_ax = (0, 0, 1, 0.0), [0.0, 0.0, 0.0]
    for ax in range(3):
        for lag in np.arange(-1.0, 1.0001, 0.02):
            g = np.interp(tt + lag, tg, G[:, ax]); cc = np.corrcoef(wr, g)[0, 1]
            per_ax[ax] = max(per_ax[ax], abs(cc))
            if abs(cc) > abs(best[0]):
                best = (cc, ax, np.sign(cc), lag)
    fit_gyro.other = max(v for k, v in enumerate(per_ax) if k != best[1])   # 다른 축 최대 상관(축이 확실한지 볼 때)
    return best


def gyro_ok(cc, other):
    """자이로 축이 확실한가: 상관 0.5 이상이고 다른 축보다 2배 이상. 네 발 걸음 진동으로 회전 속도 상관은 0.6~0.8 에 머문다(2026-10-08 전층)."""
    return abs(cc) >= 0.5 and abs(cc) >= 2.0 * other


def check(t, x, y, yaw, gyro=None, calibrated=False):
    ev = []
    dt = np.diff(t); v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(dt, 1e-3); w = np.degrees(np.abs(np.diff(yaw))) / np.maximum(dt, 1e-3)
    for i in np.where(dt > GAP_MAX)[0]:
        ev.append((float(t[i + 1]), "오돔 끊김", float(dt[i])))
    ok = dt <= GAP_MAX
    for i in np.where(ok & (v > V_MAX))[0]:
        ev.append((float(t[i + 1]), "순간 이동", float(v[i])))
    for i in np.where(ok & (w > W_MAX))[0]:
        ev.append((float(t[i + 1]), "순간 회전", float(w[i])))
    info = None
    if gyro is not None:
        tg, G = gyro
        for span in (60.0, 120.0, 180.0):                  # 축이 불확실하면 더 기다려 다시 맞춘다(그 전까지는 자이로 검사 없음)
            cc, ax, sg, lag = fit_gyro(t, yaw, tg, G, span=span); oth = getattr(fit_gyro, "other", 0.0)
            if gyro_ok(cc, oth):
                break
        gyro_from = -np.inf if calibrated else t[0] + span   # 실시간판은 이미 맞춘 값으로 부르므로 처음부터
        if not gyro_ok(cc, oth) or tg[0] > t[0] + 60 or tg[-1] < t[-1] - 60:     # 축이 불확실하거나 구간을 못 덮으면 자이로 검사는 끈다
            return sorted(ev), dict(corr=round(float(cc), 3), other_axis=round(float(oth), 3), skipped="자이로 축·구간이 불확실")
        gz = sg * G[:, ax]; cg = np.concatenate([[0], np.cumsum((gz[1:] + gz[:-1]) / 2 * np.diff(tg))])
        a = t[:-1]; b = a + GYRO_WIN
        val = (b <= t[-1]) & (a + lag >= tg[0]) & (b + lag <= tg[-1])
        nxt = np.searchsorted(t, b); val &= np.array([dt[i:max(i + 1, j - 1)].max() <= GAP_MAX if j > i else True for i, j in enumerate(nxt)])
        r = (np.interp(b, t, yaw) - np.interp(a, t, yaw)) - (np.interp(b + lag, tg, cg) - np.interp(a + lag, tg, cg))
        n = int(BIAS_WIN / max(np.median(dt), 1e-3)); res = np.full(len(r), np.nan)
        for i in range(len(r)):
            if val[i]:
                lo = max(0, i - n); res[i] = abs(r[i] - np.median(r[lo:i + 1][val[lo:i + 1]]))
        res = np.degrees(res)
        for i in np.where((res > GYRO_TOL) & (b >= gyro_from))[0]:
            ev.append((float(b[i]), "자이로와 회전 불일치", float(res[i])))
        info = dict(corr=round(float(cc), 3), axis="xyz"[ax], sign=int(sg), lag_s=round(float(lag), 2), calib_s=span,
                    resid_p99=round(float(np.nanpercentile(res, 99)), 2), resid_max=round(float(np.nanmax(res)), 2))
    return sorted(ev), info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--odom", required=True); ap.add_argument("--bag"); ap.add_argument("--out", required=True)
    ap.add_argument("--repair-to", default=None, help="고친 오돔을 담을 새 a1_3_1 폴더(원래 폴더의 samples·far 는 링크)")
    a = ap.parse_args()
    t, x, y, yaw = load_odom(a.odom); gyro = load_gyro(a.bag) if a.bag else None
    ev, info = check(t, x, y, yaw, gyro)
    out = dict(odom=a.odom, bag=a.bag, t0=float(t[0]), gyro=info, thresholds=dict(v_max=V_MAX, w_max=W_MAX, gyro_tol_deg=GYRO_TOL, gyro_win_s=GYRO_WIN, gap_max_s=GAP_MAX),
               events=[dict(t_sec=e[0], rel_sec=round(e[0] - float(t[0]), 2), reason=e[1], value=round(e[2], 2)) for e in ev])
    if a.repair_to:
        X, Y, W, bad = repair(t, x, y, yaw, out["events"], info, gyro)
        out["repaired_steps_rel_sec"] = [round(float(t[k] - t[0]), 2) for k in bad]
        write_repaired(Path(a.odom).resolve().parent, a.odom, a.repair_to, X, Y, W, dict(src=a.odom, steps=out["repaired_steps_rel_sec"], method="예측 이동으로 바꿈"))
        print("고친 걸음", out["repaired_steps_rel_sec"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(Path(a.odom).parent.name, "자이로", info, "| 이상", len(ev), [(e["rel_sec"], e["reason"]) for e in out["events"]][:12])



# ---------- 튐 고치기(2026-10-08): 이상으로 판단한 한 걸음을 예측 이동으로 바꾸고 그 뒤 궤적은 그대로 이어 붙인다 ----------
def repair(t, x, y, yaw, events, gyro_info=None, gyro=None):
    """events 중 순간 이동·회전·자이로 불일치가 가리키는 걸음 i(t[i-1]→t[i])를 예측 이동으로 바꾼다(인과: 직전 걸음 몸체 기준 이동 + 자이로 회전).
    자이로 불일치는 0.3 s 창 안에서 오돔 회전과 자이로 회전 차이가 가장 큰 걸음을 고른다. 오돔 끊김은 고치지 않는다."""
    n = len(t); bad = set()
    gz = cg = None
    if gyro is not None and gyro_info and "axis" in gyro_info:
        tg, G = gyro; g = gyro_info["sign"] * G[:, "xyz".index(gyro_info["axis"])]; lag = gyro_info["lag_s"]
        cg = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(tg))]); gi = lambda a, b: np.interp(b + lag, tg, cg) - np.interp(a + lag, tg, cg)
    for e in events:
        if e["reason"] == "오돔 끊김":
            continue
        i = int(np.searchsorted(t, e["t_sec"] - 1e-6))
        if e["reason"] == "자이로와 회전 불일치" and cg is not None:
            lo = int(np.searchsorted(t, e["t_sec"] - GYRO_WIN - 1e-6)); cand = range(max(lo, 1), min(i + 1, n))
            i = max(cand, key=lambda k: abs((yaw[k] - yaw[k - 1]) - gi(t[k - 1], t[k])), default=i)
        if 1 <= i < n:
            bad.add(i)
    X, Y, W = x.copy(), y.copy(), yaw.copy()
    cth, cx, cy = 0.0, 0.0, 0.0                       # 보정 변환: 고친 자세 = R(cth)·원래 + (cx, cy)
    def apply(k):
        c, s = np.cos(cth), np.sin(cth); return c * x[k] - s * y[k] + cx, s * x[k] + c * y[k] + cy, yaw[k] + cth
    for k in range(1, n):
        if k in bad:
            px, py, pw = X[k - 1], Y[k - 1], W[k - 1]
            if k >= 2 and (k - 1) not in bad:             # 직전 걸음의 몸체 기준 이동을 시간 비율로
                dx, dy = X[k - 1] - X[k - 2], Y[k - 1] - Y[k - 2]; c, s = np.cos(-W[k - 2]), np.sin(-W[k - 2])
                bx, by = c * dx - s * dy, s * dx + c * dy; sc = (t[k] - t[k - 1]) / max(t[k - 1] - t[k - 2], 1e-3)
            else:
                bx = by = 0.0; sc = 0.0
            dth = gi(t[k - 1], t[k]) if cg is not None else (W[k - 1] - W[k - 2]) * sc if k >= 2 else 0.0
            c, s = np.cos(pw), np.sin(pw); X[k] = px + sc * (c * bx - s * by); Y[k] = py + sc * (s * bx + c * by); W[k] = pw + dth
            cth = W[k] - yaw[k]; c, s = np.cos(cth), np.sin(cth); cx = X[k] - (c * x[k] - s * y[k]); cy = Y[k] - (s * x[k] + c * y[k])
        else:
            X[k], Y[k], W[k] = apply(k)
    return X, Y, W, sorted(bad)


def write_repaired(src_dir, odom_csv, dst_dir, X, Y, W, note):
    """원래 a1_3_1 폴더는 그대로 두고 새 폴더에 samples·far 는 링크, odom_link.csv 만 고친 사본."""
    import os, math
    src, dst = Path(src_dir), Path(dst_dir); dst.mkdir(parents=True, exist_ok=False)
    for p in src.iterdir():
        if p.name != "odom_link.csv" and not p.name.startswith("INJECTION"):
            os.symlink(p.resolve(), dst / p.name)
    rows = list(csv.DictReader(open(odom_csv)))
    with open(dst / "odom_link.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r, a, b, c in zip(rows, X, Y, W):
            r = dict(r); r["fastlio_x"] = f"{a:.4f}"; r["fastlio_y"] = f"{b:.4f}"; r["fastlio_yaw_rad"] = f"{math.atan2(math.sin(c), math.cos(c)):.5f}"; w.writerow(r)
    json.dump(note, open(dst / "REPAIR.json", "w"), ensure_ascii=False, indent=1)


def _gyro_integrator(gyro, info):
    tg, G = gyro; g = info["sign"] * G[:, "xyz".index(info["axis"])]; lag = info["lag_s"]
    cg = np.concatenate([[0], np.cumsum((g[1:] + g[:-1]) / 2 * np.diff(tg))])
    return lambda a, b: np.interp(b + lag, tg, cg) - np.interp(a + lag, tg, cg)


class OdomGuard:
    """실시간판(2026-10-08). FAST-LIO 자세(odom_link 규약)와 자이로를 받는 대로 넣으면, DELAY 초 늦게 판단이 끝난 자세를 고쳐서 내준다.
    판단은 check()·repair() 를 최근 BUF 초 버퍼에 그대로 돌린다(일괄 처리와 같은 규칙). 자이로 축·부호·시각 차이는 처음 60 s 로 한 번 정한다.
        g = OdomGuard(); g.add_gyro(t, gx, gy, gz); out = g.add_pose(t, x, y, yaw)   # out = [(t, x, y, yaw, 고쳤나), ...] 확정된 것만
    """
    DELAY, BUF = GYRO_WIN + 0.1, BIAS_WIN + 3.0

    def __init__(self, use_gyro=True):
        self.P, self.G, self.use_gyro, self.info = [], [], use_gyro, None
        self.n_done = 0; self.out = []; self.fixed = []

    def add_gyro(self, t, gx, gy, gz):
        self.G.append((t, gx, gy, gz))

    def add_pose(self, t, x, y, yaw):
        if self.P:
            yaw = self.P[-1][3] + float(np.angle(np.exp(1j * (yaw - self.P[-1][3]))))
        self.P.append((t, x, y, yaw)); P = np.array(self.P); tt = P[:, 0]
        span = next((sp for sp in (60.0, 120.0, 180.0) if sp not in getattr(self, "tried", set()) and tt[-1] - tt[0] >= sp), None)
        if self.use_gyro and (self.info is None or "skipped" in self.info) and span and len(self.G) > 10:
            self.tried = getattr(self, "tried", set()) | {span}
            G = np.array(self.G); cc, ax, sg, lag = fit_gyro(tt, P[:, 3], G[:, 0], G[:, 1:], span=span)
            self.info = dict(corr=float(cc), axis="xyz"[ax], sign=int(sg), lag_s=float(lag), calib_s=span) if gyro_ok(cc, fit_gyro.other) else dict(skipped=True, corr=float(cc))
        i0 = int(np.searchsorted(tt, tt[-1] - self.BUF)); Pb = P[i0:]
        gyro = None
        if self.info and "axis" in self.info:
            G = np.array(self.G); gm = G[:, 0] >= Pb[0, 0] - 2.0; gyro = (G[gm, 0], G[gm, 1:])
        ev, _ = check_with_info(Pb[:, 0], Pb[:, 1], Pb[:, 2], Pb[:, 3], gyro, self.info)
        ev = [dict(t_sec=e[0], reason=e[1], value=e[2]) for e in ev]
        _, _, _, bad = repair(Pb[:, 0], Pb[:, 1], Pb[:, 2], Pb[:, 3], ev, self.info, gyro)
        bad = {i0 + k for k in bad}
        gi = _gyro_integrator(gyro, self.info) if gyro is not None else None
        new = []
        while self.n_done < len(tt) and tt[self.n_done] <= tt[-1] - self.DELAY:
            k = self.n_done; new.append(self._emit(k, P, k in bad and k >= 1, gi)); self.n_done += 1
        return new

    def _emit(self, k, P, bad, gi):
        t, x, y, yaw = P[k]
        if not self.out:
            self.cth = self.cx = self.cy = 0.0
        if bad:
            t1, X1, Y1, W1, _ = self.out[-1]
            if len(self.out) >= 2 and not self.out[-1][4]:
                t2, X2, Y2, W2, _ = self.out[-2]; dx, dy = X1 - X2, Y1 - Y2; c, s = np.cos(-W2), np.sin(-W2)
                bx, by = c * dx - s * dy, s * dx + c * dy; sc = (t - t1) / max(t1 - t2, 1e-3)
                dth = gi(t1, t) if gi is not None else (W1 - W2) * sc
            else:
                bx = by = sc = 0.0; dth = gi(t1, t) if gi is not None else 0.0
            c, s = np.cos(W1), np.sin(W1); X = X1 + sc * (c * bx - s * by); Y = Y1 + sc * (s * bx + c * by); W = W1 + dth
            self.cth = W - yaw; c, s = np.cos(self.cth), np.sin(self.cth); self.cx = X - (c * x - s * y); self.cy = Y - (s * x + c * y)
            self.fixed.append(float(t))
        else:
            c, s = np.cos(self.cth), np.sin(self.cth); X, Y, W = c * x - s * y + self.cx, s * x + c * y + self.cy, yaw + self.cth
        self.out.append((float(t), float(X), float(Y), float(W), bool(bad)))
        return self.out[-1]


def check_with_info(t, x, y, yaw, gyro, info):
    """check() 와 같되 자이로 축·부호·시각 차이를 미리 정한 값으로 쓴다(실시간판용)."""
    if gyro is None or info is None or "axis" not in info:
        return check(t, x, y, yaw, None)
    global fit_gyro
    keep = fit_gyro; fit_gyro = lambda *a, **k: (info["corr"], "xyz".index(info["axis"]), info["sign"], info["lag_s"]); fit_gyro.other = 0.0
    try:
        return check(t, x, y, yaw, gyro, calibrated=True)
    finally:
        fit_gyro = keep

if __name__ == "__main__":
    main()
