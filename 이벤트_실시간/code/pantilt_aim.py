#!/usr/bin/env python3
"""RealSense RGB 픽셀을 팬틸트 서보 명령각으로 바꾼다.

두 카메라가 떨어져 있어서 같은 픽셀이라도 거리에 따라 돌려야 할 각도가 다르다.
그래서 픽셀과 함께 거리(RealSense depth, 없으면 가정값)를 받는다.

좌표는 전부 로봇 기준 앞+x, 왼쪽+y, 위+z, 단위 m다. 값은
configs/pantilt_calib.json 에서 읽는다.

계산 순서
    1. 픽셀 + 거리 → RealSense 광학좌표의 3D 점
    2. RealSense → 팬축 기준 좌표
    3. 팬각: 카메라의 좌우 치우침까지 반영해 목표가 광축 위에 오도록
    4. 틸트각: 팬을 돌린 뒤, 틸트축과 렌즈 사이 간격까지 반영
    5. 실측 영점과 방향으로 서보 명령각

사용 예:
    python3 scripts/pantilt_aim.py --self-test
    python3 scripts/pantilt_aim.py --pixel 420 180 --dist 3.0
    python3 scripts/pantilt_aim.py --parallax --assume 3.0
    python3 scripts/pantilt_aim.py --parallax --what-if-offset -0.15 0 0.35
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pantilt_calib.json"


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    # 오른손 좌표계. 양수면 앞(+x)이 아래(-z)로 숙여진다.
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def vec(d: dict, fill_missing: bool = False) -> np.ndarray:
    vals = []
    for k in ("x", "y", "z"):
        v = d.get(k)
        if v is None:
            if not fill_missing:
                raise ValueError(k)
            v = 0.0
        vals.append(float(v))
    return np.array(vals)


class Aim:
    def __init__(self, cfg: dict, fill_missing: bool = False):
        rs = cfg["realsense_color"]
        self.fx, self.fy, self.cx, self.cy = rs["fx"], rs["fy"], rs["cx"], rs["cy"]
        missing = []
        for key in ("pan_axis_from_realsense", "tilt_axis_from_pan_axis", "camera_from_tilt_axis"):
            for k in ("x", "y", "z"):
                if cfg[key].get(k) is None:
                    missing.append(f"{key}.{k}")
        if missing and not fill_missing:
            raise SystemExit("실측 전 값이 있습니다: " + ", ".join(missing)
                             + "\n  configs/pantilt_calib.json 을 채우거나 --what-if-offset 으로 가정값을 주세요.")
        self.missing = missing
        self.t_pan = vec(cfg["pan_axis_from_realsense"], True)
        self.t_tilt = vec(cfg["tilt_axis_from_pan_axis"], True)
        self.c_cam = vec(cfg["camera_from_tilt_axis"], True)
        m = cfg["realsense_mount_deg"]
        self.r_mount = (rot_z(math.radians(m["yaw"])) @ rot_y(math.radians(m["pitch"]))
                        @ rot_x(math.radians(m["roll"])))
        s = cfg["servo"]
        self.pan_zero, self.pan_gain = s["pan_zero_deg"], s["pan_deg_per_left_deg"]
        self.tilt_zero, self.tilt_gain = s["tilt_zero_deg"], s["tilt_deg_per_up_deg"]
        self.pan_limit, self.tilt_limit = s["pan_limit_deg"], s["tilt_limit_deg"]

    def pixel_to_point(self, u: float, v: float, depth: float) -> np.ndarray:
        """RealSense depth는 광축 방향 거리(z)다. 광학좌표(오른쪽·아래·앞)를 로봇축으로 바꾼다."""
        xo = (u - self.cx) / self.fx * depth
        yo = (v - self.cy) / self.fy * depth
        zo = depth
        p_rs = np.array([zo, -xo, -yo])
        return self.r_mount @ p_rs - self.t_pan

    def point_to_angles(self, p: np.ndarray) -> tuple[float, float]:
        """팬축 기준 점을 광축 위에 올리는 팬각·틸트각(라디안). 왼쪽·위가 양수."""
        lateral = self.t_tilt[1] + self.c_cam[1]
        rho = math.hypot(p[0], p[1])
        if rho <= abs(lateral):
            raise ValueError("목표가 팬축에 너무 가깝습니다")
        pan = math.atan2(p[1], p[0]) - math.asin(lateral / rho)

        q = rot_z(-pan) @ p - self.t_tilt
        r = math.hypot(q[0], q[2])
        if r <= abs(self.c_cam[2]):
            raise ValueError("목표가 틸트축에 너무 가깝습니다")
        tilt = math.atan2(q[2], q[0]) - math.asin(self.c_cam[2] / r)
        return pan, tilt

    def angles_to_servo(self, pan: float, tilt: float) -> tuple[float, float]:
        return (self.pan_zero + self.pan_gain * math.degrees(pan),
                self.tilt_zero + self.tilt_gain * math.degrees(tilt))

    def aim(self, u: float, v: float, depth: float) -> tuple[float, float]:
        return self.angles_to_servo(*self.point_to_angles(self.pixel_to_point(u, v, depth)))

    def optical_to_servo(self, p_opt) -> tuple[float, float]:
        """RealSense 광학좌표(오른쪽·아래·앞, m)의 3D 점을 서보 명령각으로. ArUco 위치를 바로 넣을 때 쓴다."""
        x, y, z = (float(v) for v in p_opt)
        p = self.r_mount @ np.array([z, -x, -y]) - self.t_pan
        return self.angles_to_servo(*self.point_to_angles(p))

    def optical_axis_miss(self, p: np.ndarray, pan: float, tilt: float) -> float:
        """검산용. 계산한 각도로 돌렸을 때 목표가 광축에서 몇 m 떨어지는지."""
        cp, sp = math.cos(tilt), math.sin(tilt)
        center = self.t_tilt + np.array([self.c_cam[0] * cp - self.c_cam[2] * sp,
                                         self.c_cam[1],
                                         self.c_cam[0] * sp + self.c_cam[2] * cp])
        direction = np.array([cp, 0.0, sp])
        center, direction = rot_z(pan) @ center, rot_z(pan) @ direction
        d = p - center
        return float(np.linalg.norm(d - direction * float(d @ direction)))


def self_test() -> int:
    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    zero = {"x": 0.0, "y": 0.0, "z": 0.0}
    for key in ("pan_axis_from_realsense", "tilt_axis_from_pan_axis", "camera_from_tilt_axis"):
        base[key] = dict(zero)
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) < tol
        ok &= good
        print(f"  {'통과' if good else '실패'}  {name}: {got:.4f} (기대 {want:.4f})")

    a = Aim(base)
    pan, tilt = a.aim(a.cx, a.cy, 3.0)
    check("간격 0, 화면 중앙 → 팬", pan, 100.0)
    check("간격 0, 화면 중앙 → 틸트", tilt, 160.0)
    pan, _ = a.aim(a.cx + a.fx * math.tan(math.radians(10)), a.cy, 3.0)
    check("간격 0, 오른쪽 10° → 팬 각도는 줄어야 함", pan, 90.0)
    _, tilt = a.aim(a.cx, a.cy - a.fy * math.tan(math.radians(10)), 3.0)
    check("간격 0, 위쪽 10° → 틸트 각도는 줄어야 함", tilt, 150.0)

    cfg = json.loads(json.dumps(base))
    cfg["pan_axis_from_realsense"]["z"] = 0.35
    a = Aim(cfg)
    _, tilt = a.aim(a.cx, a.cy, 3.0)
    check("팬축이 35cm 위, 3m 정면 → 아래로 6.65° (틸트 166.65)",
          tilt, 160.0 + math.degrees(math.atan2(0.35, 3.0)))

    cfg = json.loads(json.dumps(base))
    cfg["camera_from_tilt_axis"]["z"] = 0.05
    a = Aim(cfg)
    _, tilt = a.aim(a.cx, a.cy, 3.0)
    check("렌즈가 틸트축보다 5cm 위 → 살짝 아래로", tilt,
          160.0 + math.degrees(math.asin(0.05 / 3.0)))

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        cfg = json.loads(json.dumps(base))
        for key, scale in (("pan_axis_from_realsense", 0.4), ("tilt_axis_from_pan_axis", 0.1),
                           ("camera_from_tilt_axis", 0.08)):
            cfg[key] = dict(zip("xyz", map(float, rng.uniform(-scale, scale, 3))))
        cfg["realsense_mount_deg"] = {"yaw": float(rng.uniform(-5, 5)),
                                      "pitch": float(rng.uniform(-10, 10)),
                                      "roll": float(rng.uniform(-3, 3))}
        a = Aim(cfg)
        u, v, d = rng.uniform(0, 640), rng.uniform(0, 480), rng.uniform(1.0, 6.0)
        p = a.pixel_to_point(u, v, d)
        pan, tilt = a.point_to_angles(p)
        worst = max(worst, a.optical_axis_miss(p, pan, tilt))
    good = worst < 1e-9
    ok &= good
    print(f"  {'통과' if good else '실패'}  무작위 500건 역검산: 목표와 광축 최대 거리 {worst:.2e} m")
    print("자체 시험", "전부 통과" if ok else "실패 있음")
    return 0 if ok else 1


def parallax_table(a: Aim, assume: float, lens_mm: float | None, sensor_mm: float) -> None:
    """거리를 가정했을 때 실제 거리가 다르면 몇 도 틀리는지."""
    dists = [1.0, 1.5, 2.0, 3.0, 4.5, 6.0]
    pixels = {"중앙": (a.cx, a.cy), "왼쪽 위": (80, 60), "오른쪽 위": (560, 60)}
    print(f"거리를 {assume:.1f} m로 가정했을 때의 조준 오차 (도, 팬/틸트)")
    print(f"{'픽셀':>8} " + " ".join(f"{d:>11.1f}m" for d in dists))
    for name, (u, v) in pixels.items():
        ref = a.aim(u, v, assume)
        cells = []
        for d in dists:
            got = a.aim(u, v, d)
            cells.append(f"{ref[0] - got[0]:+5.1f}/{ref[1] - got[1]:+5.1f}")
        print(f"{name:>8} " + " ".join(f"{c:>12}" for c in cells))
    if lens_mm:
        half = math.degrees(math.atan(sensor_mm / 2 / lens_mm))
        print(f"\n팬틸트 카메라 수평 반화각 약 {half:.1f}° (렌즈 {lens_mm} mm). "
              "오차가 이보다 작으면 글자는 화면 안에 들어온다.")
    else:
        print("\n렌즈 초점거리를 알면 이 오차가 화면 안에 들어오는지 판정할 수 있다.")


def coverage_table(a: Aim, width: int, height: int, dists) -> None:
    """RealSense 화면 어느 자리에 글자가 떠야 팬틸트가 그쪽을 볼 수 있는지.

    서보 범위가 화면 화각보다 좁으면 가장자리는 원리적으로 못 본다. 그걸 미리 안다.
    """
    lo_p, hi_p = a.pan_limit
    lo_t, hi_t = a.tilt_limit
    us = np.linspace(0, width - 1, 41)
    vs = np.linspace(0, height - 1, 31)
    print(f"서보 범위 팬 {lo_p:.0f}~{hi_p:.0f}°, 틸트 {lo_t:.0f}~{hi_t:.0f}°")
    print(f"{'거리':>6} {'조준 가능':>9}   {'막히는 곳':<24} {'필요한 최대 각도':>18}")
    for d in dists:
        okc = fail = 0
        blocked = set()
        pans, tilts = [], []
        for v in vs:
            for u in us:
                try:
                    pan, tilt = a.aim(float(u), float(v), d)
                except ValueError:
                    fail += 1
                    continue
                pans.append(pan)
                tilts.append(tilt)
                if lo_p <= pan <= hi_p and lo_t <= tilt <= hi_t:
                    okc += 1
                else:
                    fail += 1
                    if pan < lo_p:
                        blocked.add("오른쪽")
                    if pan > hi_p:
                        blocked.add("왼쪽")
                    if tilt < lo_t:
                        blocked.add("위")
                    if tilt > hi_t:
                        blocked.add("아래")
        total = okc + fail
        need = (f"팬 {min(pans):.0f}~{max(pans):.0f}, 틸트 {min(tilts):.0f}~{max(tilts):.0f}"
                if pans else "-")
        print(f"{d:5.1f}m {100.0 * okc / total:8.1f}%   "
              f"{'전부 가능' if not blocked else ' '.join(sorted(blocked)) + ' 가장자리':<24} {need:>18}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="RealSense 화면 각 자리를 서보 범위 안에서 조준할 수 있는지 표로 본다")
    ap.add_argument("--pixel", nargs=2, type=float, metavar=("U", "V"))
    ap.add_argument("--dist", type=float, default=3.0, help="광축 방향 거리 m. depth가 없으면 가정값")
    ap.add_argument("--parallax", action="store_true", help="거리 가정 오차표")
    ap.add_argument("--assume", type=float, default=3.0)
    ap.add_argument("--what-if-offset", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    help="실측 전 가정값으로 팬축 위치를 넣어 본다 (m)")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    what_if = args.what_if_offset is not None
    if what_if:
        cfg["pan_axis_from_realsense"] = dict(zip("xyz", args.what_if_offset))
    a = Aim(cfg, fill_missing=what_if)
    if a.missing:
        print("※ 가정값 계산입니다. 비어 있는 값은 0으로 뒀습니다: " + ", ".join(a.missing))

    if args.pixel:
        pan, tilt = a.aim(args.pixel[0], args.pixel[1], args.dist)
        lo_p, hi_p = a.pan_limit
        lo_t, hi_t = a.tilt_limit
        print(f"픽셀 ({args.pixel[0]:.0f}, {args.pixel[1]:.0f}), 거리 {args.dist:.2f} m")
        print(f"  팬 {pan:.1f}°  {'범위 안' if lo_p <= pan <= hi_p else f'범위 밖 ({lo_p}~{hi_p})'}")
        print(f"  틸트 {tilt:.1f}°  {'범위 안' if lo_t <= tilt <= hi_t else f'범위 밖 ({lo_t}~{hi_t})'}")

    if args.coverage:
        rs = cfg["realsense_color"]
        coverage_table(a, rs["width"], rs["height"], [1.0, 1.5, 2.0, 3.0, 4.5, 6.0])

    if args.parallax:
        lens = cfg["pantilt_lens"]
        parallax_table(a, args.assume, lens.get("focal_mm"), lens["sensor_width_mm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
