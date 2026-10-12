#!/usr/bin/env python3
"""조준 콘솔. 리얼센스 화면의 한 점을 누르면 팬틸트가 그리로 돌고 초점을 맞춘다.

CAMERA_AIM_JOB 의 시험판이다. 지금은 브라우저 클릭으로 목표를 주고, 나중에는
검출기(417 같은 호실 번호 후보)가 같은 주소로 부른다.

    GET /aim?u=<x>&v=<y>     리얼센스 컬러 640x480 기준 픽셀. &focus=0 이면 초점은 안 건드린다
                             &z=<m> 거리를 직접 줌(화면 밖 픽셀도 가능), &strict=1 가동 범위 밖이면 조준 안 함
    GET /aim_marker          리얼센스가 보고 있는 마커로 조준(마커의 실제 거리를 씀, 검증용)
    GET /focus               서보는 그대로 두고 초점만 다시
    GET /focus_profile       초점 모터를 일정하게 끊어 돌리며 선명도·화면 변화를 기록(진단)
    GET /focus_speed_test    속도(PWM)별로 양쪽 0.8초씩 돌려 렌즈가 실제로 움직이는지 잰다(진단)
    GET /home                정면(팬 110, 틸트 160)
    GET /trim                모인 마커 검증 결과의 평균만큼 팬·틸트 영점을 고쳐 새 파일로 저장
    GET /pin_test?pin=3|5    초점 드라이버 입력 핀을 1초 HIGH (배선 확인용)
    GET /goto?pan=&tilt=     그 각도로 바로 간다(조준 반복성 시험의 출발점 만들기)
    GET /tilt_sweep?a=&b=    틸트를 a에서 b까지 1°씩 한 방향으로 올리며 마커 세로 위치를 잰다(진단)

한 번 조준할 때 하는 일
  1. 거리는 --assume-m(기본 3 m)로 가정한다. 2026-09-19 사용자: 깊이 토픽은 안 쓴다.
     --use-depth 를 주면 그 픽셀 둘레의 깊이 중앙값(컬러에 정렬된 depth)을 쓴다
  2. 캘리브레이션 결과로 팬·틸트 각도 계산 (pantilt_aim.Aim)
  3. 두 축을 조금 물렀다가 다가간다. 캘리브레이션 샘플을 모을 때와 같은 쪽으로
     기어가 물리게 해서, 계산이 가정한 상태와 맞춘다
  4. 팬틸트 화면 가운데에서 자동초점. 계속 돌리다 정점을 지나면 멈추고 지나친 만큼
     되돌아온다(2026-09-16 autofocus.py --continuous 방식). 최고치를 쫓지 않고
     최고의 70%면 받아들인다(사용자: 글자만 읽히면 된다)
  5. 고해상도 한 장을 out/aim/ 에 저장. 화면에 마커가 보이면 가운데서 몇 도
     벗어났는지도 잰다(조준 검증)

카메라·나노·리얼센스를 이 프로그램 하나가 쥔다. calib_console.py 와 동시에 못 돈다.

    python3 scripts/aim_console.py            # Go2, ROS 환경을 불러온 셸에서
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import io
import re
import statistics
import glob
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autofocus as af  # noqa: E402  2026-09-16 단계식 자동초점을 그대로 쓴다
import calib_console as cc  # noqa: E402  영상 전송, 나노, 리얼센스 수신을 그대로 쓴다
from aruco_realsense_probe import find_known_marker, make_detector, parse_sizes  # noqa: E402
from pantilt_aim import Aim  # noqa: E402
from pantilt_devices import find_camera, find_nano_port  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 조준에 쓰는 서보 범위. 펌웨어는 0~180을 다 허용하므로 여기가 안전장치다.
# 실행할 때 --pan-lim/--tilt-lim 으로 바꾼다(가동 범위 시험 2026-09-19).
# 2026-09-19 5°씩 실측: 정면(110) 기준 오른쪽 90°·왼쪽 70°(서보 끝), 수평(160) 기준 위 약 69°·아래 약 23°(서보 끝)
PAN_LIM = [20.0, 180.0]
TILT_LIM = [100.0, 180.0]
# 정면·수평 서보각. 카메라를 다시 끼우면 바뀌므로 고정값을 쓰지 않고 캘리브 설정의 영점으로 채운다
# (2026-09-28: 다시 끼운 뒤 정면이 110→95.4로 옮겨 "정면으로"가 13° 왼쪽을 봄). 아래는 설정을 읽기 전 기본값.
HOME = [110.0, 160.0]
SLEW_DEG_PER_S = 60.0     # 펌웨어가 서보를 옮기는 속도


def sync_home(nano) -> None:
    """캘리브의 정면·수평을 나노 EEPROM 부팅 자세로 써 둔다. 그래야 콘솔 없이 켜도 정면을 본다.

    2026-09-28 펌웨어에 W 명령을 넣었다. 옛 펌웨어는 W를 몰라 ERR를 돌려주므로 그때는 로그만 남긴다.
    """
    def ask(line):
        nano.ser.reset_input_buffer()
        nano.ser.write((line + "\n").encode())
        nano.ser.flush()
        for _ in range(5):
            r = nano.ser.readline().decode(errors="replace").strip()
            if r.startswith(("OK", "ERR")):
                return r
        return ""
    want = (round(HOME[0], 1), round(HOME[1], 1))
    with nano.lock:
        cur = ask("W")
        m = re.search(r"home pan=([\d.]+) tilt=([\d.]+)", cur)
        if not m:
            log(f"나노 부팅 자세를 못 읽음(펌웨어가 W 명령을 모름): {cur or '응답 없음'}")
            return
        if (float(m.group(1)), float(m.group(2))) == want:
            return
        log(f"나노 부팅 자세 {m.group(1)}/{m.group(2)} → {want[0]}/{want[1]}: {ask(f'W{want[0]:.1f},{want[1]:.1f}')}")

def to_plain(o):
    """json 이 못 쓰는 numpy 값을 파이썬 값으로. 2026-09-19 영점 보정 뒤 np.bool_ 때문에
    기록 단계에서 조준·초점 작업이 '오류'로 끝났다."""
    return o.item() if hasattr(o, "item") else str(o)


LOG = collections.deque(maxlen=16)
LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with LOG_LOCK:
        LOG.append(line)


# ---------------------------------------------------------------- 팬틸트 영상 통계

class Stats:
    """팬틸트 카메라의 최신 프레임과 가운데 영역의 선명도. 메인 루프가 매 프레임 채운다.

    초점 판단은 조준 작업 스레드가 하므로, 영상 읽기를 멈추지 않고 최신 값만 가져다 쓴다.
    """

    def __init__(self, roi_frac: float):
        self.roi_frac = roi_frac
        self.lock = threading.Lock()
        self.seq = 0
        self.frame = None
        self.sharp = 0.0
        self.small = None

    def update(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        rw, rh = int(w * self.roi_frac), int(h * self.roi_frac)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        g = cv2.cvtColor(cv2.resize(frame[y0:y0 + rh, x0:x0 + rw], (480, 270)), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        small = cv2.resize(g, (160, 90)).astype(np.float32)
        with self.lock:
            self.seq += 1
            self.frame, self.sharp, self.small = frame, sharp, small

    def get(self):
        with self.lock:
            return self.seq, self.sharp

    def fresh(self, after_seq: int, timeout: float = 1.0):
        """after_seq 다음에 찍힌 프레임의 값이 올 때까지 기다린다."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            seq, s = self.get()
            if seq > after_seq:
                return seq, s
            time.sleep(0.004)
        return self.get()

    def measure(self, n: int = 3) -> float:
        """새로 찍힌 프레임 n장의 선명도 평균. 한 장은 잡음이 크다."""
        seq, _ = self.get()
        vals = []
        for _ in range(n):
            seq, s = self.fresh(seq)
            vals.append(s)
        return float(np.mean(vals))

    def small_copy(self):
        with self.lock:
            return None if self.small is None else self.small.copy()

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


# ---------------------------------------------------------------- 초점

class AfCam:
    """autofocus.stepwise 가 기대하는 카메라(measure, set_note, latest_sharp)를 Stats 위에 만든다.

    선명도는 2026-09-16 과 똑같이 잰다: 1280x720 화면 가운데 roi 영역을 원래 크기 그대로
    라플라시안 분산, 60ms 간격 두 장의 중앙값, 움직임 비교용은 160x120. 재는 방식이 다르면
    stepwise 안의 문턱값(5, 1.5, 3)이 맞지 않는다(2026-09-19 연속 방식 이식 실패 교훈).
    """

    def __init__(self, stats: "Stats", roi: float = 0.5, samples: int = 2):
        self.stats, self.roi, self.samples = stats, roi, samples
        self.note = ""

    def _one(self):
        frame = self.stats.snapshot()
        f = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        h, w = f.shape[:2]
        rw, rh = int(w * self.roi), int(h * self.roi)
        x0, y0 = (w - rw) // 2, (h - rh) // 2
        gray = cv2.cvtColor(f[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2GRAY)
        return (float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                cv2.resize(gray, (160, 120)).astype(np.float32))

    def measure(self):
        vals, img = [], None
        seq, _ = self.stats.get()
        for _ in range(self.samples):
            seq, _ = self.stats.fresh(seq)
            s, img = self._one()
            vals.append(s)
            time.sleep(0.06)
        return statistics.median(vals), img

    def set_note(self, text: str) -> None:
        self.note = text

    def latest_sharp(self) -> float:
        return self._one()[0]


class AfFocus:
    """autofocus.Focus 와 같은 모양(speed, move, start, stop)을 콘솔이 쥔 나노 위에 만든다."""

    def __init__(self, nano, speed: int):
        self.nano, self.speed = nano, speed

    def move(self, direction: int, ms: int) -> None:
        self.nano.cmd(f"F{self.speed * direction},{int(ms)}")
        time.sleep(ms / 1000.0 + 0.35)   # 16일과 같다: 구동 시간 + 진동 가라앉을 시간

    def start(self, direction: int, ms: int) -> None:
        self.nano.cmd(f"F{self.speed * direction},{int(ms)}")

    def stop(self) -> None:
        self.nano.cmd("F0")


class _LogLines(io.TextIOBase):
    """stepwise 가 print 하는 줄을 파일 로그로 보내고, 중요한 줄은 웹 로그에도 올린다."""

    def __init__(self, real):
        self.real, self.buf = real, ""

    def write(self, s):
        self.real.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.strip()
            if line and ("단계" in line or "→" in line or "끝에 닿" in line or line.startswith("시작")
                         or line.startswith("서기")):
                with LOG_LOCK:
                    LOG.append(time.strftime("%H:%M:%S ") + "  " + line)
        return len(s)

    def flush(self):
        self.real.flush()


class Focuser:
    """초점 모터(나노 F 명령)를 돌리며 Stats 의 선명도를 본다. 위치는 기억하지 않는다.

    2026-09-16 실측: PWM 140~180 이 잘 물고 255 는 헛돈다. 방향을 바꾸면 60~320ms 헛돈다.
    조리개 F5.6(2026-09-19)이라 F1.4 때보다 정점이 넓어 연속 훑기로 충분히 선다.
    """

    def __init__(self, nano, stats: Stats, speed: int = 160, af: dict = None):
        self.nano, self.stats, self.speed = nano, stats, speed
        # 자동초점 설정. 복도에서 바꿔 볼 수 있게 실행 옵션(--af-*)으로 받는다.
        self.af = af or {"speed": 180, "coarse_ms": 300, "fine_speed": 160, "fine_ms": 50, "roi": 0.5}

    def start(self, direction: int, max_ms: int) -> None:
        self.nano.cmd(f"F{direction * self.speed},{int(max_ms)}")

    def stop(self) -> None:
        self.nano.cmd("F0")

    def run_for(self, direction: int, ms: float) -> None:
        self.nano.cmd(f"F{direction * self.speed},{int(ms)}")
        time.sleep(ms / 1000.0 + 0.3)     # 구동 + 진동 가라앉기

    def takeup_lash(self, direction: int, probe_ms: int = 60, max_probes: int = 8,
                    thresh: float = 2.0) -> None:
        """방향을 바꾼 직후 기어 유격을 메운다. 화면이 변하기 시작할 때까지 짧게 민다."""
        prev = self.stats.small_copy()
        for _ in range(max_probes):
            self.run_for(direction, probe_ms)
            self.stats.measure(1)
            cur = self.stats.small_copy()
            if prev is not None and cur is not None and float(np.mean(np.abs(cur - prev))) > thresh:
                return
            prev = cur

    def autofocus(self, direction: int = 1) -> dict:
        """2026-09-16 autofocus.py 의 단계식(stepwise)을 그대로 부른다.

        설정은 그날 마지막으로 글자가 읽혔던 실행 값이다: 속도 180, 한 걸음 300ms,
        가운데 절반(roi 0.5). 미세만 160/50ms(오늘 160 아래는 모터가 거의 안 움직임). 2026-09-19 에 새로 짠 연속 방식은 16일에도
        실패했던 방식이라 버렸다(사용자: "140 찾던 알고리즘이 있었는데 왜 퇴보했냐").
        """
        t0 = time.time()
        cam = AfCam(self.stats, roi=self.af["roi"])
        foc = AfFocus(self.nano, self.af["speed"])
        try:
            # 많이 흐리면(선명도 < 8) 한 걸음에 화면이 거의 안 변해서 stepwise 가 "끝에 닿았다"로
            # 오판하고 방향만 뒤집는다(2026-09-19 선명도 4에서 못 벗어남, 모의시험에서도 재현).
            # 그때만 큰 걸음(0.5초)으로 먼저 선명해지는 쪽을 찾아 stepwise 에 넘긴다.
            start0, _ = cam.measure()
            if start0 < 8.0:
                log(f"  많이 흐림({start0:.0f}) → 큰 걸음으로 선명해지는 쪽부터 찾습니다")
                found = False
                for d, steps in ((direction, 24), (-direction, 48)):
                    for _ in range(steps):
                        foc.move(d, 500)
                        s, _ = cam.measure()
                        if s > max(8.0, start0 * 2.0):
                            found, direction = True, d
                            break
                    if found:
                        log(f"  {'+' if d > 0 else '-'} 쪽에서 선명도 {s:.0f}로 올라옴 → 단계식 시작")
                        break
                if not found:
                    log("  끝에서 끝까지 가도 선명해지지 않았습니다(가운데에 무늬가 없거나 너무 어두움)")
            with contextlib.redirect_stdout(_LogLines(sys.stdout)):
                # 미세는 160/50ms: 2026-09-19 에는 160 아래에서 모터가 거의 안 움직였다
                start, final, best = af.stepwise(cam, foc, direction, speed=self.af["speed"],
                                                 coarse_ms=self.af["coarse_ms"],
                                                 fine_speed=self.af["fine_speed"],
                                                 fine_ms=self.af["fine_ms"])
        finally:
            self.stop()
        result = {"start": float(start), "final": float(final), "best": float(best),
                  "ok": bool(final >= best * 0.5), "sec": time.time() - t0}
        log(f"  초점 끝: {start:.0f} → {final:.0f} (구간 최고 {best:.0f}), {result['sec']:.1f}초")
        return result


# ---------------------------------------------------------------- 조준 작업

class Aimer:
    def __init__(self, args, nano, rs, stats: Stats, aim: Aim, cfg: dict, cfg_path: str):
        self.args, self.nano, self.rs, self.stats, self.aim = args, nano, rs, stats, aim
        self.cfg, self.cfg_path = cfg, cfg_path
        self.checks = []          # 마커 검증에서 잰 (가로°, 세로°) 벗어남. 영점 보정에 쓴다
        self.focuser = Focuser(nano, stats, args.focus_speed,
                               af={"speed": args.af_speed, "coarse_ms": args.af_coarse_ms,
                                   "fine_speed": args.af_fine_speed, "fine_ms": args.af_fine_ms,
                                   "roi": args.af_roi})
        self.busy = threading.Lock()
        self.job = 0
        self.status = "대기 중. 리얼센스 화면을 누르면 그 자리로 조준합니다"
        self.det = make_detector(args.dict, args.poly_accuracy)
        self.sizes = parse_sizes(args.sizes)
        os.makedirs(os.path.join(ROOT, "out", "aim"), exist_ok=True)

    def depth_at(self, u: int, v: int, half: int = 4):
        d = self.rs.depth
        if d is None or time.time() - self.rs.depth_t > 1.0:
            return None
        h, w = d.shape
        win = d[max(0, v - half):min(h, v + half + 1), max(0, u - half):min(w, u + half + 1)]
        win = win[win > 0]
        return float(np.median(win)) / 1000.0 if win.size else None

    def move_to(self, pan: float, tilt: float, lash: float = None) -> None:
        """두 축 모두 물렀다가 한쪽에서 다가간다. 기어가 매번 같은 쪽에 물리게 한다.

        2026-09-19 실측: 같은 틸트 148 명령에 어디서 왔느냐에 따라 약 4° 다른 곳에 섰다.
        1.5°만 물러서는 유격이 다 안 먹혀 영점 보정이 크게 넘쳤다. 그래서 --lash-deg(기본 5°)
        만큼 물러난다.
        """
        lash = self.args.lash_deg if lash is None else lash
        st = self.nano.cmd("?")
        cur_p, cur_t = st.get("pan", pan), st.get("tilt", tilt)
        self.nano.cmd(f"P{pan - lash:.1f}")
        self.nano.cmd(f"T{tilt - lash:.1f}")
        time.sleep(max(abs(pan - lash - cur_p), abs(tilt - lash - cur_t)) / SLEW_DEG_PER_S + 0.3)
        self.nano.cmd(f"P{pan:.1f}")
        self.nano.cmd(f"T{tilt:.1f}")
        time.sleep(lash / SLEW_DEG_PER_S + 0.5)

    def marker_error(self, frame, dist):
        """화면에 마커가 보이면 가운데서 벗어난 각도(가로, 세로). 오른쪽·아래가 +."""
        # 초점이 나간 화면에서는 원래 크기로 못 찾는 일이 많다. 줄이면 번진 폭도 픽셀로
        # 같이 줄어 찾힌다(2026-09-19: 흐린 사진 3장 모두 0.5·0.33·0.25배에서 찾음, 중심은
        # 배율끼리 몇 픽셀 안에서 일치). 찾으면 모서리를 원래 크기로 되돌린다.
        mid, pts = None, None
        for s in (1.0, 0.5, 0.33, 0.25):
            img = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            mid, pts = find_known_marker(self.det, img, self.sizes)
            if pts is not None:
                pts = pts / s
                break
        if pts is None:
            log("  팬틸트 화면에서 마커를 못 찾아 벗어난 각도는 못 쟀습니다")
            return None
        h, w = frame.shape[:2]
        e = pts.mean(axis=0) - np.array([w / 2.0, h / 2.0])
        side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
        # 각도로 바꿀 때는 리얼센스가 잰 마커 거리를 쓴다. 가정 거리(3 m)를 쓰면 2 m 마커가
        # 1.5배 작게 계산된다(2026-09-19 "초점만"에서 +2.7°가 +1.8°로 나온 버그).
        p_rs = self.rs.fresh_point()
        if p_rs is not None:
            dist = float(np.linalg.norm(p_rs))
        # 마커의 실제 크기와 거리로 이 화면의 1픽셀이 몇 도인지 잡는다
        deg_per_px = math.degrees(2 * math.atan(self.sizes[mid] / 2 / dist)) / side
        return e * deg_per_px, e

    def finish(self, rec: dict, dist: float, count_check: bool = True) -> None:
        frame = self.stats.snapshot()
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ROOT, "out", "aim", f"aim_{ts}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rec["image"] = os.path.relpath(path, ROOT)
        me = self.marker_error(frame, dist)
        if me is not None:
            deg, px = me
            rec["marker_err_deg"] = [round(float(v), 2) for v in deg]
            if count_check:   # 조준했을 때만 센다. 같은 자리를 다시 잰 건 영점 보정에 안 넣는다
                self.checks.append((float(deg[0]), float(deg[1])))
            log(f"  마커가 팬틸트 가운데서 가로 {deg[0]:+.1f}°, 세로 {deg[1]:+.1f}° 벗어남 "
                f"({px[0]:+.0f}, {px[1]:+.0f} px)")
        # 글자가 읽히는지 보려고 가운데를 1:1 픽셀로 잘라 보여준다
        h, w = frame.shape[:2]
        cw, ch = min(960, w), min(540, h)
        crop = frame[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
        cc.put("last_jpg", cc.encode(crop, cw))
        cc.put("full_jpg", cc.encode(frame, 1280))   # OCR·VLM 이 볼 원본 전체(가로 1280 으로만 줄임)
        with open(os.path.join(ROOT, "out", "aim", "aim_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=to_plain) + "\n")
        self.job += 1

    def aim_job(self, u: int, v: int, known_dist=None, do_focus: bool = True, strict: bool = False,
                dist_src: str = "마커 거리") -> None:
        try:
            t0 = time.time()
            z = known_dist if known_dist else self.depth_at(u, v)
            dist = z if z else self.args.assume_m
            if known_dist:
                src = f"{dist_src} {z:.2f} m"
            else:
                src = f"깊이 {z:.2f} m" if z else f"거리 {dist:.1f} m 가정"
            try:
                pan, tilt = self.aim.aim(u, v, dist)
            except ValueError as e:
                log(f"조준 불가: {e}")
                return
            pan_c = float(np.clip(pan, *PAN_LIM))
            tilt_c = float(np.clip(tilt, *TILT_LIM))
            pan, tilt = float(pan), float(tilt)
            clipped = bool(abs(pan_c - pan) > 0.05 or abs(tilt_c - tilt) > 0.05)
            if clipped and strict:               # 자동 조준: 끝까지만 돌려 엉뚱한 곳을 찍느니 안 한다
                log(f"조준 안 함: 리얼센스 ({u}, {v}), {src} → 팬 {pan:.1f}° 틸트 {tilt:.1f}° (가동 범위 밖)")
                return
            log(f"조준 #{self.job + 1}: 리얼센스 ({u}, {v}), {src} → 팬 {pan:.1f}° 틸트 {tilt:.1f}°"
                + ("  ※가동 범위 밖이라 끝까지만 감" if clipped else ""))
            self.status = f"조준 #{self.job + 1} 서보 이동 중"
            self.move_to(pan_c, tilt_c)
            t_aim = time.time() - t0
            if do_focus:
                self.status = f"조준 #{self.job + 1} 초점 맞추는 중"
                foc = self.focuser.autofocus()
            else:
                time.sleep(0.5)
                s = self.stats.measure()
                foc = {"skipped": True, "final": s, "sec": 0.0}
                log(f"  초점은 건드리지 않음 (선명도 {s:.0f})")
            rec = {"time": time.strftime("%Y%m%d_%H%M%S"), "u": u, "v": v,
                   "depth_m": z, "dist_used_m": dist, "pan": pan, "tilt": tilt,
                   "clipped": clipped, "aim_sec": round(t_aim, 2),
                   "focus": {k: (round(v2, 2) if isinstance(v2, float) else v2) for k, v2 in foc.items()},
                   "total_sec": round(time.time() - t0, 2)}
            self.finish(rec, dist)
            log(f"  끝: 서보 {t_aim:.1f}초 + 초점 {foc['sec']:.1f}초 = {rec['total_sec']:.1f}초, "
                f"저장 {rec['image']}")
            self.status = f"조준 #{self.job} 끝 ({rec['total_sec']:.1f}초). 다른 곳을 누르세요"
        except Exception as e:  # 작업 스레드가 조용히 죽으면 원인을 알 수 없다
            log(f"조준 작업 오류: {type(e).__name__}: {e}")
            self.focuser.stop()
            self.status = "오류. 로그를 보세요"
        finally:
            self.busy.release()

    def profile_job(self, step_ms: int = 400, first: int = -1, n_first: int = 20, n_back: int = 40,
                    speed: int = None) -> None:
        """초점 모터가 렌즈를 실제로 돌리는지, 어디가 선명한지 재는 진단.

        한 방향으로 n_first 걸음, 반대로 n_back 걸음(전체 이동을 넘게) 돌리며 매 걸음
        선명도와 직전 걸음 대비 화면 변화량을 기록한다. 화면 변화가 없으면 끝단에 닿았거나
        기어가 헛도는 것이다. 결과는 out/aim/focus_profile_*.json.
        """
        f = self.focuser
        keep = f.speed
        if speed:
            f.speed = speed
        try:
            rows = []
            log(f"초점 진단: {'-' if first < 0 else '+'} 방향 {n_first}걸음, 반대 {n_back}걸음, 한 걸음 {step_ms}ms")
            self.status = "초점 진단 중"
            prev = self.stats.small_copy()
            s0 = self.stats.measure()
            rows.append({"i": 0, "dir": 0, "sharp": s0, "moved": 0.0})
            k = 0
            for direction, n in ((first, n_first), (-first, n_back)):
                for _ in range(n):
                    k += 1
                    f.run_for(direction, step_ms)
                    s = self.stats.measure()
                    cur = self.stats.small_copy()
                    moved = float(np.mean(np.abs(cur - prev))) if prev is not None else 0.0
                    prev = cur
                    rows.append({"i": k, "dir": direction, "sharp": round(s, 1), "moved": round(moved, 2)})
                    log(f"    {k:2d} {'+' if direction > 0 else '-'} 선명도 {s:6.1f}  화면변화 {moved:5.1f}")
                    self.status = f"초점 진단 {k}/{n_first + n_back}  선명도 {s:.0f}  화면변화 {moved:.1f}"
            best = max(rows, key=lambda r: r["sharp"])
            still = sum(1 for r in rows[1:] if r["moved"] < 1.0)
            path = os.path.join(ROOT, "out", "aim", time.strftime("focus_profile_%Y%m%d_%H%M%S.json"))
            with open(path, "w", encoding="utf-8") as fp:
                json.dump({"step_ms": step_ms, "speed": f.speed, "first_dir": first, "rows": rows}, fp,
                          ensure_ascii=False, indent=1)
            log(f"  진단 끝(PWM {f.speed}): 최고 선명도 {best['sharp']:.0f} ({best['i']}번째 걸음), "
                f"화면이 안 변한 걸음 {still}/{len(rows) - 1}, 저장 {os.path.relpath(path, ROOT)}")
            self.status = "초점 진단 끝"
        except Exception as e:
            log(f"초점 진단 오류: {type(e).__name__}: {e}")
            self.focuser.stop()
        finally:
            f.speed = keep
            self.busy.release()

    def speed_test_job(self, speeds=(160, 180, 200, 230, 255), ms: int = 800) -> None:
        """속도별로 양쪽을 돌려 화면이 변하는지 본다. 2026-09-19: 160으로 24초를 돌려도
        화면이 안 변해, 힘이 모자라 멈춰 있는지 확인하려고 넣었다."""
        f = self.focuser
        keep = f.speed
        try:
            log(f"속도 시험: {speeds}, 양쪽 {ms}ms씩")
            for sp in speeds:
                f.speed = sp
                res = []
                for direction in (+1, -1):
                    prev = self.stats.small_copy()
                    f.run_for(direction, ms)
                    s = self.stats.measure()
                    cur = self.stats.small_copy()
                    moved = float(np.mean(np.abs(cur - prev))) if prev is not None else 0.0
                    res.append(f"{'+' if direction > 0 else '-'} 화면변화 {moved:4.1f} 선명도 {s:4.0f}")
                log(f"  PWM {sp}: " + " / ".join(res))
                self.status = f"속도 시험 PWM {sp}"
            self.status = "속도 시험 끝"
        except Exception as e:
            log(f"속도 시험 오류: {type(e).__name__}: {e}")
            f.stop()
        finally:
            f.speed = keep
            self.busy.release()

    def marker_v(self):
        """지금 팬틸트 화면에서 마커 중심의 세로 위치(px). 못 찾으면 None."""
        self.stats.measure(2)
        frame = self.stats.snapshot()
        for s in (1.0, 0.5, 0.33, 0.25):
            img = frame if s == 1.0 else cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            _, pts = find_known_marker(self.det, img, self.sizes)
            if pts is not None:
                return (pts / s).mean(axis=0)
        return None

    def tilt_sweep_job(self, a: float, b: float) -> None:
        """틸트를 한 방향으로만 1°씩 옮기며 마커가 몇 픽셀 움직이는지 잰다.

        2026-09-19: 틸트 명령 +2°(147.9→150.0)에 화면이 약 6.6° 움직여 영점 보정이 크게
        넘쳤다. 서보가 명령에 고르게 따라가는지, 한꺼번에 튀는지 본다. 한 방향으로만 가니
        백래시는 섞이지 않는다.
        """
        try:
            log(f"틸트 훑기: {a:.0f} → {b:.0f}, 1°씩")
            self.nano.cmd(f"T{a - 3:.0f}")
            time.sleep(1.0)
            prev = None
            deg = a
            while deg <= b + 1e-6:
                self.nano.cmd(f"T{deg:.0f}")
                time.sleep(0.9)
                c = self.marker_v()
                if c is None:
                    log(f"  틸트 {deg:.0f}: 마커 못 찾음")
                else:
                    step = "" if prev is None else f"  (직전 대비 {c[1] - prev:+.0f}px)"
                    log(f"  틸트 {deg:.0f}: 마커 세로 {c[1]:.0f}px{step}")
                    prev = c[1]
                deg += 1.0
            self.status = "틸트 훑기 끝"
        except Exception as e:
            log(f"틸트 훑기 오류: {type(e).__name__}: {e}")
        finally:
            self.busy.release()

    def focus_job(self) -> None:
        try:
            log(f"초점만 #{self.job + 1}")
            self.status = "초점 맞추는 중"
            foc = self.focuser.autofocus()
            self.finish({"time": time.strftime("%Y%m%d_%H%M%S"), "focus_only": True,
                         "focus": {k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in foc.items()}}, self.args.assume_m, count_check=False)
            self.status = "초점 끝. 다른 곳을 누르세요"
        except Exception as e:
            log(f"초점 작업 오류: {type(e).__name__}: {e}")
            self.focuser.stop()
        finally:
            self.busy.release()

    def apply_trim(self) -> str:
        """모인 검증 결과의 평균만큼 영점을 옮긴다. 원래 파일은 두고 새 파일에 쓴다.

        2026-09-19: 캘리브레이션 뒤 하드웨어를 만지면서 틸트가 약 3° 밀렸다(2 m에서 세로
        +2.6°, +2.9°, 1.6 m에서 약 +3.3°). 샘플을 다시 모으지 않고 영점만 고친다.
        """
        if len(self.checks) < 2:
            return f"마커 검증 결과가 2개 이상 있어야 합니다 (지금 {len(self.checks)}개)"
        e = np.mean(self.checks, axis=0)
        s = self.cfg["servo"]
        # 마커가 오른쪽(+)이면 오른쪽으로 더 돌아야 하고, 아래(+)면 더 내려봐야 한다.
        d_pan = -e[0] * s["pan_deg_per_left_deg"]
        d_tilt = -e[1] * s["tilt_deg_per_up_deg"]
        new = copy.deepcopy(self.cfg)
        new["servo"]["pan_zero_deg"] = s["pan_zero_deg"] + d_pan
        new["servo"]["tilt_zero_deg"] = s["tilt_zero_deg"] + d_tilt
        new.setdefault("_trim", []).append({
            "time": time.strftime("%Y%m%d_%H%M%S"), "checks": len(self.checks),
            "mean_err_deg": [round(float(v), 2) for v in e],
            "d_pan_cmd": round(float(d_pan), 2), "d_tilt_cmd": round(float(d_tilt), 2),
            "from": os.path.basename(self.cfg_path)})
        stem = re.sub(r"_trim\d+$", "", os.path.splitext(os.path.basename(self.cfg_path))[0])
        k = 1
        while os.path.exists(os.path.join(ROOT, "configs", f"{stem}_trim{k}.json")):
            k += 1
        path = os.path.join(ROOT, "configs", f"{stem}_trim{k}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=2, default=to_plain)
        self.cfg, self.cfg_path = new, path
        self.aim = Aim(new, fill_missing=True)
        HOME[:] = [new["servo"]["pan_zero_deg"], new["servo"]["tilt_zero_deg"]]
        sync_home(self.nano)
        n = len(self.checks)
        self.checks = []
        msg = (f"영점 보정: 검증 {n}개 평균 가로 {e[0]:+.1f}° 세로 {e[1]:+.1f}° → "
               f"팬 {d_pan:+.1f}, 틸트 {d_tilt:+.1f} (명령각). 저장 {os.path.relpath(path, ROOT)}")
        log(msg)
        return msg

    def marker_pixel(self):
        """리얼센스가 지금 보는 마커의 픽셀 위치와 거리(광축 방향 z). 못 보면 None."""
        p = self.rs.fresh_point()
        if p is None:
            return None
        x, y, z = (float(c) for c in p)
        fx, fy, cx, cy = self.aim.fx, self.aim.fy, self.aim.cx, self.aim.cy
        return int(round(fx * x / z + cx)), int(round(fy * y / z + cy)), z

    def request(self, kind: str, u: int = 0, v: int = 0, do_focus: bool = True,
                z: float = None, strict: bool = False) -> str:
        src = "마커 거리"
        if kind == "aim_marker":
            mp = self.marker_pixel()
            if mp is None:
                return "리얼센스가 마커를 못 보고 있습니다"
            u, v, known = mp
        else:
            known, src = z, "받은 거리"             # 화면 밖 픽셀은 깊이를 못 읽으므로 거리를 같이 받는다
        if not self.busy.acquire(blocking=False):
            return "이미 작업 중입니다"
        target = {"aim": lambda: self.aim_job(u, v, known, do_focus, strict, src),
                  "aim_marker": lambda: self.aim_job(u, v, known, do_focus),
                  "focus": self.focus_job, "profile": self.profile_job,
                  "speed_test": self.speed_test_job}[kind]
        threading.Thread(target=target, daemon=True).start()
        return "시작"


AIMER = {"a": None}

PAGE = """<!doctype html><meta charset="utf-8"><title>pantilt aim</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:12px}
.row{display:flex;gap:8px}.wrap{position:relative;width:49%}.wrap img{width:100%;cursor:crosshair}
img{border:1px solid #444}#pt{width:49%}#mark{position:absolute;width:26px;height:26px;margin:-13px 0 0 -13px;
border:3px solid #ff0;border-radius:50%;display:none;pointer-events:none}
#s{font-size:18px;padding:6px 0}button{font-size:16px;margin-right:8px}
#log{font-size:13px;background:#000;padding:6px;white-space:pre-wrap}</style>
<h3>RealSense (누르면 그 자리로 조준하고 초점을 맞춥니다) / pan-tilt</h3>
<div class=row><div class=wrap><img id=rs src="/rs.mjpg"><div id=mark></div></div><img id=pt src="/pt.mjpg"></div>
<div id=s>...</div>
<label style="margin-right:12px"><input type=checkbox id=af> 누를 때 초점도 맞추기(초점 모터 고장 시 꺼 두기)</label>
<button onclick="fetch('/aim_marker?focus=0')">마커 조준(검증)</button><button onclick="fetch('/trim')">영점 보정(검증 평균)</button>
<button onclick="fetch('/home')">정면으로</button><button onclick="fetch('/focus')">초점만 다시</button>
<button onclick="fetch('/pin_test?pin=3')">D3 1초</button><button onclick="fetch('/pin_test?pin=5')">D5 1초</button>
<h3>OCR·VLM 이 볼 실제 사진 (원본 전체, 자르지 않음)</h3><img id=full src="/full.jpg" style="width:100%;max-width:1280px">
<h3>가운데만 1:1 픽셀로 확대 (글자 확인용)</h3><img id=last src="/last.jpg">
<div id=log></div>
<script>
const rs=document.getElementById('rs');
rs.addEventListener('click',ev=>{const r=rs.getBoundingClientRect();
 const u=Math.round((ev.clientX-r.left)/r.width*640), v=Math.round((ev.clientY-r.top)/r.height*480);
 const m=document.getElementById('mark');m.style.left=(ev.clientX-r.left)+'px';m.style.top=(ev.clientY-r.top)+'px';
 m.style.display='block';fetch(`/aim?u=${u}&v=${v}&focus=${document.getElementById('af').checked?1:0}`);});
let job=-1;
setInterval(async()=>{const j=await (await fetch('/state')).json();
 document.getElementById('s').textContent=j.status+'   (마커 검증 '+j.checks+'개 모임)';document.getElementById('log').textContent=j.log.join('\\n');
 if(j.job!==job){job=j.job;document.getElementById('last').src='/last.jpg?'+j.job;
   document.getElementById('full').src='/full.jpg?'+j.job;}},500);
</script>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body: bytes, ctype: str = "text/plain; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        a = AIMER["a"]
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self.reply(PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/state":
            with LOG_LOCK:
                lines = list(LOG)
            return self.reply(json.dumps({"status": a.status, "job": a.job, "log": lines,
                                          "checks": len(a.checks)},
                                         ensure_ascii=False).encode(), "application/json")
        # ---- 실시간 이벤트 실행기(노트북)용. 콘솔이 이미 받는 컬러·깊이를 그대로 내준다(구독이 늘지 않음) ----
        if url.path == "/live.mjpg":            # 원본 컬러(640x480) 전 프레임, 부분마다 촬영 시각
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            last = -1
            try:
                while not cc.get("stop"):
                    rs = a.rs
                    if rs.color is not None and rs.color_seq != last:
                        last, img, st = rs.color_seq, rs.color, rs.color_stamp
                        jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                         + f"X-Stamp: {st:.6f}\r\nX-Seq: {last}\r\n".encode()
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return None
        if url.path == "/live/depth.png":       # 최신 정렬 깊이(mm, 16비트 PNG)
            d, st = a.rs.depth, a.rs.depth_stamp
            if d is None:
                return self.send_error(404)
            body = cv2.imencode(".png", d)[1].tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Stamp", f"{st:.6f}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        if url.path == "/live/odom":
            o = a.rs.odom
            if o is None:
                return self.send_error(404)
            return self.reply(json.dumps(dict(x=o[0], y=o[1], yaw=o[2], stamp=o[3])).encode(), "application/json")
        if url.path == "/aim":
            u = int(float(q.get("u", ["320"])[0]))
            v = int(float(q.get("v", ["240"])[0]))
            z = float(q["z"][0]) if "z" in q else None
            return self.reply(a.request("aim", u, v, q.get("focus", ["1"])[0] != "0", z,
                                        q.get("strict", ["0"])[0] == "1").encode())
        if url.path == "/focus":
            return self.reply(a.request("focus").encode())
        if url.path == "/focus_speed_test":
            return self.reply(a.request("speed_test").encode())
        if url.path == "/focus_profile":
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            kw = dict(step_ms=int(q.get("step", ["400"])[0]), first=int(q.get("dir", ["-1"])[0]),
                      n_first=int(q.get("n1", ["20"])[0]), n_back=int(q.get("n2", ["40"])[0]),
                      speed=int(q.get("speed", ["0"])[0]) or None)
            threading.Thread(target=a.profile_job, kwargs=kw, daemon=True).start()
            return self.reply("시작".encode())
        if url.path == "/aim_marker":
            return self.reply(a.request("aim_marker", do_focus=q.get("focus", ["1"])[0] != "0").encode())
        if url.path == "/home":
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            try:
                a.nano.cmd(f"P{HOME[0]:.1f}")
                a.nano.cmd(f"T{HOME[1]:.1f}")
            finally:
                a.busy.release()
            log(f"정면으로 (팬 {HOME[0]:.0f}, 틸트 {HOME[1]:.0f})")
            return self.reply("정면으로".encode())
        if url.path == "/trim":
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            try:
                return self.reply(a.apply_trim().encode())
            finally:
                a.busy.release()
        if url.path == "/goto":
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            try:
                pn = float(np.clip(float(q.get("pan", [HOME[0]])[0]), *PAN_LIM))
                tl = float(np.clip(float(q.get("tilt", [HOME[1]])[0]), *TILT_LIM))
                a.nano.cmd(f"P{pn:.1f}")
                a.nano.cmd(f"T{tl:.1f}")
                time.sleep(1.0)
            finally:
                a.busy.release()
            log(f"이동: 팬 {pn:.0f}, 틸트 {tl:.0f}")
            return self.reply("이동".encode())
        if url.path == "/tilt_sweep":
            ta = float(q.get("a", ["145"])[0])
            tb = float(q.get("b", ["153"])[0])
            if not (TILT_LIM[0] + 3 <= ta <= tb <= TILT_LIM[1]):
                return self.reply("범위가 틀렸습니다".encode())
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            threading.Thread(target=a.tilt_sweep_job, args=(ta, tb), daemon=True).start()
            return self.reply("시작".encode())
        if url.path == "/pin_test":
            pin = int(q.get("pin", ["0"])[0])
            ms = min(1500, int(q.get("ms", ["1000"])[0]))
            if pin not in (3, 5):
                return self.reply("초점 입력 핀 3, 5만 됩니다".encode())
            if not a.busy.acquire(blocking=False):
                return self.reply("이미 작업 중입니다".encode())
            try:
                log(f"D{pin} {ms}ms HIGH (초점 배선 확인)")
                a.nano.cmd(f"X{pin},{ms}")
                time.sleep(ms / 1000.0 + 0.2)
                a.nano.cmd("F0")
            finally:
                a.busy.release()
            return self.reply(f"D{pin} 시험 끝".encode())
        if url.path in ("/last.jpg", "/full.jpg"):
            jpg = cc.get("last_jpg" if url.path == "/last.jpg" else "full_jpg")
            if jpg is None:
                return self.send_error(404)
            return self.reply(jpg, "image/jpeg")
        key = {"/rs.mjpg": "rs_jpg", "/pt.mjpg": "pt_jpg"}.get(url.path)
        if key is None:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            last = None
            while not cc.get("stop"):
                jpg = cc.get(key)
                if jpg and jpg is not last:
                    last = jpg
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                time.sleep(cc.SEND_INTERVAL)
        except (BrokenPipeError, ConnectionResetError):
            pass


def newest_fitted_config() -> str:
    found = sorted(glob.glob(os.path.join(ROOT, "configs", "pantilt_calib_fitted_*.json")))
    if not found:
        sys.exit("캘리브레이션 결과(configs/pantilt_calib_fitted_*.json)가 없습니다.")
    return found[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="캘리브레이션 결과. 생략하면 가장 최근 pantilt_calib_fitted_*.json")
    ap.add_argument("--assume-m", type=float, default=3.0, help="목표까지 가정할 거리")
    ap.add_argument("--odom-topic", default="",
                    help="실시간 이벤트 실행기에 내줄 로봇 오돔(nav_msgs/Odometry) 토픽. 비우면 구독 안 함")
    ap.add_argument("--use-depth", action="store_true",
                    help="리얼센스 깊이로 거리를 잰다. 기본은 안 쓴다(2026-09-19 사용자 결정)")
    ap.add_argument("--roi", type=float, default=0.3, help="초점을 재는 가운데 영역(화면 비율)")
    ap.add_argument("--focus-speed", type=int, default=160, help="2026-09-16 실측 잘 무는 대역 140~180")
    ap.add_argument("--lash-deg", type=float, default=5.0, help="조준 전 두 축을 이만큼 물렀다 다가간다")
    # 자동초점(16일 단계식). 기본값은 2026-09-19 실물에서 162 → 389(최고의 98%)가 나온 설정.
    ap.add_argument("--af-speed", type=int, default=180, help="훑기 속도(PWM)")
    ap.add_argument("--af-coarse-ms", type=int, default=300, help="1단계 한 걸음(ms)")
    ap.add_argument("--af-fine-speed", type=int, default=160, help="정점에 서기 속도. 150 아래는 안 움직였다")
    ap.add_argument("--af-fine-ms", type=int, default=50, help="정점에 서기 한 걸음(ms)")
    ap.add_argument("--af-roi", type=float, default=0.5, help="선명도를 재는 가운데 영역(화면 비율)")
    ap.add_argument("--pan-lim", nargs=2, type=float, default=PAN_LIM, metavar=("LO", "HI"))
    ap.add_argument("--tilt-lim", nargs=2, type=float, default=TILT_LIM, metavar=("LO", "HI"))
    ap.add_argument("--dict", default="4x4_50")
    ap.add_argument("--sizes", default="0:0.18", help="검증용 마커. 18 cm id 0 한 장만 쓴다")
    ap.add_argument("--poly-accuracy", type=float, default=0.08)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--view-width", type=int, default=480)
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args()
    args.depth = args.use_depth
    PAN_LIM[:] = args.pan_lim
    TILT_LIM[:] = args.tilt_lim

    cfg_path = args.config or newest_fitted_config()
    cfg = json.loads(open(cfg_path, encoding="utf-8").read())
    aim = Aim(cfg, fill_missing=True)
    HOME[:] = [cfg["servo"]["pan_zero_deg"], cfg["servo"]["tilt_zero_deg"]]
    fit = cfg.get("_fit", {})
    n_samples = fit.get("samples") or sum(fit.get("sessions", {}).values()) or "?"
    log(f"캘리브레이션: {os.path.relpath(cfg_path, ROOT)} (샘플 {n_samples}개), 정면 팬 {HOME[0]:.1f} 틸트 {HOME[1]:.1f}")

    cc.put("last_jpg", None)
    cc.put("full_jpg", None)
    rs = cc.RealSenseWatcher(args)
    rs.start()
    nano = cc.Nano(find_nano_port())
    # 나노는 포트를 열 때 리셋되어 펌웨어 기본자세(팬 110, 틸트 160)로 간다. 캘리브의 정면으로 옮긴다.
    nano.cmd(f"P{HOME[0]:.1f}")
    nano.cmd(f"T{HOME[1]:.1f}")
    sync_home(nano)

    cap = cv2.VideoCapture(find_camera("Arducam"), cv2.CAP_V4L2)
    if not cap.isOpened():
        cc.put("stop", True)
        sys.exit("팬틸트 카메라를 열지 못했습니다.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    stats = Stats(args.roi)
    aimer = Aimer(args, nano, rs, stats, aim, cfg, cfg_path)
    AIMER["a"] = aimer
    server = cc.Server(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(f"브라우저에서 http://<Go2 주소>:{args.port} 를 여세요")

    last_send = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            stats.update(frame)
            now = time.time()
            if now - last_send < cc.SEND_INTERVAL:
                continue
            last_send = now
            view = frame.copy()
            # 초점은 가운데 args.roi 영역에서 재지만 화면에는 그리지 않는다(2026-09-19 사용자).
            h, w = view.shape[:2]
            cv2.drawMarker(view, (w // 2, h // 2), (0, 160, 255), cv2.MARKER_CROSS, 40, 3)
            _, s = stats.get()
            # 영상 위 글씨는 영문만 된다(OpenCV). 한글 상태는 페이지 아래에 나온다.
            for i, text in enumerate((f"focus {s:.0f}", "BUSY" if aimer.busy.locked() else "")):
                if text:
                    cv2.putText(view, text, (20, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 0, 0), 6, cv2.LINE_AA)
                    cv2.putText(view, text, (20, 50 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 230, 0), 3, cv2.LINE_AA)
            cc.put("pt_jpg", cc.encode(view, args.view_width))
    except KeyboardInterrupt:
        pass
    finally:
        cc.put("stop", True)
        try:
            nano.cmd("F0")
        except Exception:
            pass
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
