#!/usr/bin/env python3
"""캘리브레이션 콘솔. Go2에서 실행하고 노트북 브라우저로 본다.

Go2에는 모니터가 없어서 화면을 볼 수 없다. 그래서 두 카메라 화면을 웹으로 내보낸다.
브라우저에서 http://<Go2 주소>:8088 을 열면 RealSense와 팬틸트 화면이 나란히 보이고,
마커가 잡히면 박스와 거리, 서보 각도, 모은 샘플 수가 함께 표시된다.

기본은 보기 전용이다. --track 을 주면 마커를 따라 팬틸트가 움직이고, 화면 중앙에
맞을 때마다 샘플을 자동으로 저장한다. 마커를 들고 이리저리 옮기면 그게 곧 데이터 수집이다.

안전
    - 한 걸음 최대 --max-step-deg (기본 4°). 가동 범위는 펌웨어가 한 번 더 막는다.
    - 마커를 놓치면 그 자리에 멈춘다. 찾아다니지 않는다.
    - Ctrl+C 로 끝내면 서보는 그 자리에 그대로 둔다.

사용 예 (Go2, ROS 환경을 불러온 셸):
    python3 scripts/calib_console.py                 # 보기만
    python3 scripts/calib_console.py --track         # 따라가며 샘플 저장
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pantilt_devices import find_camera, find_nano_port  # noqa: E402
from aruco_realsense_probe import (DICTS, find_known_marker, image_to_bgr,  # noqa: E402
                                   make_detector, marker_object_points, parse_sizes)

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    sys.exit("pyserial이 없습니다.")

NANO_REF = {"nano": None}     # 웹에서 서보를 조금 움직일 수 있게 공유한다

STATE_LOCK = threading.Lock()
SHARED = {
    "rs_jpg": None, "pt_jpg": None,
    "rs_text": "RealSense 영상 기다리는 중", "pt_text": "팬틸트 영상 기다리는 중",
    "status": "준비 중", "samples": 0, "stop": False,
}


def put(key, value):
    with STATE_LOCK:
        SHARED[key] = value


def get(key):
    with STATE_LOCK:
        return SHARED[key]


def draw(frame, marker_pts, lines, ok_color=(0, 220, 0)):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (0, 160, 255), cv2.MARKER_CROSS, 28, 2)
    if marker_pts is not None:
        cv2.polylines(out, [marker_pts.astype(np.int32)], True, ok_color, 3)
        c = marker_pts.mean(axis=0).astype(int)
        cv2.line(out, (w // 2, h // 2), tuple(c), ok_color, 1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    ok_color if marker_pts is not None else (60, 60, 255), 2, cv2.LINE_AA)
    return out


JPEG_QUALITY = 55     # 와이파이로 두 화면을 보내야 해서 화질을 낮춘다


def encode(frame, width):
    if frame.shape[1] != width:
        frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


# ---------------------------------------------------------------- 검출



# ---------------------------------------------------------------- RealSense 스레드

class RealSenseWatcher(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        # 검출기는 스레드마다 따로 만든다. 하나를 같이 쓰면 검출이 들쭉날쭉해진다.
        self.args = args
        self.detector = make_detector(args.dict, args.poly_accuracy)
        self.sizes = parse_sizes(args.sizes)
        self.K = None
        self.latest = None       # (p_opt, 잰 시각)
        self.images = 0          # 받은 RealSense 영상 수. 0이면 통신 문제다
        self.error = None
        # 컬러에 맞춰 정렬된 깊이(mm). args.depth 가 참일 때만 받는다(조준 콘솔용).
        self.depth = None
        self.depth_t = 0.0
        # 실시간 이벤트 실행기(노트북)에 내줄 원본 컬러·깊이·오돔과 촬영 시각(header stamp, 초)
        self.color, self.color_stamp, self.color_seq = None, 0.0, 0
        self.depth_stamp = 0.0
        self.odom = None                          # (x, y, yaw, stamp)

    def run(self):
        try:
            self._run()
        except Exception as e:  # 스레드가 조용히 죽으면 원인을 알 수 없다
            self.error = f"{type(e).__name__}: {e}"
            put("status", "RealSense 수신 스레드 오류: " + self.error)

    def _run(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        rclpy.init()
        node = Node("pantilt_calib_console")

        def on_info(msg):
            if self.K is None:
                self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        def on_image(msg):
            self.images += 1
            img = image_to_bgr(msg)
            self.color, self.color_stamp = img, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.color_seq += 1
            mid, pts = find_known_marker(self.detector, img, self.sizes)
            lines = []
            if pts is not None and self.K is not None:
                obj = marker_object_points(self.sizes[mid])
                ok, rvec, tvec = cv2.solvePnP(obj, pts, self.K, np.zeros(5),
                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    t = tvec.flatten()
                    self.latest = (t, time.time())
                    side = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
                    lines = [f"marker id {mid} ({self.sizes[mid] * 100:.1f}cm)  "
                             f"{np.linalg.norm(t):.3f} m  {side:.0f}px",
                             f"forward {t[2]:+.2f}  left {-t[0]:+.2f}  up {-t[1]:+.2f} m"]
            else:
                lines = ["marker not found"]
            put("rs_jpg", encode(draw(img, pts, lines), self.args.view_width))
            put("rs_text", " / ".join(lines))

        node.create_subscription(CameraInfo, "/camera/color/camera_info", on_info, qos_profile_sensor_data)
        node.create_subscription(Image, "/camera/color/image_raw", on_image, qos_profile_sensor_data)
        if getattr(self.args, "depth", False):
            def on_depth(msg):
                if msg.encoding not in ("16UC1", "mono16"):
                    return
                d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
                self.depth = d[:, :msg.width].copy()
                self.depth_t = time.time()
                self.depth_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            node.create_subscription(Image, "/camera/aligned_depth_to_color/image_raw", on_depth,
                                     qos_profile_sensor_data)
        odom_topic = getattr(self.args, "odom_topic", "")
        if odom_topic:
            from nav_msgs.msg import Odometry

            def on_odom(msg):
                q = msg.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                self.odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw,
                             msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            node.create_subscription(Odometry, odom_topic, on_odom, qos_profile_sensor_data)
        while not get("stop"):
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()
        rclpy.shutdown()

    def fresh_point(self, max_age=0.6):
        if self.latest is None:
            return None
        p, t = self.latest
        return p if time.time() - t < max_age else None


# ---------------------------------------------------------------- 겨누기

def rs_angles(p_opt):
    """리얼센스 광학좌표 점의 좌우·위아래 각도(도). 오른쪽·아래가 +."""
    x, y, z = (float(v) for v in p_opt)
    return math.degrees(math.atan2(x, z)), math.degrees(math.atan2(y, z))


class Guide:
    """리얼센스가 본 방향으로 팬틸트를 바로 겨눈다. 눈 감고 훑지 않는다.

    2026-09-19: 전에는 마커를 놓치면 정면 부근 작은 상자(팬 96~108, 틸트 152~166)를
    위아래로만 훑었다. 사용자 지적대로 리얼센스가 마커를 보고 있으니 그리로 바로 가면 된다.
    모은 샘플로 '리얼센스 각도 → 서보 명령각'을 직선으로 맞추고, 샘플이 늘 때마다 다시
    맞춘다. 두 카메라가 떨어져 있어 거리마다 조금 다르지만(시차) 찾기에는 팬틸트 화면
    (가로 약 27°, 세로 약 15°) 안에만 들어오면 된다. 정밀한 관계는 pantilt_calib_fit.py가 푼다.
    """
    # 샘플이 모자랄 때 쓰는 기울기. 팬: 리얼센스 오른쪽 1° → 팬 -1°(각도가 크면 왼쪽).
    # 틸트: 아래 1° → 틸트 +1/1.5°(위 서보가 명령 1°에 약 1.5° 돈다, 2026-09-19 실측).
    DEFAULT = {"pan": (100.0, -1.0), "tilt": (160.0, 1.0 / 1.5)}
    # 겨눴는데 안 보이면 그 주변을 이 순서로 둘러본다(팬°, 틸트°).
    AROUND = [(0, 0), (5, 0), (-5, 0), (0, 4), (0, -4), (5, 4), (-5, 4), (5, -4), (-5, -4)]

    def __init__(self):
        self.pairs = []          # (좌우각, 위아래각, 팬, 틸트)
        self.model = dict(self.DEFAULT)

    def add(self, p_opt, pan, tilt):
        ax, ay = rs_angles(p_opt)
        self.pairs.append((ax, ay, float(pan), float(tilt)))
        a = np.array(self.pairs)
        for key, xi, yi in (("pan", 0, 2), ("tilt", 1, 3)):
            x, y = a[:, xi], a[:, yi]
            slope0 = self.DEFAULT[key][1]
            if len(a) >= 3 and np.ptp(x) > 4.0:
                slope, icpt = np.polyfit(x, y, 1)
                # 부호가 반대이거나 기본값과 3배 넘게 다르면 믿지 않는다
                if slope * slope0 > 0 and 0.33 < slope / slope0 < 3.0:
                    self.model[key] = (float(icpt), float(slope))
                    continue
            # 퍼짐이 모자라면 기울기는 기본값, 절편만 샘플 평균으로
            self.model[key] = (float(np.mean(y - slope0 * x)), slope0)

    def target(self, p_opt, k=0):
        ax, ay = rs_angles(p_opt)
        dp, dt = self.AROUND[k % len(self.AROUND)]
        pan = self.model["pan"][0] + self.model["pan"][1] * ax + dp
        tilt = self.model["tilt"][0] + self.model["tilt"][1] * ay + dt
        return float(np.clip(pan, 20, 180)), float(np.clip(tilt, 100, 180))   # 2026-09-19 실측 범위


# ---------------------------------------------------------------- 나노

class Nano:
    def __init__(self, port, tries=4, reopen=4):
        self.lock = threading.Lock()
        # 2026-09-19 Go2 실측: 포트를 열 때마다 한 번씩 번갈아 나노가 한 바이트도 안
        # 돌려준다(여닫기 8번에 OXOXOXOX, 카메라와 무관). 한번 먹통으로 열린 연결은
        # 기다려도, DTR로 리셋해도 안 살아난다. 닫고 다시 열면 된다. 원리는 아직 모른다.
        for attempt in range(reopen):
            self.ser = serial.Serial(port, 115200, timeout=0.4)
            # 포트를 열 때 나노가 리셋됐을 수 있다(hupcl). 부팅이 끝날 때까지 쉰다.
            time.sleep(2.0)
            for i in range(tries):
                self.ser.reset_input_buffer()
                if self.cmd("?"):
                    if attempt or i:
                        print(f"나노 응답: 포트 {attempt + 1}번째 열기, {i + 1}번째 물음")
                    return
                time.sleep(0.8)
            self.ser.close()
            time.sleep(0.5)
        raise RuntimeError("나노가 응답하지 않습니다 (포트를 여러 번 다시 열어도)")

    def cmd(self, line):
        with self.lock:
            self.ser.reset_input_buffer()
            self.ser.write((line + "\n").encode())
            self.ser.flush()
            for _ in range(5):
                reply = self.ser.readline().decode(errors="replace").strip()
                m = re.search(r"pan=(\S+) tilt=([\d.]+) tgt=([\d.]+),([\d.]+)", reply)
                if m:
                    return {"attached": m.group(1) != "idle",
                            "pan": float(m.group(3)), "tilt": float(m.group(4))}
            return {}

    def close(self):
        self.ser.close()


# ---------------------------------------------------------------- 웹

SEND_INTERVAL = 0.12   # 초당 8장 정도면 눈으로 보기 충분하다

PAGE = b"""<!doctype html><meta charset="utf-8"><title>pantilt calibration</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:12px}
img{width:49%;border:1px solid #444}h3{margin:6px 0 2px}#s{font-size:18px;padding:6px 0}</style>
<h3>RealSense (left) / pan-tilt camera (right)</h3>
<img src="/rs.mjpg"><img src="/pt.mjpg">
<div id="s">...</div>
<script>setInterval(async()=>{document.getElementById('s').textContent=await (await fetch('/status')).text()},500)</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)
            return
        if self.path.startswith("/goto") or self.path.startswith("/move"):
            # /goto?pan=102&tilt=160  또는  /move?dpan=0&dtilt=-2
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            nano = NANO_REF["nano"]
            if nano is None:
                body = "따라가기 모드가 아니라 서보를 움직일 수 없습니다".encode()
            else:
                if self.path.startswith("/goto"):
                    for key, letter in (("pan", "P"), ("tilt", "T")):
                        if key in q:
                            nano.cmd(f"{letter}{float(q[key][0]):.1f}")
                else:
                    dp = float(q.get("dpan", [0])[0])
                    dt = float(q.get("dtilt", [0])[0])
                    nano.cmd(f"R{dp:.2f},{dt:.2f}")
                time.sleep(0.6)
                body = str(nano.cmd("?")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/status":
            body = f"{get('status')}   저장된 샘플 {get('samples')}개".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        key = {"/rs.mjpg": "rs_jpg", "/pt.mjpg": "pt_jpg"}.get(self.path)
        if key is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            last = None
            while not get("stop"):
                jpg = get(key)
                # 같은 화면을 다시 보내지 않는다. 와이파이 전송량이 절반 이하로 준다.
                if jpg and jpg is not last:
                    last = jpg
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                time.sleep(SEND_INTERVAL)
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------- 본체

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="4x4_50", choices=sorted(DICTS))
    # 2026-09-19 사용자 결정: 캘리브레이션은 18 cm(검은 사각형 한 변) id 0 한 장만 쓴다.
    # 4.1 cm 기본값이 남아 있어 2.78 m를 0.63 m로 잘못 계산한 적이 있다.
    ap.add_argument("--sizes", default="0:0.18",
                    help="번호별 검은 사각형 한 변(m). 두 마커를 번갈아 써도 알아서 구분한다")
    ap.add_argument("--poly-accuracy", type=float, default=0.08)
    ap.add_argument("--width", type=int, default=1920, help="팬틸트 카메라 촬영 해상도")
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--view-width", type=int, default=480, help="브라우저로 보낼 폭. 작을수록 안 끊긴다")
    ap.add_argument("--view-fps", type=float, default=8.0, help="브라우저로 보낼 초당 장수")
    ap.add_argument("--jpeg-quality", type=int, default=55)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--track", action="store_true", help="마커를 따라가며 샘플 저장")
    ap.add_argument("--search", action="store_true", default=True,
                    help="마커를 놓치면 천천히 훑어서 찾는다")
    ap.add_argument("--no-search", dest="search", action="store_false")
    ap.add_argument("--max-step-deg", type=float, default=2.5)
    # 2026-09-19 실측: 서보 불감대가 약 1°(팬 1° 명령에 0px)라 12px(0.14°) 안으로는
    # 영원히 못 맞추고 흔들렸다. 1° 안쪽이면 맞춘 것으로 보고, 남은 오차는 저장할 때
    # 화면 픽셀로 보정한다(샘플의 pan/tilt = 명령각 + J⁻¹·오차).
    ap.add_argument("--tol-px", type=float, default=100.0, help="이 안에 들어오면 맞춘 것으로 본다")
    # 2026-09-19: 1.5°로는 유격이 다 안 먹혀 같은 명령에 출발점 따라 4°씩 다른 곳에 섰다.
    # 조준 콘솔과 같은 5°로 모아야 캘리브레이션과 실제 조준이 같은 조건이 된다.
    ap.add_argument("--lash-deg", type=float, default=5.0,
                    help="저장 직전 두 축을 이만큼 물렀다 다시 다가가 기어가 늘 같은 쪽으로 물리게 한다")
    ap.add_argument("--probe-deg", type=float, default=6.0,
                    help="비율 잴 때 움직일 각도. 기어 유격보다 커야 한다")
    ap.add_argument("--min-sep-m", type=float, default=0.12, help="새 샘플로 인정할 최소 이동 거리")
    ap.add_argument("--out", default="out/calib_samples.jsonl")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes)
    global SEND_INTERVAL, JPEG_QUALITY
    SEND_INTERVAL = 1.0 / max(1.0, args.view_fps)
    JPEG_QUALITY = args.jpeg_quality
    detector = make_detector(args.dict, args.poly_accuracy)
    rs = RealSenseWatcher(args)
    rs.start()

    cap = cv2.VideoCapture(find_camera(), cv2.CAP_V4L2)
    if not cap.isOpened():
        put("stop", True)
        sys.exit("팬틸트 카메라를 열지 못했습니다.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    nano = None
    if args.track:
        try:
            nano = Nano(find_nano_port())
        except (RuntimeError, serial.SerialException) as e:
            put("stop", True)
            sys.exit(f"{e}. 포트 권한(sudo chmod 666 /dev/ttyUSB0)과 USB 연결을 확인하세요.")
        NANO_REF["nano"] = nano
        st = nano.cmd("?")
        if not st["attached"]:
            nano.cmd("P110")   # 2026-09-19 캘리브레이션(팬 영점 109.7°)과 사용자 눈 확인으로 정면은 110°. 16일의 100°는 눈대중이었다.
            time.sleep(1.0)

    server = Server(("0.0.0.0", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"브라우저에서 http://<Go2 주소>:{args.port} 를 여세요. 끝내려면 Ctrl+C.")
    if not args.track:
        print("지금은 보기 전용입니다. 서보는 움직이지 않습니다.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # 팬·틸트 1도에 화면이 몇 픽셀 움직이는지.
    # 2026-09-18 격자 실측: 팬을 올리면 마커가 화면 오른쪽으로(u 증가, 약 80px/°),
    # 틸트를 올리면 위로(v 감소, 약 74px/°). 1920 기준 값이라 해상도에 맞춰 줄인다.
    #
    # 고정값으로 쓴다. 전에는 움직일 때마다 실제 이동량으로 고쳐 배웠는데(Broyden),
    # 명령 직후 서보가 아직 안 움직인 프레임으로 배워서 "명령해도 안 움직인다"고
    # 학습해 버렸다. 그 뒤 점점 크게 휘두르다 20° 안전장치에 걸렸다(2026-09-19).
    # 백래시(약 2°) 때문에 넘치지 않게 계산한 양의 TRACK_GAIN 만큼만 움직인다.
    # 2026-09-19 jacobian_check.py 직선 기울기 실측: 팬 1° → 가로 +71px, 틸트 1° → 세로 약
    # -90~-130px(들쭉날쭉, 중간값 -107). 틸트가 팬의 약 1.5배라 위 서보는 270°형으로 보인다.
    k = args.width / 1920.0
    J = np.array([[71.0 * k, 0.0], [0.0, -107.0 * k]])
    TRACK_GAIN = 0.7
    lash_done = False           # 저장 직전 백래시 정리를 마쳤는지
    recorded = []               # 저장한 샘플의 RealSense 위치들
    guide = Guide()
    # 재시작해도 이어서 모은다. 이미 저장한 샘플을 불러와 간격 판정과 겨누기에 쓴다.
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                if s.get("ok", True):
                    recorded.append(np.array(s["p_opt"]))
                    guide.add(s["p_opt"], s["pan"], s["tilt"])
        put("samples", len(recorded))
        if recorded:
            print(f"이미 모은 샘플 {len(recorded)}개를 불러왔습니다")
    last_move = 0.0
    settled_since = None
    moved_total = 0.0           # 마지막으로 가까워진 뒤 누적으로 얼마나 명령했는지
    lost_since = None           # 마커를 놓친 시각
    search_idx = 0              # 찾기 순찰 위치
    best_err = None             # 지금까지 가장 작았던 화면 오차
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            mid, pts = find_known_marker(detector, frame, sizes)
            h, w = frame.shape[:2]
            center = np.array([w / 2.0, h / 2.0])
            lines = []
            st = nano.cmd("?") if nano else {}
            if rs.images == 0:
                put("status", "RealSense 영상이 아직 한 장도 안 왔습니다"
                    + (f" ({rs.error})" if rs.error else " — ROS 환경/토픽 확인 필요"))
            if st:
                lines.append(f"pan {st['pan']:.1f}  tilt {st['tilt']:.1f}")
            if pts is None:
                lines.append("marker not found")
                now = time.time()
                if lost_since is None:
                    lost_since = now
                if not args.track or not args.search or now - lost_since < 1.0:
                    put("status", "팬틸트 화면에서 마커를 못 찾는 중")
                elif now - last_move > 1.2:
                    p_rs = rs.fresh_point()
                    if p_rs is None:
                        put("status", "두 카메라 모두 마커를 못 봅니다. 리얼센스 화면 안에 두세요")
                    else:
                        pn, t = guide.target(p_rs, search_idx)
                        nano.cmd(f"P{pn:.1f}")
                        nano.cmd(f"T{t:.1f}")
                        search_idx += 1
                        last_move = now
                        put("status", f"리얼센스가 본 위치로 겨누는 중 (팬 {pn:.0f}, 틸트 {t:.0f}, "
                                      f"샘플 {len(guide.pairs)}개로 추정)")
                settled_since = None
            else:
                e = pts.mean(axis=0) - center
                lines.append(f"error ({e[0]:+.0f}, {e[1]:+.0f}) px")
                p_opt = rs.fresh_point()
                if p_opt is None:
                    put("status", "RealSense 쪽에서 마커가 안 보입니다. 두 카메라에 모두 보이게 두세요")
                    settled_since = None
                elif not args.track:
                    put("status", f"보기 전용. RealSense 거리 {np.linalg.norm(p_opt):.2f} m")
                else:
                    now = time.time()
                    lost_since = None
                    search_idx = 0
                    if False:
                        pass
                    # 백래시 정리 뒤엔 1° 남짓 어긋날 수 있다. 그때 다시 따라가면 정리→어긋남이
                    # 끝없이 되풀이되므로 허용 오차를 2배로 보고 그대로 저장한다(픽셀로 보정됨).
                    elif np.hypot(*e) > args.tol_px * (2.0 if lash_done else 1.0):
                        settled_since = None
                        lash_done = False
                        err_now = float(np.hypot(*e))
                        if best_err is None or err_now < best_err * 0.9:
                            best_err, moved_total = err_now, 0.0
                        if moved_total > 30.0:
                            put("status", "30° 넘게 돌렸는데 오차가 안 줄어듭니다. "
                                          "서보가 헛돌거나 비율이 틀렸습니다. 따라가기를 멈춥니다")
                        elif now - last_move > 0.45:
                            d = TRACK_GAIN * np.linalg.solve(J, -e)
                            d = np.clip(d, -args.max_step_deg, args.max_step_deg)
                            nano.cmd(f"R{d[0]:.2f},{d[1]:.2f}")
                            moved_total += float(np.abs(d).sum())
                            last_move = now
                            put("status", f"따라가는 중 (오차 {np.hypot(*e):.0f}px, "
                                          f"팬{d[0]:+.1f}° 틸트{d[1]:+.1f}°)")
                    else:
                        best_err, moved_total = None, 0.0
                        if settled_since is None:
                            settled_since = now
                        elif now - settled_since > 0.4 and not lash_done:
                            # 두 축 모두 뒤로 물렀다가 다시 다가간다. 그러면 샘플마다 기어가
                            # 같은 쪽으로 물려 있어 백래시(약 2°)가 들쭉날쭉 섞이지 않는다.
                            nano.cmd(f"R{-args.lash_deg:.2f},{-args.lash_deg:.2f}")
                            time.sleep(0.7)
                            nano.cmd(f"R{args.lash_deg:.2f},{args.lash_deg:.2f}")
                            lash_done = True
                            settled_since = None
                            last_move = time.time()
                            put("status", "맞춤. 백래시 정리 후 멈추면 저장합니다")
                        elif now - settled_since > 0.8:
                            far_enough = all(np.linalg.norm(np.array(p) - p_opt) > args.min_sep_m
                                             for p in recorded)
                            if far_enough and st:
                                resid = np.linalg.solve(J, -e)
                                sample = {
                                    "time": time.strftime("%Y%m%d_%H%M%S"),
                                    "marker_id": mid, "marker_size_m": sizes[mid],
                                    "p_opt": [float(v) for v in p_opt],
                                    "rs_dist_m": float(np.linalg.norm(p_opt)),
                                    "pan_cmd": st["pan"], "tilt_cmd": st["tilt"],
                                    "pan": float(st["pan"] + resid[0]),
                                    "tilt": float(st["tilt"] + resid[1]),
                                    "imx_err_px": [float(v) for v in e],
                                    "px_per_deg": J.tolist(), "ok": True,
                                }
                                with open(args.out, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                                recorded.append(p_opt)
                                guide.add(p_opt, sample["pan"], sample["tilt"])
                                put("samples", len(recorded))
                                print(f"샘플 {len(recorded)}: 거리 {sample['rs_dist_m']:.2f} m, "
                                      f"팬 {sample['pan']:.2f}, 틸트 {sample['tilt']:.2f}")
                                settled_since = None
                            else:
                                put("status", "맞췄음. 마커를 다른 위치로 옮기면 다음 샘플을 저장합니다")
            # 손으로 초점을 맞출 수 있게 가운데 영역 선명도를 같이 띄운다.
            gh, gw = frame.shape[0] // 4, frame.shape[1] // 4
            crop = frame[gh:gh * 3, gw:gw * 3]
            sharp = cv2.Laplacian(cv2.cvtColor(cv2.resize(crop, (480, 270)), cv2.COLOR_BGR2GRAY),
                                  cv2.CV_64F).var()
            lines.append(f"focus {sharp:.0f}   RS frames {rs.images}")
            put("pt_jpg", encode(draw(frame, pts, lines), args.view_width))
            time.sleep(0.02)   # 본 루프가 CPU를 독점하면 수신 스레드가 굶는다
    except KeyboardInterrupt:
        print("\n끝냅니다.")
    finally:
        put("stop", True)
        time.sleep(0.2)
        cap.release()
        if nano:
            nano.close()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
