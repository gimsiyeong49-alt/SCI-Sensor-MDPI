#!/usr/bin/env python3
"""RealSense 컬러 영상에서 ArUco 마커를 찾아 거리와 3D 위치를 출력한다.

depth 없이 마커의 실제 크기만으로 거리를 계산한다. 줄자로 잰 거리와 비교해서
이 방법을 캘리브레이션에 써도 되는지 확인하는 용도다.

ROS 토픽을 받아보기만 한다. 구독자가 하나 늘 뿐 RealSense 설정은 바꾸지 않는다.

사용 예 (Go2, 로그인 셸에서):
    cd ~/pantilt_calib
    python3 scripts/aruco_realsense_probe.py
    python3 scripts/aruco_realsense_probe.py --frames 60 --save out/rs_aruco.jpg
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np


# ROS가 없는 컴퓨터에서도 이 파일의 검출 함수만 빌려 쓸 수 있어야 한다. 그래서 여기서 죽지 않는다.
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object

DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "6x6_50": cv2.aruco.DICT_6X6_50,
}


def image_to_bgr(msg) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.width), cv2.COLOR_GRAY2BGR)
    raise ValueError(f"지원하지 않는 인코딩: {msg.encoding}")


class _OldDetector:
    """OpenCV 4.7 이전에는 ArucoDetector 클래스가 없다. 같은 사용법으로 감싼다."""

    def __init__(self, dictionary, params):
        self.dictionary, self.params = dictionary, params

    def detectMarkers(self, gray):
        return cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)


def make_detector(dict_name: str, poly_accuracy: float = 0.08):
    """흐릿하거나 어두운 화면에서도 잡히도록 사각형 허용치를 넓힌 검출기."""
    key = DICTS[dict_name]
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(key)
    else:  # pragma: no cover
        dictionary = cv2.aruco.Dictionary_get(key)
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:  # pragma: no cover
        params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.polygonalApproxAccuracyRate = poly_accuracy
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return _OldDetector(dictionary, params)  # pragma: no cover


def find_marker(detector, bgr, marker_id):
    """화면에서 지정한 번호의 마커 네 모서리를 찾는다. 없으면 None."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    for c, mid in zip(corners, ids.flatten()):
        if int(mid) == marker_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def parse_sizes(text: str) -> dict:
    """"0:0.041,1:0.18" → {0: 0.041, 1: 0.18}. 마커마다 실제 크기가 다르므로 번호별로 둔다."""
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        mid, size = part.split(":")
        out[int(mid)] = float(size)
    return out


def find_known_marker(detector, bgr, sizes: dict):
    """크기를 아는 마커 중 먼저 보이는 것을 돌려준다. (번호, 네 모서리) 또는 (None, None)."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None, None
    for c, mid in zip(corners, ids.flatten()):
        if int(mid) in sizes:
            return int(mid), c.reshape(4, 2).astype(np.float64)
    return None, None


def marker_object_points(size_m: float) -> np.ndarray:
    """ArUco 모서리 순서(왼위, 오위, 오아래, 왼아래)에 맞춘 마커 평면 좌표."""
    h = size_m / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


class Probe(Node):
    def __init__(self, args):
        super().__init__("pantilt_aruco_probe")
        self.args = args
        self.K = None
        self.D = np.zeros(5)
        self.samples = []
        self.images_seen = 0
        self.last_annotated = None
        self.detector = make_detector(args.dict, args.poly_accuracy)
        self.obj = marker_object_points(args.marker_size)
        self.create_subscription(CameraInfo, args.info_topic, self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            if len(msg.d) >= 5:
                self.D = np.array(msg.d[:5], dtype=np.float64)
            self.get_logger().info(
                f"intrinsics fx={self.K[0, 0]:.2f} fy={self.K[1, 1]:.2f} "
                f"cx={self.K[0, 2]:.2f} cy={self.K[1, 2]:.2f}")

    def on_image(self, msg):
        if self.K is None or len(self.samples) >= self.args.frames:
            return
        self.images_seen += 1
        img = image_to_bgr(msg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return
        for c, mid in zip(corners, ids.flatten()):
            if int(mid) != self.args.id:
                continue
            pts = c.reshape(4, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(self.obj, pts, self.K, self.D,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            t = tvec.flatten()
            side_px = float(np.mean([np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]))
            # 마커 면이 카메라를 얼마나 비스듬히 보는지. 0이면 정면.
            R, _ = cv2.Rodrigues(rvec)
            normal = R[:, 2]
            oblique = math.degrees(math.acos(min(1.0, abs(float(normal @ (-t / np.linalg.norm(t)))))))
            self.samples.append({
                "u": float(pts[:, 0].mean()), "v": float(pts[:, 1].mean()),
                "side_px": side_px, "t": t, "dist": float(np.linalg.norm(t)), "oblique": oblique,
            })
            if self.args.save:
                cv2.aruco.drawDetectedMarkers(img, [c], np.array([[mid]]))
                cv2.drawFrameAxes(img, self.K, self.D, rvec, tvec, self.args.marker_size * 0.75)
                cv2.putText(img, f"id {mid}  {np.linalg.norm(t):.3f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                self.last_annotated = img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-topic", default="/camera/color/image_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    # 갖고 있는 4.1 cm 마커는 DICT_4X4_50의 0번이다. RealSense 영상에서 칸을 직접 읽어 확인했다.
    # 사전이 클수록(4x4_1000) 배경 무늬를 마커로 잘못 읽기 쉽다. 실제로 683번 오검출이 있었다.
    ap.add_argument("--dict", default="4x4_50", choices=sorted(DICTS))
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--poly-accuracy", type=float, default=0.08,
                    help="사각형 인정 오차. 기본 0.03이면 흐리거나 어두울 때 모서리가 뭉개져 탈락한다")
    ap.add_argument("--marker-size", type=float, default=0.18, help="검은 사각형 한 변 (m). 18 cm id 0 한 장만 쓴다")
    ap.add_argument("--frames", type=int, default=30, help="마커가 잡힌 프레임을 몇 장 모을지")
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--save", help="마지막 검출 프레임에 표시를 그려 저장할 경로")
    args = ap.parse_args()

    if not ROS_AVAILABLE:
        sys.exit("rclpy가 없습니다. Go2에서 ROS 환경을 불러온 셸로 실행하세요 "
                 "(source ~/unitree_ros2/setup.sh).")
    rclpy.init()
    node = Probe(args)
    t0 = time.time()
    try:
        while rclpy.ok() and len(node.samples) < args.frames and time.time() - t0 < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        samples, seen, K, annotated = node.samples, node.images_seen, node.K, node.last_annotated
        node.destroy_node()
        rclpy.shutdown()

    if K is None:
        print("camera_info를 받지 못했습니다. ROS 환경을 불러온 셸인지 확인하세요.")
        return 1
    if seen == 0:
        print("영상을 한 장도 받지 못했습니다. 토픽 이름이나 통신 설정을 확인하세요.")
        return 1
    if not samples:
        print(f"영상 {seen}장을 받았지만 마커(id {args.id})를 못 찾았습니다. "
              "가까이 두거나 조명을 밝게 해 보세요.")
        return 1

    d = [s["dist"] for s in samples]
    t = np.array([s["t"] for s in samples])
    med = np.median(t, axis=0)
    print(f"마커 {len(samples)}프레임 검출 (받은 영상 {seen}장)")
    print(f"  한 변 {statistics.median(s['side_px'] for s in samples):.1f} px, "
          f"화면 위치 u={statistics.median(s['u'] for s in samples):.0f} "
          f"v={statistics.median(s['v'] for s in samples):.0f}")
    print(f"  거리 {statistics.median(d):.3f} m  (흔들림 표준편차 {statistics.pstdev(d) * 1000:.1f} mm)")
    print(f"  정면 방향 거리(z) {med[2]:.3f} m")
    print(f"  로봇축 위치  앞 {med[2]:+.3f}  왼쪽 {-med[0]:+.3f}  위 {-med[1]:+.3f}  (m, RealSense 렌즈 기준)")
    print(f"  비스듬한 정도 {statistics.median(s['oblique'] for s in samples):.1f}° (0이면 정면)")
    if args.save and annotated is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        cv2.imwrite(args.save, annotated)
        print(f"  저장: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
