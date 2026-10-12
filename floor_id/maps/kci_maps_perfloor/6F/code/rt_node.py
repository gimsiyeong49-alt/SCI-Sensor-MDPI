#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rt_node.py — 실시간 인과 위치추정 ROS2 노드 (Humble).

rt_localizer.RTLocalizer.localize_buf 를 그대로 쓴다. 오프라인 평가와 **같은 코드 경로**라
검증된 성능(held 중앙 0.74m·4/5)이 그대로 적용된다.

상태기계
  BOOTSTRAP : 시동 후 boot_s 초 동안 스캔·odom 만 모아 φ 4-mode 복구 -> 시동창 경쟁으로 1개 확정
  TRACKING  : rate Hz 로 '현재 pose 를 과거 win_s 초' 로 추정, top-k 마커 발행

토픽 (파라미터로 변경 가능)
  구독  ~scan  sensor_msgs/PointCloud2   구조점(a1_3_1 상당). xy(m, 센서프레임)만 사용
        ~odom nav_msgs/Odometry          FAST-LIO pose
  발행  ~pose  geometry_msgs/PoseWithCovarianceStamped   확정 추정
        ~topk  visualization_msgs/MarkerArray            후보 시각화

주의
  - 의미(VLM) 항은 obs 를 넣어줄 때만 동작한다. 카메라 파이프라인이 붙기 전에는 LiDAR 만으로
    동작하며, 그래도 인과 부트스트랩·창 누적은 동일하다.
  - depth_scene 은 지금 오프라인 깊이맵에 의존한다. 노드에서는 스캔 기반 근사를 쓴다(_scene_from_scan).
    분류가 달라지면 PRIOR 가 달라지므로, 배포 전 오프라인 결과와 대조할 것.
"""
import numpy as np, threading, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray, Marker
import rt_localizer as RT
from eval_semloc import peaks


def yaw_of(q):
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))


class RTLocalizerNode(Node):
    def __init__(self):
        super().__init__('rt_floorplan_localizer')
        p = self.declare_parameters('', [
            ('scan_topic', '/lidar/structure_points'),
            ('odom_topic', '/odom'),
            ('boot_s', 15.0), ('win_s', 45.0), ('dt', 3.0),
            ('rate_hz', 0.5), ('lam_sem', 0.5), ('w_nogo', 3.0), ('topk', 40),
        ])
        g = lambda k: self.get_parameter(k).value
        self.boot_s, self.win_s, self.dt = g('boot_s'), g('win_s'), g('dt')
        self.K = int(g('topk'))
        self.get_logger().info('지도/의미지도 로딩 중...')
        self.Z = RT.RTLocalizer(lam_sem=g('lam_sem'), w_nogo=g('w_nogo'), verbose=False)
        self.L = self.Z.L
        self.lock = threading.Lock()
        self.buf = []              # [(t, pts Nx2 m, xy, yaw)]
        self.odom = None           # (t, xy, yaw)
        self.phi = None
        self.t0 = None
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(PointCloud2, g('scan_topic'), self.on_scan, qos)
        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, qos)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, '~/pose', 1)
        self.pub_topk = self.create_publisher(MarkerArray, '~/topk', 1)
        self.create_timer(1.0/max(g('rate_hz'), 1e-3), self.tick)
        self.get_logger().info(f"준비 완료. 시동 {self.boot_s:.0f}s 수집 후 φ 확정 → 창 {self.win_s:.0f}s 추정")

    # ---------- 콜백 ----------
    def on_odom(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        self.odom = (t, np.array([m.pose.pose.position.x, m.pose.pose.position.y]),
                     yaw_of(m.pose.pose.orientation))

    def on_scan(self, m):
        if self.odom is None: return
        t = m.header.stamp.sec + m.header.stamp.nanosec*1e-9
        pts = np.array([[x, y] for x, y, _ in pc2.read_points(m, ('x', 'y', 'z'), True)], float)
        if len(pts) < 20: return
        with self.lock:
            if self.t0 is None: self.t0 = t
            _, xy, yaw = self.odom
            self.buf.append((t, pts, xy, yaw))
            keep = max(self.win_s, self.boot_s) + 5.0
            self.buf = [b for b in self.buf if t - b[0] <= keep]

    # ---------- 장면 분류(스캔 기반 근사) ----------
    def _scene_from_scan(self, pts):
        a = np.arctan2(pts[:, 1], pts[:, 0]); r = np.hypot(pts[:, 0], pts[:, 1])
        def band(lo, hi):
            m = (a >= lo) & (a < hi)
            return float(np.percentile(r[m], 20)) if m.sum() > 5 else 30.0
        fwd = band(-0.35, 0.35); left = band(1.2, 1.9); right = band(-1.9, -1.2)
        wide = max(left, right)
        if fwd < 4.5: return 'dead-end'
        if (fwd > 12 and wide < 7.5) or wide < 4.5: return 'corridor'
        return 'junction'

    def _frames(self, t_now, span):
        """버퍼에서 dt 간격으로 프레임 선택(최신이 마지막)."""
        out = []; want = t_now - np.arange(0, span+1e-6, self.dt)[::-1]
        for tw in want:
            c = min(self.buf, key=lambda b: abs(b[0]-tw), default=None)
            if c is None or abs(c[0]-tw) > self.dt: continue
            if out and out[-1][0] == c[0]: continue
            out.append(c)
        return out

    def _mkbuf(self, frames):
        return [dict(pts=p, xy=xy, yaw=yaw, t=t, scene=self._scene_from_scan(p), obs=None)
                for (t, p, xy, yaw) in frames]

    # ---------- 주기 실행 ----------
    def tick(self):
        with self.lock:
            if not self.buf: return
            t_now = self.buf[-1][0]
            if self.t0 is None or t_now - self.t0 < self.boot_s:
                return
            frames_b = self._frames(self.t0 + self.boot_s, self.boot_s)
            frames_w = self._frames(t_now, self.win_s)
            buf_b = self._mkbuf(frames_b); buf_w = self._mkbuf(frames_w)
        if self.phi is None:
            if len(buf_b) < 3: return
            cloud = self._cloud(frames_b)
            h0 = RT.ori_hist(self.L.PPM*(cloud @ RT._A(0.0).T))   # ★y 뒤집기 필수
            n = lambda v: (v-v.mean())/(np.linalg.norm(v-v.mean())+1e-9)
            C = np.array([np.dot(n(h0), np.roll(n(self.Z.h_map), k)) for k in range(180)])
            modes = [(int(np.argmax(C))+d) % 360 for d in (0, 90, 180, 270)]
            sc = {p: self.Z.localize_buf(buf_b, [p], buf_b[-1]['scene'],
                                         self._open(frames_b[-1][1]))['score'] for p in modes}
            self.phi = max(sc, key=sc.get)
            self.get_logger().info(f"φ 확정 = {self.phi}  후보 {modes}  점수 "
                                   + str({k: round(v, 2) for k, v in sc.items()}))
        if len(buf_w) < 3: return
        r = self.Z.localize_buf(buf_w, [self.phi], buf_w[-1]['scene'],
                                self._open(frames_w[-1][1]))
        self._publish(r, t_now)

    def _cloud(self, frames):
        a0 = frames[0]; out = []
        for (t, p, xy, yaw) in frames:
            if len(p) > 250: p = p[np.linspace(0, len(p)-1, 250).astype(int)]
            out.append((xy-a0[2])[None, :] + p @ RT._R(yaw+RT.BETA).T)
        return np.vstack(out)

    def _open(self, pts):
        r = np.hypot(pts[:, 0], pts[:, 1])
        return float(np.median(np.sort(r)[:max(5, len(r)//5)]))

    def _publish(self, r, t_now):
        L = self.L
        ac = peaks({self.phi: r['P'][self.phi]}, L.Pf, allowed=self.Z.allow)[:self.K]
        px = r['pos']; mx = (px[0]-L.x0)/L.PPM; my = (L.rc-px[1])/L.PPM
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg(); msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(mx); msg.pose.pose.position.y = float(my)
        self.pub_pose.publish(msg)
        ma = MarkerArray()
        for i, (s_, p_, _) in enumerate(ac):
            mk = Marker(); mk.header = msg.header; mk.ns = 'topk'; mk.id = i
            mk.type = Marker.SPHERE; mk.action = Marker.ADD
            mk.pose.position.x = float((p_[0]-L.x0)/L.PPM)
            mk.pose.position.y = float((L.rc-p_[1])/L.PPM)
            mk.scale.x = mk.scale.y = mk.scale.z = 0.35 if i else 0.7
            mk.color.a = 0.9 if i == 0 else 0.45
            mk.color.r = 0.0 if i else 0.1; mk.color.g = 1.0 if i == 0 else 0.6
            mk.color.b = 0.2
            ma.markers.append(mk)
        self.pub_topk.publish(ma)
        self.get_logger().info(f"[{r['scene']}] pose=({mx:.2f}, {my:.2f}) φ={self.phi} "
                               f"프레임 {len(ac)}후보")


def main():
    rclpy.init(); n = RTLocalizerNode()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
