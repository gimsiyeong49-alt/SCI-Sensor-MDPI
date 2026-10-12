#!/usr/bin/env python3
"""Go2 조준 콘솔의 실시간 주소(/live.mjpg, /live/depth.png, /live/odom, /aim)를 녹화 데이터로 흉내 낸다.

필드에 나가기 전에 노트북에서 live_event_runner.py live 의 받기·시각·깊이·오돔·기록 흐름을 시험하려고 만든다.
컬러는 녹화 프레임을 JPEG(품질 90)로, 녹화 속도 그대로 보낸다(콘솔과 같은 인코딩).
깊이는 캐시(1초 간격)에서 지금 재생 시각에 가장 가까운 것을, 오돔은 FAST-LIO odom_link 를 보간해서 준다.
yaw 에는 4층 규약(228°)을 더해 'RealSense 가 보는 방향'으로 준다(Go2 오돔을 그대로 쓰는 실제 경우와 같은 의미).

    python3 scripts/fake_go2_console.py --port 8099

2026-10-05: 다른 층(5 Hz 녹화, --time-col bag_relative_sec), 위치추정 1초 기록(--loc, /live/loc),
깊이·오돔을 요청한 시각(?stamp=)으로 주기를 더했다. 프레임 시각(X-Stamp)은 t0 + --time-col 값이다(replay 와 같은 시각축).
/live/loc?after=<stamp> 는 재생 시각(−--loc-delay)까지 나온 기록 중 stamp 뒤의 것을 JSON 목록으로 준다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import region_memory as M  # noqa: E402

S = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/live.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            rows, root = S["rows"], S["root"]
            t_wall0 = time.time()
            S["t_wall0"] = t_wall0
            tc = S["time_col"]
            h0 = float(rows[0][tc])
            try:
                for r in rows:
                    rel = float(r[tc]) - h0
                    lag = rel - (time.time() - t_wall0)
                    if lag > 0:
                        time.sleep(lag)
                    img = cv2.imread(str(root / r["filename"]))
                    jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                    st = S["t0"] + float(r[tc])
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     + f"X-Stamp: {st:.6f}\r\nX-Seq: {r['frame_index']}\r\n".encode()
                                     + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        # 재생 시각(t0 기준 초). ?stamp= 를 주면 그 시각(실행기가 장면 시각을 보냄; 실제 콘솔은 무시하고 최신 것을 줌)
        now = S["h0"] + time.time() - S.get("t_wall0", time.time())
        req = float(q["stamp"][0]) - S["t0"] if "stamp" in q else now
        if path == "/live/loc":
            after = float(q.get("after", ["-1e18"])[0])
            upto = S["t0"] + now - S["loc_delay"]
            body = json.dumps([d for d in S["loc"] if after < d["stamp"] <= upto]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/live/depth.png":
            keys = S["dkeys"]
            k = keys[int(np.argmin(np.abs(np.array(keys) - req)))]
            body = cv2.imencode(".png", S["dz"][f"{k:.1f}"])[1].tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("X-Stamp", f"{S['t0'] + k:.6f}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/live/odom":
            x, y, yaw = S["odom"](req)
            body = json.dumps(dict(x=float(x), y=float(y), yaw=float(yaw), stamp=S["t0"] + req)).encode()
        elif path == "/aim":
            body = "조준 요청 받음(흉내)".encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data2/kci_datasets/4F_RGB_FULL_V0_1")
    ap.add_argument("--depth", default="/data3/kci_datasets/depth_cache_sign_eval/4F_full.npz")
    ap.add_argument("--odom", default="/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv")
    ap.add_argument("--t0", type=float, default=1783340324.059948032)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--time-col", default="bag_relative_sec", help="프레임 시각 열(4층 전체 주행 4F_RGB_FULL_V0_1 도 bag 시각으로 맞춤)")
    ap.add_argument("--loc", default="", help="위치추정 1초 기록 폴더(loc_topk_series.py 출력: timeline.csv, topk_series.csv)")
    ap.add_argument("--loc-delay", type=float, default=0.0, help="위치추정 기록이 늦게 나오는 시간(초) 흉내")
    a = ap.parse_args()
    S["root"] = Path(a.root)
    S["rows"] = list(csv.DictReader(open(S["root"] / "index.csv")))
    S["dz"] = np.load(a.depth)
    S["dkeys"] = sorted(float(k) for k in S["dz"].keys())
    S["odom"] = M.Odom(a.odom, a.t0)
    S["t0"] = a.t0
    S["time_col"] = a.time_col
    S["h0"] = float(S["rows"][0][a.time_col])
    S["loc_delay"] = a.loc_delay
    S["loc"] = []
    if a.loc:
        tk = {}
        for r in csv.DictReader(open(Path(a.loc) / "topk_series.csv")):
            tk.setdefault(float(r["relative_sec"]), []).append(
                [float(r["score"]), float(r["x_px"]), float(r["y_px"]), float(r["yaw_rad"])])
        for r in csv.DictReader(open(Path(a.loc) / "timeline.csv")):
            t = float(r["relative_sec"])
            S["loc"].append(dict(stamp=a.t0 + t, level=r["confidence_level"], margin=float(r["confidence_margin"]),
                                 ppm=float(r["ppm_est"]), odom=[float(r["odom_x_m"]), float(r["odom_y_m"])],
                                 odom_yaw=float(r["odom_yaw_rad"]), cands=tk[t]))
    print(f"흉내 콘솔 http://127.0.0.1:{a.port}  (프레임 {len(S['rows'])}장, 위치추정 기록 {len(S['loc'])}초)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
