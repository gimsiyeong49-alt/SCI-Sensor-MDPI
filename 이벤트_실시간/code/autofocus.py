#!/usr/bin/env python3
"""선명도 되먹임으로 초점을 맞춘다.

초점링에는 위치 센서가 없고 기어가 가끔 헛돈다. 그래서 "몇 초 돌리면 어디"라는
계산은 못 쓴다. 위치를 기억하지 않고, 매 걸음마다 찍어서 선명도만 보고 판단한다.

3단계로 간다.

    1. 훑기    큰 걸음으로 한 방향을 훑어 정점을 지나칠 때까지 간다.
               끝에 닿으면(화면이 안 변하면) 방향을 한 번 뒤집는다.
    2. 되돌기  방향을 뒤집고 걸음을 줄여 정점을 다시 넘어간다.
    3. 다듬기  더 작은 걸음으로 한 번 더 반복한다.

각 단계는 "정점을 지나쳤다"는 것만 보고 넘어간다. 누적 시간으로 위치를 계산하지
않으므로 헛돌아도 판단이 틀어지지 않는다.

사용 예:
    python3 scripts/autofocus.py
    python3 scripts/autofocus.py --speed 150 --roi 0.25
"""

from __future__ import annotations

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pantilt_devices import find_camera, find_nano_port  # noqa: E402

import argparse
import glob
import statistics
import sys
import threading
import time

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("opencv가 없습니다.")
try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.exit("numpy가 없습니다.")
try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    sys.exit("pyserial이 없습니다.")


def find_port() -> str:
    # 번호가 아니라 CH340 칩으로 찾는다. Go2처럼 다른 시리얼 장치가 있어도 안전하다.
    return find_nano_port()

class Camera:
    """프레임 읽기와 화면 표시를 전용 스레드에서 계속 돌린다.

    자동초점 쪽에서 필요할 때만 읽으면, 모터를 돌리고 진동을 기다리는 동안
    화면이 멈춘다. 그래서 읽기는 항상 돌아가게 두고 최신 프레임만 가져다 쓴다.
    버퍼에 쌓인 옛 프레임을 읽는 문제도 이 구조에서 자연히 사라진다.
    """

    def __init__(self, device: str, width: int, height: int, roi: float, samples: int,
                 roi_cx: float = 0.5, roi_cy: float = 0.5, view: bool = False):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"{device} 를 열지 못했습니다. 화면 보기 창이 떠 있으면 닫으세요.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.roi = roi
        self.roi_cx, self.roi_cy = roi_cx, roi_cy
        self.samples = samples
        self.view = view
        self.latest = None          # (선명도, 축소 흑백)
        self.note = ""              # 화면에 같이 띄울 진행 상황
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        time.sleep(1.0)             # 자동노출이 자리잡을 시간

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            rw, rh = int(w * self.roi), int(h * self.roi)
            x0 = max(0, min(w - rw, int(w * self.roi_cx) - rw // 2))
            y0 = max(0, min(h - rh, int(h * self.roi_cy) - rh // 2))
            gray = cv2.cvtColor(frame[y0:y0 + rh, x0:x0 + rw], cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            small = cv2.resize(gray, (160, 120)).astype(np.float32)
            with self._lock:
                self.latest = (sharp, small)
                note = self.note
            if self.view:
                shown = frame
                cv2.rectangle(shown, (x0, y0), (x0 + rw, y0 + rh), (0, 255, 0), 3)
                cv2.putText(shown, f"sharpness {sharp:.1f}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                if note:
                    cv2.putText(shown, note, (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("autofocus", cv2.resize(shown, (960, 540)))
                cv2.waitKey(1)

    def set_note(self, text: str) -> None:
        with self._lock:
            self.note = text

    def measure(self):
        """최근 몇 장의 중앙값. 진동이나 노출 흔들림에 덜 속는다."""
        vals, img = [], None
        for _ in range(self.samples):
            with self._lock:
                latest = self.latest
            if latest is not None:
                vals.append(latest[0])
                img = latest[1]
            time.sleep(0.06)
        if not vals:
            sys.exit("프레임을 읽지 못했습니다.")
        return statistics.median(vals), img

    def latest_sharp(self) -> float:
        with self._lock:
            return self.latest[0] if self.latest else 0.0

    def save(self, path: str) -> None:
        ok, frame = self.cap.read()
        if ok:
            cv2.imwrite(path, frame)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()


class Focus:
    def __init__(self, port: str, speed: int, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)
        self.ser.reset_input_buffer()
        self.speed = speed

    def move(self, direction: int, ms: int) -> None:
        self.ser.write(f"F{self.speed * direction},{ms}\n".encode())
        self.ser.flush()
        time.sleep(ms / 1000.0 + 0.35)   # 구동 시간 + 진동이 가라앉을 시간
        while self.ser.in_waiting:
            self.ser.readline()

    def start(self, direction: int, ms: int) -> None:
        """멈추라고 할 때까지 계속 돈다. ms는 안전용 상한이다."""
        self.ser.write(f"F{self.speed * direction},{ms}\n".encode())
        self.ser.flush()

    def stop(self) -> None:
        self.ser.write(b"F0\n")
        self.ser.flush()

    def close(self):
        self.ser.write(b"F0\n")
        self.ser.flush()
        time.sleep(0.2)
        self.ser.close()


def takeup_lash(cam, foc, direction, probe_ms=80, max_probes=10, thresh=3.0):
    """방향을 바꾼 직후의 기어 유격을 메운다.

    유격 안에서는 모터만 돌고 초점링은 안 움직인다. 그 구간의 측정값을 그대로
    믿으면 오르막·내리막 판단이 어긋난다. 화면이 실제로 변하기 시작할 때까지
    짧게 밀어 놓고 나서 측정을 시작한다.
    """
    _, prev = cam.measure()
    for i in range(1, max_probes + 1):
        foc.move(direction, probe_ms)
        _, img = cam.measure()
        d = float(np.mean(np.abs(img - prev)))
        if d > thresh:
            print(f"  유격 메움 {i}걸음({i * probe_ms}ms)에서 움직이기 시작")
            return i
        prev = img
    print(f"  유격 메움 {max_probes}걸음에도 변화 없음 (끝일 수 있음)")
    return max_probes


def run_pass(cam, foc, direction, step_ms, max_steps, best, label, drop_ratio,
             stall_limit=3):
    """한 방향으로 훑는다. 정점을 지나치거나 끝에 닿으면 멈춘다.

    반환값은 (이번 구간 최고 선명도, 끝에 닿았는지, 방향)이다.
    """
    prev_s, prev_img = cam.measure()
    # 이번 구간의 최고값만 기준으로 삼는다. 앞 단계에서 찍은 높은 값을 그대로
    # 들고 오면 이후 측정이 전부 "정점을 지나쳤다"로 판정돼 한 걸음 만에 끝난다.
    local_best = prev_s
    stall = 0
    drops = 0
    for i in range(1, max_steps + 1):
        cam.set_note(f"{label} {i}  step {step_ms}ms  dir {'+' if direction > 0 else '-'}")
        foc.move(direction, step_ms)
        s, img = cam.measure()
        moved = float(np.mean(np.abs(img - prev_img))) if prev_img is not None else 99
        note = ""
        if moved < 1.5:
            stall += 1
            note = f"화면 변화 없음 {stall}/{stall_limit}"
        else:
            stall = 0
        if s > local_best:
            local_best = s
            note = note or "최고 갱신"
        print(f"  {label} {i:2d}  {'+' if direction > 0 else '-'} {step_ms:5d}ms  "
              f"선명도 {s:8.1f}  이동 {moved:6.1f}  {note}")
        prev_s, prev_img = s, img

        if stall >= stall_limit:
            print(f"  → 끝에 닿았습니다. 방향을 뒤집습니다.")
            return local_best, True, -direction
        if local_best > 5.0 and s < local_best * drop_ratio:
            drops += 1
            # 잡음 한 번에 속지 않도록 두 번 연속 하락을 요구한다.
            if drops >= 2:
                print(f"  → 정점을 지나쳤습니다(최고 {local_best:.1f} → 지금 {s:.1f}).")
                return local_best, False, -direction
        else:
            drops = 0
    return local_best, False, -direction


def sweep_continuous(cam, foc, direction, speed, label, drop_ratio, max_ms=4000,
                     min_peak=8.0, confirm=3):
    """계속 돌리면서 매 프레임 보다가, 정점을 지나는 순간 바로 멈춘다.

    걸음으로 끊어 움직이면 정점이 걸음 사이에 들어가 버린다. 계속 돌리면 정지
    정밀도가 걸음 크기가 아니라 프레임 간격으로 결정되므로 훨씬 곱게 선다.
    """
    foc.speed = speed
    cam.set_note(f"{label} {'+' if direction > 0 else '-'} speed {speed}")
    start_s = cam.latest_sharp()
    best, best_t = start_s, 0.0
    t0 = time.time()
    foc.start(direction, max_ms)
    stopped_by = "시간 초과"
    below = 0
    while True:
        elapsed = (time.time() - t0) * 1000
        if elapsed >= max_ms:
            break
        s = cam.latest_sharp()
        if s > best:
            best, best_t = s, elapsed
            below = 0
        # 정점으로 인정하려면 출발값보다 확실히 높아야 한다. 안 그러면 잡음 하나에
        # "정점을 지났다"고 판정해 몇십 ms 만에 멈춰 버린다.
        elif best > max(min_peak, start_s * 1.3) and s < best * drop_ratio:
            below += 1
            if below >= confirm:   # 연속 몇 프레임 낮아야 인정한다
                stopped_by = f"정점 통과 감지 (최고 {best:.0f} → {s:.0f})"
                break
        else:
            below = 0
        time.sleep(0.008)
    foc.stop()
    time.sleep(0.4)
    final = cam.latest_sharp()
    print(f"  {label}: {stopped_by}, 구동 {elapsed:.0f}ms, 정점은 {best_t:.0f}ms 지점, "
          f"멈춘 뒤 {final:.1f}")
    return best, best_t, elapsed, final


def stepwise(cam, foc, direction: int = 1, speed: int = 150, coarse_ms: int = 500,
             fine_speed: int = 120, fine_ms: int = 40):
    """단계식 자동초점. 2026-09-16 에 정점을 가장 잘 찾은 방식이다(최고 112·174·400·143).

    같은 날 시험한 연속 훑기(--continuous)는 49·29·3·4로 대부분 실패했다. 조준 콘솔도
    이 함수를 그대로 불러 쓴다(2026-09-19). 반환 (시작, 끝, 구간 최고) 선명도.
    """
    foc.speed = speed
    start, _ = cam.measure()
    print(f"시작 선명도 {start:.1f}, 속도 {speed}\n")
    best = start
    # 1단계 훑기. 끝에 닿으면 반대로 한 번 더 훑는다.
    print("1단계 훑기")
    b1, hit_end, direction = run_pass(cam, foc, direction, coarse_ms, 30,
                                      best, "훑기", 0.55)
    best = max(best, b1)
    if hit_end:
        takeup_lash(cam, foc, direction)
        b1b, _, direction = run_pass(cam, foc, direction, coarse_ms, 30,
                                     best, "훑기", 0.55)
        best = max(best, b1b)

    # 2단계 되돌기. 걸음을 줄여 정점을 반대쪽에서 다시 넘는다.
    print("\n2단계 되돌기")
    takeup_lash(cam, foc, direction)
    b2, _, direction = run_pass(cam, foc, direction, max(120, coarse_ms // 3),
                                16, best, "되돌", 0.75)
    best = max(best, b2)

    # 3단계 다듬기.
    print("\n3단계 다듬기")
    takeup_lash(cam, foc, direction, probe_ms=60)
    b3, _, direction = run_pass(cam, foc, direction, max(60, coarse_ms // 8),
                                12, best, "다듬", 0.85)
    best = max(best, b3)

    # 4단계. 가장 잘게 한 번 더. 심도가 얕아 정점이 좁다.
    print("\n4단계 미세")
    takeup_lash(cam, foc, direction, probe_ms=60)
    b4, _, direction = run_pass(cam, foc, direction, 60, 10, best, "미세", 0.9)
    best = max(best, b4)

    # 5단계. 정점에 서기. 앞 단계들은 "지나쳤다"를 보고 끝나므로 항상 정점을 조금 지나쳐 있다.
    # 2026-09-19: 예전엔 "더 안 오르면 멈춤"이었는데, 미세 속도 120에서 모터가 거의 안 움직여
    # 들어가자마자 멈췄고, 최고 352를 보고도 137(39%)에 섰다. 이제 되돌아가며 최고의 85%에
    # 닿는 순간 멈춘다. 또 지나치면 방향을 바꾸고, 그 구간 최고에 맞춰 목표를 낮춰 다시 간다.
    print("\n5단계 정점에 서기")
    foc.speed = fine_speed
    target = best * 0.85
    floor = best * 0.6          # 목표를 낮춰도 최고의 60% 아래로는 안 내린다
    step_ms = fine_ms
    s, _ = cam.measure()
    for attempt in range(1, 7):
        if s >= target:
            break
        takeup_lash(cam, foc, direction, probe_ms=step_ms)
        s, _ = cam.measure()
        pass_best, drops = s, 0
        for i in range(1, 40):
            if s >= target:
                break
            foc.move(direction, step_ms)
            s, _ = cam.measure()
            print(f"  서기{attempt} {i:2d}  {'+' if direction > 0 else '-'} {step_ms:4d}ms  "
                  f"선명도 {s:8.1f}  (목표 {target:.0f})")
            if s > pass_best:
                pass_best, drops = s, 0
            elif s < pass_best * 0.85:
                drops += 1
                if drops >= 2:      # 이 구간 정점을 또 지나쳤다
                    break
        best = max(best, pass_best)
        if s >= target:
            break
        direction = -direction
        # 이 구간에서 목표까지 못 올랐으면 정점이 생각보다 낮거나 한 걸음에 건너뛴 것이다.
        # 목표는 그 높이에 맞춰 낮추되 하한을 두고, 걸음은 반으로 줄인다(최소 30ms).
        target = max(floor, min(target, pass_best * 0.85))
        step_ms = max(30, step_ms // 2)
        print(f"  → 또 지나침. 반대로, 목표 {target:.0f}, 걸음 {step_ms}ms")
    if s >= target:
        print(f"  → 최고 {best:.0f}의 {s / max(best, 1):.0%}에서 멈춤")

    final, _ = cam.measure()
    print(f"\n시작 {start:.1f} → 지금 {final:.1f} (구간 최고 {best:.1f})")
    return start, final, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", help="생략하면 이름(Arducam)으로 찾는다. 노트북 video4, Go2 video6")
    ap.add_argument("--width", type=int, default=1280, help="측정용. 낮을수록 빠르다")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--roi", type=float, default=0.4)
    ap.add_argument("--roi-center", nargs=2, type=float, default=[0.5, 0.5],
                    metavar=("CX", "CY"),
                    help="측정 영역의 중심 위치를 화면 비율로 지정 (0~1). 무늬가 있는 곳을 골라야 한다")
    ap.add_argument("--port")
    ap.add_argument("--speed", type=int, default=150, help="헛돌지 않는 대역에서 고른다")
    ap.add_argument("--coarse-ms", type=int, default=500)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--dir", type=int, default=1, choices=[1, -1], help="처음 훑을 방향")
    ap.add_argument("--fine-speed", type=int, default=120,
                    help="마지막 정점 정지 단계의 속도. 느릴수록 한 걸음이 짧다")
    ap.add_argument("--fine-ms", type=int, default=40)
    ap.add_argument("--save")
    ap.add_argument("--view", action="store_true", help="진행 화면을 창으로 보여준다")
    ap.add_argument("--continuous", action="store_true",
                    help="끊어 움직이지 않고 계속 돌리며 정점에서 즉시 멈춘다")
    args = ap.parse_args()
    args.device = args.device or find_camera()

    cam = Camera(args.device, args.width, args.height, args.roi, args.samples,
                 args.roi_center[0], args.roi_center[1], args.view)
    foc = Focus(args.port or find_port(), args.speed)
    try:
        start, _ = cam.measure()
        print(f"시작 선명도 {start:.1f}, 속도 {args.speed}\n")
        best = start
        direction = args.dir

        if args.continuous:
            # 1) 한 방향으로 훑으며 정점이 몇 ms 지점이었는지 기록한다.
            #    정점이 안 나오면 반대 방향으로 한 번 더 훑는다.
            sweep_speed = args.speed
            b, b_t, elapsed, final = sweep_continuous(
                cam, foc, direction, sweep_speed, "훑기", 0.6, 6000)
            if b < start * 1.5 and b < 40:
                print("  이 방향에서 정점을 못 봤습니다. 반대로 훑습니다.")
                direction = -direction
                takeup_lash(cam, foc, direction, probe_ms=60)
                b, b_t, elapsed, final = sweep_continuous(
                    cam, foc, direction, sweep_speed, "훑기2", 0.6, 8000)
            best = max(best, b)

            # 2) 지나친 만큼 되돌아간다. 같은 속도로 돌리면 거리도 대략 같다.
            for attempt in range(1, 4):
                back_ms = int(max(0, elapsed - b_t))
                if back_ms < 20:
                    break
                print(f"\n되돌아가기 {attempt}: {back_ms}ms 만큼 반대로")
                direction = -direction
                takeup_lash(cam, foc, direction, probe_ms=60)
                foc.speed = sweep_speed
                cam.set_note(f"return {back_ms}ms")
                foc.move(direction, back_ms)
                got, _ = cam.measure()
                print(f"  되돌아간 자리 선명도 {got:.1f} (최고 {best:.1f}의 {got / max(best, 1):.0%})")
                if got >= best * 0.7:
                    print("  → 초점 근처입니다.")
                    break
                # 못 맞췄으면 그 부근만 다시 훑는다. 속도는 낮추지 않는다.
                # 140 아래로 내리면 기어가 물지 못해 아예 안 도는 것을 실측했다.
                # 정밀도는 속도가 아니라 구동 시간으로 줄인다.
                sweep_speed = max(140, sweep_speed)
                b2, b_t, elapsed, final = sweep_continuous(
                    cam, foc, direction, sweep_speed, f"재훑기{attempt}", 0.7,
                    max(400, 1500 // attempt))
                best = max(best, b2)

            final, _ = cam.measure()
            print(f"\n시작 {start:.1f} → 지금 {final:.1f} (구간 최고 {best:.1f})")
            if args.save:
                cam.save(args.save)
                print(f"저장: {args.save}")
            return 0

        _, final, best = stepwise(cam, foc, direction, args.speed, args.coarse_ms,
                                  args.fine_speed, args.fine_ms)
        if args.save:
            cam.save(args.save)
            print(f"저장: {args.save}")
    finally:
        foc.close()
        cam.close()
        if args.view:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
