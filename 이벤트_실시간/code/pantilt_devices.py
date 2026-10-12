#!/usr/bin/env python3
"""팬틸트 카메라와 나노를 장치 번호가 아니라 이름으로 찾는다.

노트북에서는 IMX477이 /dev/video4였지만 Go2에서는 RealSense가 video0~5를
차지해서 video6이 된다. 번호를 박아두면 컴퓨터를 옮길 때마다 틀린다.
"""

from __future__ import annotations

import glob
import os
import sys

# 나노 클론마다 USB-시리얼 칩이 다르다. 2026-09-16 보드는 CH340, 2026-09-18 교체한
# 보드는 FTDI FT232R다. 둘 다 받아 준다.
NANO_VENDORS = {
    "1a86": "CH340",
    "0403": "FTDI",
    "10c4": "CP210x",
    "2341": "Arduino 정품",
}


def find_camera(name_part: str = "Arducam") -> str:
    """이름에 name_part가 들어간 첫 캡처 장치. 같은 카메라가 video 노드를 두 개
    만드는데(영상, 메타데이터), 번호가 작은 쪽이 영상이다."""
    hits = []
    for path in glob.glob("/sys/class/video4linux/video*"):
        try:
            with open(os.path.join(path, "name")) as f:
                name = f.read().strip()
        except OSError:
            continue
        if name_part.lower() in name.lower():
            hits.append(int(os.path.basename(path)[5:]))
    if not hits:
        sys.exit(f"'{name_part}' 카메라를 찾지 못했습니다. USB 연결을 확인하세요.")
    return f"/dev/video{min(hits)}"


def _usb_vendor(dev: str) -> str | None:
    """시리얼 장치 노드에서 USB 벤더 ID를 거슬러 올라가 찾는다."""
    node = os.path.realpath(f"/sys/class/tty/{os.path.basename(dev)}/device")
    for _ in range(4):
        vendor_file = os.path.join(node, "idVendor")
        if os.path.exists(vendor_file):
            with open(vendor_file) as f:
                return f.read().strip()
        node = os.path.dirname(node)
    return None


def find_nano_port() -> str:
    """나노가 달린 시리얼 포트. 다른 시리얼 장치가 있어도 헷갈리지 않는다."""
    candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    for dev in candidates:
        vendor = _usb_vendor(dev)
        if vendor in NANO_VENDORS:
            return dev
    if candidates:
        print(f"아는 USB-시리얼 칩이 없어 첫 포트를 씁니다: {candidates[0]}", file=sys.stderr)
        return candidates[0]
    sys.exit("나노 시리얼 포트를 찾지 못했습니다. USB 연결을 확인하세요.")


if __name__ == "__main__":
    print("카메라:", find_camera())
    port = find_nano_port()
    vendor = _usb_vendor(port)
    print("나노:  ", port, f"({NANO_VENDORS.get(vendor, vendor)})")
