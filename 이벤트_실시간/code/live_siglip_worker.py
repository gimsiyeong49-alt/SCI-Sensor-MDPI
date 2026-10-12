#!/usr/bin/env python3
"""실시간 이벤트 실행기용 SigLIP 작업자. vlm_env(torch)에서 돌고, 표준 입출력으로 주고받는다.

실행기(시스템 파이썬, OpenCV 4.5)는 torch가 없고, vlm_env는 OpenCV 5.0이라 소실점 함수가 깨진다.
그래서 SigLIP 점수만 이 프로세스에서 매긴다. 점수 코드는 region_siglip_rerank.RegionSiglip 그대로다.

요청: 4바이트 길이 + pickle {"bgr": ndarray, "regions": [[x0,y0,x1,y1], ...]}
응답: 4바이트 길이 + pickle {"sim": [float, ...], "sec": float}
"""

import os
import pickle
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from region_siglip_rerank import RegionSiglip  # noqa: E402


def read_msg(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    n = struct.unpack("<I", head)[0]
    return pickle.loads(f.read(n))


def write_msg(f, obj):
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                    # 라이브러리 출력이 응답 통로를 더럽히지 않게
    vlm = RegionSiglip()
    vlm.score(np.zeros((480, 640, 3), np.uint8), [[0, 0, 200, 200]])   # 첫 호출 준비(기존 단계와 같음)
    write_msg(out, {"ready": True})
    while True:
        req = read_msg(inp)
        if req is None:
            break
        t0 = time.time()
        regs = req["regions"]
        sim = vlm.score(req["bgr"], regs, "tile") if regs else np.zeros(0)
        adv = [float(a) for a in vlm.best_adv] if regs else []
        board = [float(v) for v in vlm.board_scores(req["bgr"], regs)] if (regs and req.get("board")) else []
        ceil = [float(v) for v in vlm.last_ceil] if board else []    # 천장 조명 − 작은 표지(2026-10-03)
        tiles = [list(b) for b in vlm.best_tiles] if regs else []    # 구간마다 SigLIP 이 가장 높던 조각(2026-10-05 --cand-tile)
        write_msg(out, {"sim": [float(s) for s in sim], "sec": time.time() - t0, "adv": adv, "board": board,
                        "ceil": ceil, "tiles": tiles})


if __name__ == "__main__":
    main()
