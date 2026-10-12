#!/usr/bin/env python3
"""실시간 실행기용 VLM(Qwen3-VL) 작업자. vlm_env 에서 돌고 표준 입출력으로 주고받는다(live_siglip_worker 와 같은 방식).

쓰임: 시설확인(게이트 켜짐 + 조준 후보 없음이 이어질 때)에서 RealSense 화면에 '보이는 것'을 묻는다.
질문은 vlm_facility_check.py 의 "what" 과 같다(열린 질문. 예/아니오는 2B 가 거의 다 '예'라 쓰지 않음).

요청: 4바이트 길이 + pickle {"bgr": ndarray}                       → {"answer": str, "sec": float}
      {"bgr": ndarray, "box_label": "elevator door"} → {"boxes": [[x0,y0,x1,y1] 픽셀], "answer": str, "sec": float}
상자 질문은 '있다'고 전제하고 묻는 것이라 없는 데도 아무 문이나 짚는다(2026-10-01 4층 103초). 검출 뒤 위치·거리용으로만 쓴다.
"""

import os
import pickle
import re
import struct
import sys
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = os.environ.get("KCI_VLM_MODEL", "/data3/kci_models/Qwen3-VL-2B-Instruct")
PROMPT = "List the main things you can see in this image, in a few words separated by commas."


def read_msg(f):
    head = f.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(f.read(struct.unpack("<I", head)[0]))


def write_msg(f, obj):
    b = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                                        device_map="cuda").eval()
    write_msg(out, {"ready": True})
    while True:
        req = read_msg(inp)
        if req is None:
            break
        t0 = time.time()
        img = Image.fromarray(cv2.cvtColor(req["bgr"], cv2.COLOR_BGR2RGB))
        lab = req.get("box_label")
        q = PROMPT if not lab else (f"Locate every {lab} in the image. Output a JSON list like "
                                    f'[{{"bbox_2d":[x1,y1,x2,y2],"label":"{lab}"}}]. If there is none, output [].')
        x = proc.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": img},
                                                                   {"type": "text", "text": q}]}],
                                     tokenize=True, add_generation_prompt=True, return_dict=True,
                                     return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=30 if not lab else 160, do_sample=False)
        ans = proc.batch_decode(o[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rep = {"answer": ans, "sec": time.time() - t0}
        if lab:                                  # Qwen3-VL 상자는 0~1000 정규 좌표
            h, w = req["bgr"].shape[:2]
            rep["boxes"] = [[float(a) / 1000 * w, float(b) / 1000 * h, float(c) / 1000 * w, float(d) / 1000 * h]
                            for a, b, c, d in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", ans)]
        write_msg(out, rep)


if __name__ == "__main__":
    main()
