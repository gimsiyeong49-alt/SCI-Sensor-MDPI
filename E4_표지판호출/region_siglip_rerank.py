#!/usr/bin/env python3
"""벽 구간 상위 8개를 SigLIP으로 한 번 더 보고 순서만 다시 매긴다.

사물별 부정 문장(게시판·상자 등)은 쓰지 않는다. 긍정 문장 하나와의 유사도만 본다.
최종 순서 = (구간 점수 순위 + SigLIP 순위)의 평균. 가중치나 기준값을 따로 두지 않는다.
구간 전체를 224로 줄이면 가로 10 px 번호판이 더 작아지므로, 구간을 가로·세로 최대 100 px
조각으로 나눠 각각 보고 가장 높은 값을 그 구간 값으로 쓴다(crop=tile). crop=whole은 구간 통째.
입력: 31_벽구간이벤트/<tag>/regions.json. 정답은 채점에만 쓴다.

--nms  : 재정렬 뒤, 더 높은 순위 구간과 작은 쪽 면적의 절반 이상 겹치는 구간은 같은 자리로 보고 뺀다
         (같은 벽을 두 번 조준하지 않게. 후보를 지우는 게 아니라 중복 구간 하나로 센다).
--gate : 같은 장면에서 무작위 벽 조각(100 px, 화면 위쪽 2/3) 40개의 SigLIP 값 상위 10%를 기준으로,
         그보다 낮은 구간은 이벤트로 내지 않는다(--gate_pct로 백분위 변경). 기준은 정답이 아니라 그 장면 자체에서 나온다.
         표지 같은 곳이 없는 장면(유리 복도 등)에서 고화질 카메라를 헛되이 돌리지 않으려는 것.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import os
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
from event_criteria import SEGS, D  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
POS = "a small sign mounted on a wall"
# 2026-10-04 문구 묶음 선택(환경변수 KCI_SIGN_PROMPTS). 'room' = 호실 번호판을 직접 묻는 문구 3개(점수는 최댓값).
# 사람 판정 148개(2F·S1·7F·G1)에서 좁게 자른 상자 기준 번호판/오답 AUC: wall(기존) 대신 room 문구 0.78.
PROMPT_SETS = {"wall": [POS],
               "room": ["a room number sign next to a door", "a small room name plate on a wall", "a door sign with room number"],
               # 위치('문 옆') 가정을 뺀 대상 문구(2026-10-04 사용자 지적: 특정 배치를 박으면 범용성 해침)
               "plate": ["a room number sign", "a small room name plate", "a sign with a room number"],
               # 2026-10-05 판정 멈춤 179개로 문구 비교: 지금 문구 0.62, "a small room name plate" 0.67(AUC).
               # 사용자 선택(범용성: 위치 가정 없는 물건 이름 '호실 표지판').
               "roomplate": ["a small room name plate"],
               # 2026-10-05 표준(make_search_config.py COMMON["room"] 과 같음): 문 옆이 아닌 건물에 대비해 둘 중 최댓값.
               # 판정 멈춤 179개 AUC "...near a door" 0.76, 둘 중 최대 0.74. live_event_runner.py 기본값.
               "roomstd": ["a small room name plate", "a small room name plate near a door",
                           # 2026-10-06 복도 돌출 표지판 추가(make_search_config.py 와 같음)
                           "a hallway sign perpendicular to the wall"]}
# 2026-10-01 헷갈리는 것 문구. 번호판 멈춤 43번 중 약 17번이 조명·소화기함·창문·문·유리, 12번이 게시판·포스터였다.
# 점수(sim)는 그대로 두고, 가장 표지 같던 조각에서 '표지 − 헷갈리는 것 최고'(best_adv)만 따로 낸다.
CONF = ["a poster on a wall", "a notice board with papers", "a ceiling light", "a window", "a door",
        "a glass wall", "a fire extinguisher cabinet"]
# 2026-10-01 게시판 거르기: 구간 주변을 2.5배로 넓게 잘라 '게시판·포스터 − 문 옆 작은 표지' 점수를 본다.
# 100 px 조각으로 겨루면 번호판이 문·창과 같이 찍혀 졌다(CONF). 넓게 보면 게시판은 큰 종이판으로 보인다.
BOARD = ["a notice board with papers on a wall", "posters on a wall", "a bulletin board with many papers"]
PLATE = ["a door with a small room number sign", "a small sign next to a door", "a small plate on a wall"]
# 2026-10-03 천장 조명 거르기: 같은 넓은 조각에서 '천장 조명 − 작은 표지'(last_ceil)도 낸다.
# 8개 층 멈춤 82개 눈 분류에서 포화 > 5% 또는 (이 점수 > −0.01 이고 상자 중심 y < 200)이면 조명 27/30, 번호판 0/44.
CEIL = ["a ceiling light", "a ceiling with lights", "a bright lamp on the ceiling"]
MODEL = "/data2/kci_models/siglip-base-patch16-224"


class RegionSiglip:
    def __init__(self):
        proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
        self.model = AutoModel.from_pretrained(MODEL, local_files_only=True).to("cuda").eval().half()
        ip = proc.image_processor
        self.size = ip.size["height"]
        self.mean = np.array(ip.image_mean, np.float32)
        self.std = np.array(ip.image_std, np.float32)
        with torch.no_grad():
            # 2026-10-05: 층별 설정 파일(make_search_config.py)의 작은 표지 문구가 오면 그것을 쓴다
            pos = json.loads(os.environ["KCI_SIGN_PROMPTS_JSON"]) if os.environ.get("KCI_SIGN_PROMPTS_JSON") \
                else PROMPT_SETS[os.environ.get("KCI_SIGN_PROMPTS", "wall")]
            t = proc(text=pos, return_tensors="pt", padding="max_length").to("cuda")
            e = self.model.get_text_features(**t)
            e = getattr(e, "pooler_output", e)
            self.txt = torch.nn.functional.normalize(e.float(), dim=-1)
            t = proc(text=CONF, return_tensors="pt", padding="max_length").to("cuda")
            e = self.model.get_text_features(**t)
            e = getattr(e, "pooler_output", e)
            self.conf_txt = torch.nn.functional.normalize(e.float(), dim=-1)
            emb = []
            for ps in (BOARD, PLATE, CEIL):
                t = proc(text=ps, return_tensors="pt", padding="max_length").to("cuda")
                e = self.model.get_text_features(**t)
                emb.append(torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1))
            self.board_txt, self.plate_txt, self.ceil_txt = emb
        self.last_ceil = np.zeros(0)

    def _prep(self, rgb, b):
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        s = max(x1 - x0, y1 - y0, 32)                  # 정사각형으로 넓혀 비율 유지
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        c = rgb[int(max(0, cy - s / 2)):int(min(480, cy + s / 2)), int(max(0, cx - s / 2)):int(min(640, cx + s / 2))]
        if c.size == 0:
            c = np.zeros((8, 8, 3), np.uint8)
        c = cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return (c / 255.0 - self.mean) / self.std

    def board_scores(self, bgr, regions, k=2.5, smin=120):
        """구간마다 주변을 k배(최소 smin px)로 넓게 자른 한 장의 '게시판 − 작은 표지' 점수(> 0 이면 게시판 쪽)."""
        if not regions:
            return np.zeros(0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crops = []
        for b in regions:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(max(b[2] - b[0], b[3] - b[1]) * k, smin)
            x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            x1, y1 = int(min(rgb.shape[1], cx + s / 2)), int(min(rgb.shape[0], cy + s / 2))
            c = cv2.resize(rgb[y0:y1, x0:x1], (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
            crops.append((c / 255.0 - self.mean) / self.std)
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to("cuda").half()
        with torch.no_grad():
            e = self.model.get_image_features(pixel_values=x)
            e = torch.nn.functional.normalize(getattr(e, "pooler_output", e).float(), dim=-1)
            plate = (e @ self.plate_txt.T).max(1).values
            self.last_ceil = ((e @ self.ceil_txt.T).max(1).values - plate).cpu().numpy()
            return ((e @ self.board_txt.T).max(1).values - plate).cpu().numpy()

    def score(self, bgr, regions, crop="tile", tile=100):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crops, owner, subs_all = [], [], []
        for k, b in enumerate(regions):
            if crop == "whole":
                subs = [b]
            else:
                nx = max(1, int(np.ceil((b[2] - b[0]) / tile)))
                ny = max(1, int(np.ceil((b[3] - b[1]) / tile)))
                xs, ys = np.linspace(b[0], b[2], nx + 1), np.linspace(b[1], b[3], ny + 1)
                subs = [[xs[i], ys[j], xs[i + 1], ys[j + 1]] for i in range(nx) for j in range(ny)]
            for sb in subs:
                crops.append(self._prep(rgb, sb))
                owner.append(k)
                subs_all.append(sb)
        x = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to("cuda").half()
        with torch.no_grad():
            e = self.model.get_image_features(pixel_values=x)
            e = getattr(e, "pooler_output", e)
            en = torch.nn.functional.normalize(e.float(), dim=-1)
            sim = (en @ self.txt.T).max(1).values.cpu().numpy()   # 문구 여러 개면 최댓값(하나면 기존과 같음)
            conf = (en @ self.conf_txt.T).max(1).values.cpu().numpy()
        out = np.full(len(regions), -1.0)
        best = [None] * len(regions)
        adv = np.zeros(len(regions))
        for o, s, sb, c in zip(owner, sim, subs_all, conf):
            if s > out[o]:
                out[o], best[o], adv[o] = s, [float(x) for x in sb], s - c
        self.best_tiles = best                         # 구간마다 가장 표지 같던 조각 = 조준점 후보
        self.best_adv = adv                            # 그 조각에서 표지 − 헷갈리는 것 최고(> 0 이면 표지가 이김)
        return out


def inside(g, b):
    cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="03_기준_7m_조각1개")
    ap.add_argument("--out", default=None, help="결과 폴더(31_벽구간이벤트 기준). 없으면 입력 이름 뒤에 방식을 붙인다")
    ap.add_argument("--crop", default="tile", choices=["tile", "whole"])
    ap.add_argument("--set", default="4F", help="층 선택(sign_segsets)")
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate_pct", type=float, default=90)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rj = json.load(open(R / args.tag / "regions.json"))
    gt = {}
    import sign_segsets
    for name, gp in [(x["name"], x["gt"]) for x in sign_segsets.get(args.set)]:
        for l in open(gp):
            d = json.loads(l)
            gt[f"{name}_{float(d['t'])}"] = d
    vlm = RegionSiglip()
    vlm.score(np.zeros((480, 640, 3), np.uint8), [[0, 0, 200, 200]])   # 예열
    new, ms = {}, []
    for k, v in rj.items():
        regs = [b for _, b in v["regions"]]
        if not regs:
            new[k] = dict(v, regions=[])
            continue
        bgr = cv2.imread(str(D / v["frame"]))
        torch.cuda.synchronize()
        t0 = time.time()
        sim = vlm.score(bgr, regs, args.crop)
        tiles = vlm.best_tiles
        torch.cuda.synchronize()
        ms.append((time.time() - t0) * 1000)
        r_reg = np.arange(len(regs))                     # 이미 구간 점수순
        r_sig = np.argsort(np.argsort(-sim))
        order = sorted(range(len(regs)), key=lambda i: (r_reg[i] + r_sig[i], r_reg[i]))
        if args.nms:
            kept = []
            for i in order:
                b = regs[i]
                dup = False
                for j in kept:
                    c = regs[j]
                    ix = max(0, min(b[2], c[2]) - max(b[0], c[0]))
                    iy = max(0, min(b[3], c[3]) - max(b[1], c[1]))
                    amin = min((b[2] - b[0]) * (b[3] - b[1]), (c[2] - c[0]) * (c[3] - c[1]))
                    dup |= ix * iy >= 0.5 * amin
                if not dup:
                    kept.append(i)
            order = kept
        if args.gate:
            rb = []
            for _ in range(40):
                x, y = rng.uniform(0, 540), rng.uniform(0, 220)
                rb.append([x, y, x + 100, y + 100])
            base = float(np.percentile(vlm.score(bgr, rb, "whole"), args.gate_pct))
            order = [i for i in order if sim[i] > base]
        new[k] = dict(v, regions=[[float(sim[i]), regs[i]] for i in order], orig_rank=[int(i) for i in order],
                      tiles=[tiles[i] for i in order])

    objs, n_inst = set(), 0
    for k, d in gt.items():
        for (c, *g), o in zip(d["items"], d["object_id"]):
            if c == "plaque":
                objs.add(f"{k.rsplit('_', 1)[0]}#{o}")
                n_inst += 1
    lines = [f"층 {args.set}, 입력 {args.tag}, 자르기 {args.crop}, SigLIP {np.median(ms):.0f} ms/장면 (구간 최대 8개)"]
    for lab, src in (("이전", rj), ("SigLIP 재정렬", new)):
        for N in (1, 2, 3, 5):
            hi, ho, emp, tot = 0, set(), 0, 0
            for k, v in src.items():
                top = [b for _, b in v["regions"][:N]]
                d = gt[k]
                seg = k.rsplit("_", 1)[0]
                for (c, *g), o in zip(d["items"], d["object_id"]):
                    if c == "plaque" and any(inside(g, b) for b in top):
                        hi += 1
                        ho.add(f"{seg}#{o}")
                signs = [g for c, *g in d["items"] if c in ("plaque", "sign")]
                for b in top:
                    tot += 1
                    emp += not any(inside(g, b) for g in signs)
            lines.append(f"{lab:12s} N={N}: 번호판 관측 {hi}/{n_inst}, 물체 {len(ho)}/{len(objs)}, 헛구간 {emp}/{tot}")
    print("\n".join(lines))
    out = R / args.out if args.out else R / (f"{args.tag}_siglip_{args.crop}" + ("_tiles" if args.crop == "tile" else "") + ("_nms" if args.nms else "") + (f"_gate{args.gate_pct:g}" if args.gate else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_output.txt").write_text("\n".join(lines) + "\n")
    json.dump(new, open(out / "regions.json", "w"))


if __name__ == "__main__":
    main()
