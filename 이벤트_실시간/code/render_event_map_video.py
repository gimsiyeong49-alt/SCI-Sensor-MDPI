#!/usr/bin/env python3
"""모호함 → 이벤트 → 조준을 13번 폴더 영상과 같은 구성으로 본다(층 전체 주행, 배속).

왼쪽 위: RealSense 영상. 번호판 멈춤 = 노란 상자 + 빨간 십자(조준점), 시설 관측 = 파란 상자 + 종류·거리(계단은 RealSense+VLM 으로 걸으면서 판단, 멈추지 않음).
오른쪽 위: ① 위치추정(확신/모호함, 점수 차, 튐 확인) ② 이벤트 ③ 조준, 누적 횟수.
아래: 층 의미지도 위에 위치추정 후보(빨강 1위, 주황 = 1위 점수 1.5 안, 회색 나머지), 1위 발자취, 호실·시설 이름.
게이트 = 모호함 또는 (튐이면서 비슷한 후보 퍼짐 ≥ 5 m). 정답 위치는 그리지 않는다.
멈춤 순간은 --hold 초 동안 정지 화면으로 보여 준다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONTS = {}
KIND_COL = {"room": (70, 70, 200), "stair": (160, 90, 20), "elevator": (0, 140, 70), "restroom": (150, 0, 170)}


def font(sz):
    if sz not in FONTS:
        FONTS[sz] = ImageFont.truetype(FONT, sz)
    return FONTS[sz]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", required=True, help="제목용 층 이름(예: 4층)")
    ap.add_argument("--root", required=True, help="RGB 데이터셋(index.csv)")
    ap.add_argument("--time-col", default="header_relative_sec")
    ap.add_argument("--loc", required=True, help="위치추정 1초 기록 폴더(timeline.csv, topk_series.csv)")
    ap.add_argument("--gate", required=True)
    ap.add_argument("--events", required=True, help="번호판 멈춤(events_live_replay.json)")
    ap.add_argument("--facility", default="", help="시설 관측(facility_events.json)")
    ap.add_argument("--map", required=True)
    ap.add_argument("--sem", required=True)
    ap.add_argument("--ppm", type=float, required=True)
    ap.add_argument("--crop", type=int, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} 이 이미 있다(덮어쓰지 않음)")

    # 위치추정·게이트
    tl = list(csv.DictReader(open(Path(a.loc) / "timeline.csv")))
    tk = collections.defaultdict(list)
    for r in csv.DictReader(open(Path(a.loc) / "topk_series.csv")):
        tk[float(r["relative_sec"])].append(r)
    gate = {float(x["t"]): x for x in json.load(open(a.gate))}
    st, prev = {}, None
    for r in tl:
        t = float(r["relative_sec"])
        c = sorted(tk[t], key=lambda x: -float(x["score"]))
        top = (float(c[0]["x_px"]), float(c[0]["y_px"]))
        od = (float(r["odom_x_m"]), float(r["odom_y_m"]))
        jump = None
        if prev is not None:
            jump = (np.hypot(top[0] - prev[0][0], top[1] - prev[0][1]) / a.ppm, np.hypot(od[0] - prev[1][0], od[1] - prev[1][1]))
        prev = (top, od)
        g = gate[t]
        st[t] = dict(cands=c, jump=jump, g=g, on=bool(g["amb"] or (g["jump"] and g["spread"] >= 5)))
    loc_t = np.array(sorted(st))

    sign = {e["t"]: e for e in json.load(open(a.events))}
    fac = {e["t"]: e for e in json.load(open(a.facility))} if a.facility else {}
    stops = sorted([(t, "번호판") for t in sign] + [(t, "시설") for t in fac])

    # 지도
    im = cv2.imread(a.map)
    if a.crop:
        cx0, cy0, cx1, cy1 = a.crop
    else:
        ys, xs = np.nonzero(cv2.imread(a.map, 0) < 50)
        cx0, cy0, cx1, cy1 = max(0, xs.min() - 20), max(0, ys.min() - 20), min(im.shape[1], xs.max() + 20), min(im.shape[0], ys.max() + 20)
    MW = 1540
    sc = MW / (cx1 - cx0)
    MH = int((cy1 - cy0) * sc)
    mp = lambda x, y: ((x - cx0) * sc, (y - cy0) * sc)
    base_map = Image.fromarray(cv2.cvtColor(cv2.resize(im[cy0:cy1, cx0:cx1], (MW, MH)), cv2.COLOR_BGR2RGB))
    bd = ImageDraw.Draw(base_map)
    for sp in json.load(open(a.sem))["spaces"]:
        x, y = mp(*sp["center_px"])
        if 0 <= x < MW and 0 <= y < MH:
            kind = sp["type"] if sp["type"] in KIND_COL else "room"
            name = sp.get("name") or sp["type"]
            short = {"elevator": "EV", "restroom": name.replace("자화장실", "화장실"),
                     "stair": name.replace(" 비상계단", "비상").replace(" 계단", "계단")}.get(kind)
            lab = str(sp.get("room_number") or "") or short or name[:5]
            bd.text((x - 14, y - 10), lab, font=font(15), fill=KIND_COL[kind])

    rows = list(csv.DictReader(open(Path(a.root) / "index.csv")))
    ts = np.array([float(r[a.time_col]) for r in rows])
    W, H = MW, 480 + MH
    fps = 15
    src_fps = 1.0 / float(np.median(np.diff(ts)))         # 4층 원본 약 15장/초, 5·6층 추출본 5장/초
    stride = max(1, int(round(a.speed * src_fps / fps)))
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    counts = {"번호판": 0, "시설": 0}
    trail, last_lt = [], None

    def frame(t, bgr, stop=None):
        nonlocal last_lt
        k = loc_t[np.searchsorted(loc_t, t + 1e-6) - 1] if t >= loc_t[0] else None
        s = st.get(k) if k is not None else None
        if s is not None and k != last_lt:
            last_lt = k
            trail.append((float(s["cands"][0]["x_px"]), float(s["cands"][0]["y_px"])))
        rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(rgb)
        aim = None
        if stop and stop[1] == "번호판":
            e = sign[stop[0]]
            b = [float(v) for v in e["picks"][0]]
            u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            aim = (u, v, e["z"][0], b)
            dr.rectangle(b, outline=(255, 190, 0), width=4)
            dr.line([u - 14, v, u + 14, v], fill=(255, 0, 0), width=3)
            dr.line([u, v - 14, u, v + 14], fill=(255, 0, 0), width=3)
            dr.rectangle([6, 6, 300, 42], fill=(255, 190, 0))
            dr.text((12, 8), f"번호판 멈춤 (u={u:.0f}, v={v:.0f})", font=font(22), fill=(0, 0, 0))
        elif stop and stop[1] == "시설":
            e = fac[stop[0]]
            b = [float(v) for v in e["picks"][0]]
            u, v = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            aim = (u, v, e["z"][0], b)
            dr.rectangle(b, outline=(40, 120, 255), width=5)
            dr.line([u - 14, v, u + 14, v], fill=(255, 0, 0), width=3)
            dr.line([u, v - 14, u, v + 14], fill=(255, 0, 0), width=3)
            dr.rectangle([6, 6, 360, 42], fill=(120, 170, 255))
            dr.text((12, 8), f"시설 관측: {e['label']} {e['z'][0]:.1f} m", font=font(22), fill=(0, 0, 0))
        # 지도
        m = base_map.copy()
        md = ImageDraw.Draw(m)
        tr = trail[-20:]                          # 최근 20초만, 5 m 넘게 튄 구간은 잇지 않음
        for i in range(1, len(tr)):
            if np.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1]) / a.ppm <= 5.0:
                md.line([mp(*tr[i - 1]), mp(*tr[i])], fill=(230, 60, 60), width=3)
        if s is not None:
            smax = float(s["cands"][0]["score"])
            for c in reversed(s["cands"]):
                x, y = mp(float(c["x_px"]), float(c["y_px"]))
                sc_ = float(c["score"])
                r = 4 + 16 * max(0.0, 1 - (smax - sc_) / 3.0)
                col = (230, 30, 30) if c is s["cands"][0] else ((255, 150, 0) if sc_ >= smax - 1.5 else (130, 130, 130))
                md.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3 if c is s["cands"][0] else 2)
        # 글 칸
        pn = Image.new("RGB", (W - 640, 480), (246, 246, 246))
        pd = ImageDraw.Draw(pn)
        y = 14

        def line(txt, sz=21, col=(0, 0, 0), gap=9, bg=None):
            nonlocal y
            if bg:
                pd.rectangle([12, y - 3, W - 640 - 12, y + sz + 8], fill=bg)
            pd.text((20, y), txt, font=font(sz), fill=col)
            y += sz + gap

        line(f"모호함 → 이벤트 → 조준 · {a.floor} 전체 주행", 27, gap=4)
        line(f"시각 {t:6.1f}초   (위치추정·이벤트 판단 1초마다, {a.speed:g}배속)", 18, (90, 90, 90), 16)
        if s is None:
            line("① 위치추정: 시작 전", 21, (130, 130, 130), 14)
        else:
            g = s["g"]
            why = [w for w, f in (("점수 차 낮음", g["amb"]), ("오돔과 안 맞게 튐", g["jump"] and g["spread"] >= 5)) if f]
            line(f"① 위치추정: {'모호함 — ' + ' + '.join(why) if s['on'] else '확신'}  (점수 차 {g['margin']:.2f}, 기준 1.5)",
                 21, gap=6, bg=(255, 215, 150) if s["on"] else (200, 235, 200))
            if s["jump"]:
                d_top, d_od = s["jump"]
                bad = d_top - d_od > 3.0
                note = "못 믿음" if bad and g["spread"] >= 5 else ("같은 봉우리 안 흔들림(후보 퍼짐 < 5 m)" if bad else "맞음")
                line(f"    튐 확인: 1위 {d_top:5.1f} m 이동 / 오돔 {d_od:4.1f} m → {note}", 17,
                     (200, 0, 0) if bad and g["spread"] >= 5 else (110, 110, 110), 12)
        if stop and stop[1] == "번호판":
            line("② 이벤트: 번호판 멈춤 — 표지 같은 작은 후보(폭 0.85 m 이하)", 21, gap=6, bg=(255, 225, 150))
        elif stop and stop[1] == "시설":
            e = fac[stop[0]]
            line(f"② 이벤트: 시설 관측 — {e['label']} (VLM, 멈추지 않음)", 21, gap=6, bg=(200, 215, 255))
            line(f"    VLM: {e['answer'][:60]}", 16, (90, 90, 90), 8)
        elif s is not None and s["on"]:
            line("② 이벤트: 없음 (걷는 중)", 21, (90, 90, 90), 14)
        else:
            line("② 이벤트: — (위치 확신)", 21, (130, 130, 130), 14)
        if aim:
            u, v, z, b = aim
            line(f"③ 조준: (u, v) = ({u:.0f}, {v:.0f}), 깊이 {z:.1f} m, 구간 {b[2] - b[0]:.0f}×{b[3] - b[1]:.0f} px", 21, gap=14,
                 bg=(255, 225, 150) if stop[1] == "번호판" else (200, 215, 255))
        else:
            line("③ 조준: —", 21, (130, 130, 130), 14)
        line(f"누적: 번호판 멈춤 {counts['번호판']}번 · 시설 관측 {counts['시설']}번", 20, gap=16)
        line("지도: 빨간 원 1위 후보, 주황 비슷한 후보(점수 1.5 안), 빨간 선 최근 20초 1위 발자취", 16, gap=5)
        line("영상: 노란 상자 번호판 · 파란 상자 시설(계단·엘리베이터·화장실), 빨간 십자 조준점", 16, gap=5)
        line("팬틸트 구동·판독은 로봇에서. 정답 위치는 안 그림", 16, (130, 130, 130))
        fr = Image.new("RGB", (W, H), (255, 255, 255))
        fr.paste(rgb, (0, 0))
        fr.paste(pn, (640, 0))
        fr.paste(m, (0, 480))
        return cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR)

    si = 0
    t_lo = loc_t[0]
    for i in range(int(np.searchsorted(ts, t_lo)), len(rows), stride):
        t = ts[i]
        if t > loc_t[-1] + 1:
            break
        while si < len(stops) and t >= stops[si][0] - 1e-6:
            stt, kind = stops[si]
            counts[kind] += 1
            j = int(np.argmin(np.abs(ts - stt)))
            f = frame(stt, cv2.imread(str(Path(a.root) / rows[j]["filename"])), stops[si])
            for _ in range(int(a.hold * fps)):
                vw.write(f)
            si += 1
        vw.write(frame(t, cv2.imread(str(Path(a.root) / rows[i]["filename"]))))
    vw.release()
    print(a.out, f"번호판 멈춤 {counts['번호판']}번, 시설 관측 {counts['시설']}번")


if __name__ == "__main__":
    main()
