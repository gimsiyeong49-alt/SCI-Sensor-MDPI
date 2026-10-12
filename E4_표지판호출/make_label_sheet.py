#!/usr/bin/env python3
"""미판정 멈춤(score_labeled_stops.py 가 쓴 미판정_멈춤.json)을 사람 판정용 사진표로 만들고 판정목록에 새 번호로 덧붙인다(2026-10-05).
타일: 원본 화면 + 초록 상자 + 상자 주변 확대(구석) + '층 번호'. 한 장 4열 × 6줄.
판정목록.json 은 덧붙이기만 한다(기존 항목·번호 그대로, 실행 전 .bak 복사).

    python3 scripts/make_label_sheet.py 미판정_멈춤.json --run 새게이트_축척추정 --prefix 25_판정/추가판정D
2026-10-06: 이미 판정목록에 번호가 있는 항목(판정 대기)은 그 번호로 다시 싣는다.
"""
import argparse
import csv
import json
import shutil
import time

import cv2
import numpy as np

J = "reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/25_판정/판정목록.json"


def tile(F, it):
    root = "/data2/kci_datasets/4F_RGB_FULL_V0_1/" if F == "4F" else f"/data3/kci_datasets/{F}_RGB_FULL_5HZ_V0_1/"   # 4층은 다른 녹화 폴더(헤더 시각 기준, 2026-10-12)
    rows = list(csv.DictReader(open(root + "index.csv")))
    tb = np.array([float(r["header_relative_sec" if F == "4F" else "bag_relative_sec"]) for r in rows])
    im = cv2.imread(root + rows[int(np.argmin(abs(tb - it["t"])))]["filename"])
    x0, y0, x1, y1 = [int(v) for v in it["box"]]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    s = max(x1 - x0, y1 - y0, 40) * 3
    p0, q0 = min(max(0, cx - s // 2), max(0, 640 - s)), min(max(0, cy - s // 2), max(0, 480 - s))
    zoom = cv2.resize(im[q0:q0 + s, p0:p0 + s], (200, 200))
    cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2)
    t = cv2.resize(im, (480, 360))
    zx = 480 - 200 if cx < 320 else 0                     # 상자 반대쪽 아래 구석
    t[160:360, zx:zx + 200] = zoom
    cv2.rectangle(t, (zx, 160), (zx + 199, 359), (255, 255, 255), 2)
    cv2.rectangle(t, (0, 0), (170, 30), (0, 0, 0), -1)
    cv2.putText(t, f"{F} {it['id']}", (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unlabeled")
    ap.add_argument("--run", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    U = json.load(open(a.unlabeled))
    M = json.load(open(J))
    shutil.copy(J, J + time.strftime(".bak_%Y%m%d_%H%M%S"))
    tiles = []
    for F, items in U.items():
        nxt = max([x["id"] for x in M.get(F, [])] + [0]) + 1
        for it in sorted(items, key=lambda x: x["t"]):
            old = [x for x in M.get(F, []) if it.get("id") is not None and x["id"] == it["id"]]
            if old:                              # 2026-10-06: 이미 목록에 있고 판정만 안 된 것은 그 번호를 그대로 쓴다(새 번호로 또 붙이지 않음)
                rec = old[0]
                if a.run not in rec.setdefault("runs", []):
                    rec["runs"].append(a.run)
            else:
                rec = dict(t=it["t"], box=it["box"], z=it.get("z"), runs=[a.run], id=nxt)
                M.setdefault(F, []).append(rec)
                nxt += 1
            tiles.append(tile(F, rec))
    json.dump(M, open(J, "w"), ensure_ascii=False, indent=1)
    per = 24
    for k in range(0, len(tiles), per):
        page = tiles[k:k + per]
        page += [np.full_like(tiles[0], 255)] * (-len(page) % 4)
        img = np.vstack([np.hstack(page[i:i + 4]) for i in range(0, len(page), 4)])
        cv2.imwrite(f"{a.prefix}_{k // per + 1}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(tiles)}개 → {a.prefix}_1..{(len(tiles) - 1) // per + 1}.jpg, 판정목록에 덧붙임: " +
          ", ".join(f"{F} {min(x['id'] for x in M[F] if a.run in x['runs'])}~{max(x['id'] for x in M[F] if a.run in x['runs'])}"
                    for F in U if U[F]))


if __name__ == "__main__":
    main()
