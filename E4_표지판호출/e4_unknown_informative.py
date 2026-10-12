#!/usr/bin/env python3
"""층 모름 E4 P_INFORMATIVE(2026-10-11): 출발점 구간 실행(P_AMBIGUITY)의 멈춤마다 '읽으면 남은 가설을 가를 수 있나'를 판단한다.
층 아는 판 scripts/e4_informative.py 와 같은 규칙을 건물 전체 층 가설로 넓혔다. 판독 글자·정답은 쓰지 않는다.
  가설 = 그 장면 시각까지 쓸 수 있게 된(avail) 마지막 연결판 계산의 애매 가설(e4_unknown_gate.py 의 <출발점>_가설.json: 1위 merit + Δ 안, 이동 직후면 직전 1위 포함)
  멈춤 점(오돔 xy)을 가설마다 그 층 지도로 옮김: 지도점 = 가설 (x, y) + 축척 × A(φ) × (멈춤 xy − 계산 시각 오돔 xy), φ = hd − 228°
    (e3_seeded.py project 와 같은 식)
  호실 표지판 답 = (층, 가장 가까운 호실 덩어리) — 호실 경계 1.0 m + 거리 비례 0.08/m 안, 복도 1.5 m 안일 때. 아니면 0
  계단 답 = 지도점이 계단 칸 2.0 m + 0.2·거리 + 거리 비례 안인지
  가설 2개 이상이고 답이 둘 이상으로 갈리면 informative(호실은 적어도 하나가 호실을 가리켜야). 층마다 호실 덩어리는 협업자 패키지의
  의미지도 마스크(maps/<건물>_semmaps/<층>/sem/01_fused_room_mask.png, *stair_mask.png, *corridor_mask.png, 읽기만)에서 뽑는다.
    python3 scripts/e4_unknown_informative.py --gate-dir /data2/e4_unknown_20261011/gate_d0.2 --runs <70_층모름E4/P_AMBIGUITY> --out <폴더>
"""
import argparse, csv, glob, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

JM = Path("/data/kci_work/floor_id_pkg/floor_id/maps")
TASKS = Path("reports/_01_E2_초기위치/_작업자료/tasks/E2_tasks.csv")
INP = Path("/data2/loc_inputs_20261005")
CX, FX, YAW_OFF = 310.49, 606.80, math.radians(228.0)


def A(phi):
    """rt_localizer._A 와 같음(지도 픽셀 y 아래 방향이라 뒤집힘 포함)"""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


class FloorMasks:
    def __init__(self, building, m):
        d = JM / f"{building}_semmaps" / f"{m}F" / "sem"
        room = cv2.imread(str(d / "01_fused_room_mask.png"), 0) > 127
        self.H, self.W = room.shape
        n, lab = cv2.connectedComponents(room.astype(np.uint8))
        edge = room & cv2.dilate((~room).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.DT, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
        self.room_id = lab[iy, ix]
        co = glob.glob(str(d / "*corridor_mask.png")); st = glob.glob(str(d / "*stair_mask.png"))
        corr = cv2.imread(co[0], 0) > 127 if co else np.zeros_like(room)
        self.DTc = distance_transform_edt(~corr) if corr.any() else np.zeros(room.shape)
        stair = cv2.imread(st[0], 0) > 127 if st else np.zeros_like(room)
        self.DTs = distance_transform_edt(~stair) if stair.any() else None

    def look(self, img, p, cval):
        x, y = int(round(p[0])), int(round(p[1]))
        return img[y, x] if 0 <= x < self.W and 0 <= y < self.H else cval


def odom(F):
    d = Path("/data/kci_work/a131_full_v0_4") if F == "4F" else INP / F      # 위치추정(연결판)과 같은 오돔
    rows = list(csv.DictReader(open(d / "odom_link.csv")))
    T = np.array([float(r["t_sec"]) for r in rows])
    return T - T[0], np.c_[[float(r["fastlio_x"]) for r in rows], [float(r["fastlio_y"]) for r in rows]], np.array([float(r["fastlio_yaw_rad"]) for r in rows])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gate-dir", required=True); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=""); ap.add_argument("--seg", default="/data2/e4_unknown_20261011", help="segments.csv 폴더(시각 off)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    OFF = {r["task_id"]: float(r["off"]) for r in csv.DictReader(open(Path(a.seg) / "segments.csv"))}
    tasks = [t for t in csv.DictReader(open(TASKS)) if not a.tasks or t["task_id"] in a.tasks.split(",")]
    cache = {}; rows_all = []
    for t in tasks:
        k, F, b, ts = t["task_id"], t["floor"], t["building"], float(t["start_sec"])
        run = Path(a.runs) / k / "run"
        if not (run / "events_live_replay.json").exists():
            continue
        off = OFF[k]                                                  # 장면 시각 = 오돔 첫 샘플 기준 시각 + off
        hyp = json.load(open(Path(a.gate_dir) / f"{k}_가설.json")); ht = np.array([h["avail"] for h in hyp])
        OT, OXY, OYAW = odom(F)
        o_at = lambda tt: OXY[int(np.argmin(np.abs(OT - tt)))]
        ev = [("번호판", e, float(e["t"]), np.array(e["xyz"][:2], float), (e.get("z") or [None])[0]) for e in json.load(open(run / "events_live_replay.json")) if e.get("xyz")]
        fp = run / "facility_events.json"
        for e in (json.load(open(fp)) if fp.exists() else []):
            if e.get("label") != "계단" or not e.get("z"):
                continue
            bx = e["picks"][0]; z = float(e["z"][0]); tt = float(e["t"]); i = int(np.argmin(np.abs(OT - (tt - off))))
            fwd, left = z, -(((bx[0] + bx[2]) / 2 - CX) / FX * z); c, s = math.cos(OYAW[i] + YAW_OFF), math.sin(OYAW[i] + YAW_OFF)
            ev.append(("계단", e, tt, OXY[i] + np.array([c * fwd - s * left, s * fwd + c * left]), z))
        res = []
        for kind, e, tt, xy, z in ev:
            j = np.where(ht <= tt + 1e-6)[0]
            if not len(j):                                            # 첫 계산 전(출발 직후 3초): 가설 목록 없음 → 판단 불가, 부르지 않음
                res.append(dict(t=tt, kind=kind, n_near=0, answers=[], informative=False, why="첫 계산 전")); continue
            h = hyp[int(j[-1])]; o_now = o_at(ts + h["T"]); grow = 0.08 * float(np.linalg.norm(xy - o_now)); ans = []
            for c in h["near"]:
                key = (b, c["m"])
                if key not in cache:
                    cache[key] = FloorMasks(b, c["m"])
                M = cache[key]; s = c["s"]
                mp = np.array([c["x"], c["y"]]) + s * (A(math.radians(c["hd"]) - YAW_OFF) @ (xy - o_now))
                if kind == "번호판":
                    ok = M.look(M.DT, mp, 1e9) / s <= 1.0 + grow and M.look(M.DTc, mp, 1e9) / s <= 1.5 + grow
                    ans.append(f"{c['m']}-{int(M.look(M.room_id, mp, 0))}" if ok else "0")
                else:
                    ans.append(str(bool(M.DTs is not None and M.look(M.DTs, mp, 1e9) / s <= 2.0 + 0.2 * (z or 0) + grow)))
            d = set(ans)
            info = len(ans) >= 2 and len(d) >= 2 and (kind != "번호판" or any(v != "0" for v in d))
            res.append(dict(t=tt, kind=kind, n_near=len(ans), floors=sorted({c["m"] for c in h["near"]}), answers=ans, informative=bool(info),
                            picks=e["picks"][:1]))
        json.dump(res, open(out / f"{k}.json", "w"), ensure_ascii=False)
        rows_all += [(k, r["kind"], r["informative"]) for r in res]
    n = len(rows_all)
    for kind in ("번호판", "계단"):
        r = [x for x in rows_all if x[1] == kind]
        print(f"{kind}: 멈춤 {len(r)}개 중 informative {sum(x[2] for x in r)}")
    print(f"전체 {n}개")


if __name__ == "__main__":
    main()
