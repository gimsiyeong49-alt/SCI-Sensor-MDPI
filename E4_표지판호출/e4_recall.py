#!/usr/bin/env python3
"""E4 놓친 표지판 비율(호실 단위 재현율, 2026-10-08, PILOT). 프레임 단위 정답이 4층 60초뿐이라 의미지도와 채점용 정답 경로로 자동으로 낸다.
  지나간 호실(분모): 호실 벽 중 복도에 붙은 부분(복도 칸에서 1 m 안)이 실제 경로에서 PASS_M 안으로 들어온 호실
  본 호실(분자): 멈춤 3D 점을 정답 자세로 지도에 찍어(obs_feedback --oracle-numbers 와 같은 식) 호실 벽 1.5 m 안이면 그 호실.
                판정한 층(25_판정)은 사람이 표지판이라고 한 멈춤만, 나머지 층은 모든 멈춤(상한)
  층별·전체: 지나간 호실 수, 본 호실 수, 재현율. 정책 P_ALL·P_AMBIGUITY·P_INFORMATIVE_0.5bit.
  4층 검산: 프레임 정답 구간(148~178, 228~258 s)만 따로.
    python3 scripts/e4_recall.py
결과: paper/tables/E4_recall.md, paper/csv/E4_recall.csv, paper/csv/E4_recall_rooms.csv(호실마다)
"""
import csv, json, math, os, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(Path(__file__).parent))
from obs_feedback import A, poly_mask  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper"); J = R / "25_판정"
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)

D = Path("/data/kci_work/maps_perfloor_20261001"); L = R / "35_위치추정_축척추정"
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
POLS = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["P_ALL", "P_AMBIGUITY", "P_INFORMATIVE_0.5bit"])
PASS_M, HIT_M, CORR_M = 3.0, 1.5, 1.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def floor_rooms(F):
    im = cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE); H, W = im.shape
    sem = json.load(open(D / F / f"{F}_semantic_map.json"))
    rooms = [s for s in sem["spaces"] if s.get("type") == "room" and s.get("polygon_px")]
    corr = poly_mask((H, W), [c.get("polygon_px") for c in sem.get("corridors", [])])
    DTc = distance_transform_edt(~corr) if corr.any() else np.zeros((H, W))
    out = []
    for r in rooms:
        m = poly_mask((H, W), [r["polygon_px"]]); e = m & cv2.dilate((~m).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ys, xs = np.where(e); out.append(dict(id=r["id"], num=r.get("room_number") or r["id"], edge=np.c_[xs, ys].astype(float), DTc=DTc[ys, xs]))
    return out, (H, W)


def main():
    Mj = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json"))
    rows, rrows = [], []
    tot = {p: [0, 0] for p in POLS}; tot_j = {p: [0, 0] for p in POLS}
    for F in FL:
        rooms, _ = floor_rooms(F)
        S = json.load(open(R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]
        g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gt = np.array([float(x["relative_sec"]) for x in g]); gxy = np.array([[float(x["x_px"]), float(x["y_px"])] for x in g])
        Mg = A(math.radians(S["theta_deg"])) * ppm if S["hand"] > 0 else None
        tl = list(csv.DictReader(open(L / F / "loc/timeline.csv"))); tt = np.array([float(x["relative_sec"]) for x in tl])
        od = np.array([[float(x["odom_x_m"]), float(x["odom_y_m"])] for x in tl])
        # 지나간 호실: 복도 쪽 벽(복도 칸 1 m 안)이 경로 PASS_M 안. 첫 통과 시각도 남긴다
        passed = {}
        for r in rooms:
            cw = r["edge"][r["DTc"] / ppm <= CORR_M]
            if not len(cw):
                continue
            step = max(1, len(cw) // 400); cw = cw[::step]
            d = np.min(np.linalg.norm(gxy[::5, None, :] - cw[None], axis=2), axis=1) / ppm
            k = np.where(d <= PASS_M)[0]
            if len(k):
                passed[r["id"]] = (r, float(gt[::5][k[np.argmin(d[k])]]))     # 가장 가까이 지난 시각
        stc = {float(x["relative_sec"]): x["uncertainty_class"] for x in csv.DictReader(open(OUT / "csv/E4_trigger_decisions.csv")) if x["trial_id"] == f"PILOT_{F}"}
        sk = np.array(sorted(stc))
        amb_pass = {rid for rid, (_, t0) in passed.items() if len(sk) and abs(sk[np.argmin(abs(sk - t0))] - t0) <= 1.0
                    and stc[sk[np.argmin(abs(sk - t0))]] in ("GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT")}
        judged = F in Dj
        if judged:
            Pset = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", [])); jd = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1)))
        res = {}
        for pol in POLS:
            seen = {}
            for e in json.load(open(P / pol / F / "run/events_live_replay.json")):
                if not e.get("xyz") or Mg is None:
                    continue
                if judged:
                    m = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], e["picks"][0][:4]) > 0.3]
                    if not (m and m[0]["id"] in jd and m[0]["id"] in Pset):
                        continue                                   # 판정 층: 표지판 맞은 멈춤만
                i = int(np.argmin(abs(gt - e["t"]))); oo = od[int(np.argmin(abs(tt - e["t"])))]
                mp = gxy[i] + Mg @ (np.array(e["xyz"][:2]) - oo)
                best = None
                for rid, (r, _) in passed.items():
                    dd = np.min(np.linalg.norm(r["edge"] - mp, axis=1)) / ppm
                    if dd <= HIT_M and (best is None or dd < best[0]):
                        best = (dd, rid)
                if best:
                    seen.setdefault(best[1], float(e["t"]))
            res[pol] = seen
            tot[pol][0] += len(seen); tot[pol][1] += len(passed)
            if judged:
                tot_j[pol][0] += len(seen); tot_j[pol][1] += len(passed)
        rows.append(dict(trial_id=f"PILOT_{F}", judged=int(judged), rooms_passed=len(passed), rooms_passed_amb=len(amb_pass),
                         rooms_seen_amb_main=len(set(res.get(MAIN, {})) & amb_pass),
                         **{f"rooms_seen_{p}": len(res[p]) for p in POLS}, **{f"recall_{p}": round(len(res[p]) / max(len(passed), 1), 3) for p in POLS}))
        for rid, (r, t0) in passed.items():
            rrows.append(dict(trial_id=f"PILOT_{F}", room_id=rid, room_number=r["num"], closest_pass_sec=round(t0, 1), passed_while_ambiguous=int(rid in amb_pass),
                              **{f"seen_{p}": int(rid in res[p]) for p in POLS}))
    # 4층 검산: 프레임 정답 구간 안만
    tag = "" if len(sys.argv) <= 1 else "_" + "_".join(POLS)
    for name, rows_ in ((f"E4_recall{tag}.csv", rows), (f"E4_recall_rooms{tag}.csv", rrows)):
        with open(OUT / "csv" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_[0])); w.writeheader(); w.writerows(rows_)
    NAME = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}
    Lm = [f"호실 단위 재현율 = 지나간 호실(복도 쪽 벽이 실제 경로 {PASS_M:g} m 안) 중 멈춰서 본 호실(멈춤 점이 호실 벽 {HIT_M:g} m 안) 비율. PILOT.",
          "판정 층(●)은 사람이 표지판이라고 한 멈춤만 셈. 나머지 층은 판정이 없어 모든 멈춤을 셈(틀린 멈춤이 호실 벽 근처면 본 것으로 쳐서 상한).", "",
          "마지막 열: 헷갈리는 동안 지나간 호실만 분모로(지금 규칙은 확실할 때 일부러 안 멈춘다). 소융대 1층은 의미지도에 호실이 4개뿐이라 뜻이 없다.", "",
          "| 층 | 지나간 호실 | " + " | ".join({"P_ALL": "다 멈춤", "P_AMBIGUITY": "A 헷갈릴 때만(10-08)", "P_R3B": "C 배포판(10-09)", "P_INFORMATIVE_0.5bit": "정보 될 때만"}.get(p, p) for p in POLS) +
          f" | {MAIN_NAME}, 헷갈리는 동안 지난 호실 중 |", "|---" * (len(POLS) + 3) + "|"]
    for r in rows:
        F = r["trial_id"][6:]
        Lm.append(f"| {NAME.get(F, F.replace('F', '층'))}{' ●' if r['judged'] else ''} | {r['rooms_passed']} | " +
                  " | ".join(f"{r[f'rooms_seen_{p}']} ({100 * r[f'recall_{p}']:.0f}%)" for p in POLS) +
                  f" | {r['rooms_seen_amb_main']}/{r['rooms_passed_amb']}" + (f" ({100 * r['rooms_seen_amb_main'] / r['rooms_passed_amb']:.0f}%)" if r['rooms_passed_amb'] else "") + " |")
    ja = [r for r in rows if r["judged"]]; sa, pa = sum(r["rooms_seen_amb_main"] for r in ja), sum(r["rooms_passed_amb"] for r in ja)
    Lm.append("| **판정 층 합계** | " + str(tot_j[POLS[0]][1]) + " | " + " | ".join(f"{tot_j[p][0]} ({100 * tot_j[p][0] / max(tot_j[p][1], 1):.0f}%)" for p in POLS) + f" | {sa}/{pa} ({100 * sa / max(pa, 1):.0f}%) |")
    sa, pa = sum(r["rooms_seen_amb_main"] for r in rows), sum(r["rooms_passed_amb"] for r in rows)
    Lm.append("| 13개 층 합계(상한 포함) | " + str(tot[POLS[0]][1]) + " | " + " | ".join(f"{tot[p][0]} ({100 * tot[p][0] / max(tot[p][1], 1):.0f}%)" for p in POLS) + f" | {sa}/{pa} ({100 * sa / max(pa, 1):.0f}%) |")
    tag = "" if len(sys.argv) <= 1 else "_" + "_".join(POLS)
    (OUT / f"tables/E4_recall{tag}.md").write_text("\n".join(Lm) + "\n"); print("\n".join(Lm))


if __name__ == "__main__":
    main()
