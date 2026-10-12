#!/usr/bin/env python3
"""E4 논문용 자료 정리(2026-10-07, 실험 지침서 14절·19절, PILOT). 59_E4_호출정책/paper/ 아래에 만든다.
  csv/E4_trigger_decisions.csv  매 초(층별) 불확실성 분류와 정책별 호출 허용
  csv/E4_detections.csv         이벤트(표지판 멈춤·계단 관측)마다 정책·정보량·사람 판정·그 순간 분류
  csv/E4_event_annotations.csv  정답 주석: 4층 프레임 단위 표시 + 사람 판정(멈춘 상자)
  tables/E4_uncertainty_classes.md  분류별 시간 비율, 1위 3 m 안 비율(채점용 정답), 멈춤 수
불확실성 분류(우선순위 순):
  SENSOR_UNRELIABLE      오돔 순간 속도 > 2 m/s 또는 회전 > 120°/s(make_gate_json sensor_bad)
  GEOMETRIC_INSUFFICIENT 장소 애매 + 최근 3초 a1_3_1 벽 방향 다양성 중앙 < 0.1(벽이 거의 한 방향, 긴 직선 복도)
  GEOMETRIC_AMBIGUITY    장소 애매(1위와 5 m 밖 최고 후보 점수 차 ≤ 1.5)
  GEOMETRIC_CONFIDENT    나머지
    python3 scripts/e4_paper.py
"""
import csv, json, math
import os
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)
from pathlib import Path
import numpy as np

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper")
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
DIV_TH = 0.10
J = R / "25_판정"
GTMARK = [Path("reports/_00_SIGN_PIPELINE_20260920/1_정답/gt_marks_148_178_1s_objid.jsonl"),
          Path("reports/_00_SIGN_PIPELINE_20260920/17_평가구간_228_258/gt_marks_228_258_1s_objid.jsonl")]


def A(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, -s], [-s, -c]])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def classify(g, div):
    if g.get("sensor_bad"):
        return "SENSOR_UNRELIABLE"
    if g["amb"]:
        return "GEOMETRIC_INSUFFICIENT" if (div is not None and div < DIV_TH) else "GEOMETRIC_AMBIGUITY"
    return "GEOMETRIC_CONFIDENT"


def main():
    for d in ("csv", "tables", "figures"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    Mj = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json"))
    trig, dets, cls_stat = [], [], {}
    for F in FL:
        trial = f"PILOT_{F}"
        gate = {round(x["t"], 3): x for x in json.load(open(R / "35_위치추정_축척추정" / F / "모호함_v2.json"))}
        gc = list(csv.DictReader(open(OUT / "csv" / f"geom_cells_{F}.csv")))
        gt_t = np.array([float(r["relative_sec"]) for r in gc]); gdv = np.array([float(r["normal_diversity"]) for r in gc])
        # 채점용 정답으로 매 초 1위 맞음(3 m)
        GD = R / "36_채점용정답" / F; S = json.load(open(GD / "summary.json")); ppm = S["ppm"]
        g = list(csv.DictReader(open(GD / "warp_clean" / "gt_warp.csv")))
        gtt = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
        top = {}
        for r in csv.DictReader(open(R / "35_위치추정_축척추정" / F / "loc" / "topk_series.csv")):
            if r["rank"] == "1":
                top[round(float(r["relative_sec"]), 3)] = (float(r["x_px"]), float(r["y_px"]))
        cls_at = {}
        for t, x in sorted(gate.items()):
            m = (gt_t > t - 3.0) & (gt_t <= t + 1e-6) & np.isfinite(gdv)
            div = float(np.median(gdv[m])) if m.any() else None
            c = classify(x, div); cls_at[t] = c
            ok = None
            if t in top:
                k = int(np.argmin(abs(gtt - t))); ok = math.hypot(top[t][0] - gxy[k][0], top[t][1] - gxy[k][1]) / ppm <= 3.0
            trig.append(dict(trial_id=trial, relative_sec=t, uncertainty_class=c, margin=round(x["margin"], 4), spread_m=round(x["spread"], 2),
                             n_near=x["npk"], normal_diversity="" if div is None else round(div, 4), sensor_bad=int(bool(x.get("sensor_bad"))),
                             allow_P_ALL=1, allow_P_AMBIGUITY=int(x["amb"] and not x.get("sensor_bad")), top1_within_3m_gt="" if ok is None else int(ok)))
            st = cls_stat.setdefault(c, dict(sec=0, ok=0, n_ok=0))
            st["sec"] += 1
            if ok is not None:
                st["n_ok"] += 1; st["ok"] += ok
        # 이벤트
        info = {(round(e["t"], 3), tuple(int(v) for v in e["picks"][0][:4])): e for e in json.load(open(P / MAIN / F / "informative.json"))}
        P_ALLE = json.load(open(P / "P_ALL" / F / "run" / "events_live_replay.json"))
        P_AMBE = json.load(open(P / MAIN / F / "run" / "events_live_replay.json"))
        amb_keys = {(round(float(e["t"]), 3), tuple(int(v) for v in e["picks"][0][:4])) for e in P_AMBE}
        P_set = set(Dj.get(F, {}).get("plaque", [])) | set(Dj.get(F, {}).get("sign", [])) if F in Dj else set()
        judged = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1))) if F in Dj else set()
        ts = np.array(sorted(cls_at))
        for pol, E in (("P_ALL", P_ALLE), (MAIN, P_AMBE)):
            for i, e in enumerate(E):
                b = e["picks"][0]; key = (round(float(e["t"]), 3), tuple(int(v) for v in b[:4]))
                lab = ""
                if F in Dj:
                    mm = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], b[:4]) > 0.3]
                    lab = ("TP" if mm[0]["id"] in P_set else "FP") if (mm and mm[0]["id"] in judged) else "UNJUDGED"
                k = float(ts[np.argmin(abs(ts - float(e["t"])))]) if len(ts) else None
                inf = info.get(key) if pol == MAIN else None
                dets.append(dict(trial_id=f"PILOT_{F}", run_id=pol, event_id=f"{F}_{pol}_{i:04d}", relative_sec=float(e["t"]),
                                 bbox_x=round(b[0], 1), bbox_y=round(b[1], 1), bbox_w=round(b[2] - b[0], 1), bbox_h=round(b[3] - b[1], 1),
                                 predicted_class="room_sign", siglip_score=round(float(e.get("cur_sim", float("nan"))), 4),
                                 depth_m=round(float((e.get("z") or [float("nan")])[0] or float("nan")), 2),
                                 uncertainty_class=cls_at.get(k, ""), info_bits="" if inf is None else round(inf["info_bits"], 3),
                                 trigger_main=int(key in amb_keys), human_label=lab))
            fe = json.load(open((P / pol / F / "run" / "facility_events.json")))
            for i, e in enumerate(fe):
                b = e["picks"][0]; key = (round(float(e["t"]), 3), tuple(int(v) for v in b[:4]))
                k = float(ts[np.argmin(abs(ts - float(e["t"])))]) if len(ts) else None
                inf = info.get(key) if pol == MAIN else None
                dets.append(dict(trial_id=f"PILOT_{F}", run_id=pol, event_id=f"{F}_{pol}_S{i:03d}", relative_sec=float(e["t"]),
                                 bbox_x=round(b[0], 1), bbox_y=round(b[1], 1), bbox_w=round(b[2] - b[0], 1), bbox_h=round(b[3] - b[1], 1),
                                 predicted_class="stair", siglip_score="", depth_m=round(float(e["z"][0]), 2),
                                 uncertainty_class=cls_at.get(k, ""), info_bits="" if inf is None else round(inf["info_bits"], 3),
                                 trigger_main=int(pol == MAIN), human_label=""))
    for name, rows in (("E4_trigger_decisions.csv", trig), ("E4_detections.csv", dets)):
        with open(OUT / "csv" / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # 정답 주석
    ann = []
    for p in GTMARK:
        for l in open(p):
            d = json.loads(l)
            for (c, *b), o in zip(d["items"], d["object_id"]):
                ann.append(dict(trial_id="PILOT_4F", source="frame_gt", relative_sec=d["t"], class_label=c, bbox_x=b[0], bbox_y=b[1],
                                bbox_w=b[2] - b[0], bbox_h=b[3] - b[1], object_id="" if o is None else o, human_label=""))
    for F in Mj:
        if F.startswith("_") or F not in Dj:
            continue
        P_set = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", []))
        judged = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1)))
        for it in Mj[F]:
            if it["id"] in judged:
                b = it["box"]
                ann.append(dict(trial_id=f"PILOT_{F}", source="stop_judgment", relative_sec=it["t"], class_label="room_sign" if it["id"] in P_set else "not_sign",
                                bbox_x=round(b[0], 1), bbox_y=round(b[1], 1), bbox_w=round(b[2] - b[0], 1), bbox_h=round(b[3] - b[1], 1), object_id=it["id"],
                                human_label="TP" if it["id"] in P_set else "FP"))
    with open(OUT / "csv" / "E4_event_annotations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ann[0])); w.writeheader(); w.writerows(ann)
    # 분류 표
    tot = sum(v["sec"] for v in cls_stat.values())
    amb_det = [d for d in dets if d["run_id"] == MAIN and d["predicted_class"] == "room_sign"]
    lines = [f"| 분류 | 시간 비율 | 1위 3 m 안(채점용 정답) | {MAIN_NAME} 표지판 멈춤 | 그중 판정 TP/(TP+FP) |", "|---|---|---|---|---|"]
    for c in ("GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT", "SENSOR_UNRELIABLE"):
        v = cls_stat.get(c, dict(sec=0, ok=0, n_ok=0)); ds = [d for d in amb_det if d["uncertainty_class"] == c]
        tp = sum(d["human_label"] == "TP" for d in ds); fp = sum(d["human_label"] == "FP" for d in ds)
        lines.append(f"| {c} | {100 * v['sec'] / tot:.1f}% ({v['sec']}초) | {100 * v['ok'] / max(v['n_ok'], 1):.1f}% | {len(ds)} | "
                     f"{tp}/{tp + fp}" + (f" ({100 * tp / (tp + fp):.0f}%)" if tp + fp else "") + " |")
    txt = "\n".join(lines); (OUT / "tables" / "E4_uncertainty_classes.md").write_text(
        f"13개 층 PILOT, 총 {tot}초. 벽 방향 다양성 기준 {DIV_TH}.\n\n" + txt + "\n"); print(txt)
    print("csv:", len(trig), "초,", len(dets), "이벤트,", len(ann), "주석")


if __name__ == "__main__":
    main()
