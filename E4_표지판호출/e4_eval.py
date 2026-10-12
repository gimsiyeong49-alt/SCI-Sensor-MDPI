#!/usr/bin/env python3
"""E4 호출 정책 비교 채점(2026-10-07, 실험 지침서 14.5~14.6). 파일럿(개발 데이터) 결과다.
정책: P_ALL(게이트 없음), P_AMBIGUITY(장소 애매 + 센서 정상), P_INFORMATIVE_b(P_AMBIGUITY 멈춤 중 정보량 ≥ b bit, e4_informative.py).
층마다:
  요청 수: 표지판 멈춤 + 계단 관측
  표지판 정밀도: 사람 판정(25_판정, 판정된 4개 층) — score_labeled_stops.py 와 같은 대응
  4층 재현율: 정답 구간 번호판 27개 중 멈춤 상자에 든 것
  호출 효용: 그 정책의 이벤트만으로 '번호를 읽었다면'(obs_feedback.py --oracle-numbers, 거리 허용 0.08/m, 모순 1) 위치추정이
            얼마나 나아지나 — 1위 3 m 안 비율, 확신하며 틀린 비율, 애매 비율(채점용 정답 36_)
    python3 scripts/e4_eval.py --out <59_E4_호출정책>
"""
import argparse, csv, json, re, subprocess, sys
from pathlib import Path
import numpy as np

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
D = Path("/data/kci_work/maps_perfloor_20261001")
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
J = R / "25_판정"
GT4 = {json.loads(l)["t"]: json.loads(l) for f in ("1_정답/gt_marks_148_178_1s_objid.jsonl", "17_평가구간_228_258/gt_marks_228_258_1s_objid.jsonl")
       for l in open(Path("reports/_00_SIGN_PIPELINE_20260920") / f)}
inside = lambda g, b: b[0] <= (g[0] + g[2]) / 2 <= b[2] and b[1] <= (g[1] + g[3]) / 2 <= b[3]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--bits", default="0.5,1.0")
    ap.add_argument("--fb-name", default="feedback_oracle", help="층마다 obs_feedback 결과 폴더 이름")
    ap.add_argument("--fb-args", default="", help="obs_feedback 에 더 줄 인자(예: --dedup-num)")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙임")
    ap.add_argument("--extra", default="", help="이미 만든 정책 폴더 이름(쉼표). <out>/<이름>/<층>/run 이 있어야 함")
    a = ap.parse_args(); out = Path(a.out)
    Mj = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json"))
    pols = {"P_ALL": out / "P_ALL", "P_AMBIGUITY": out / "P_AMBIGUITY"}
    # P_INFORMATIVE: P_AMBIGUITY 이벤트에 정보량을 매겨 거른 run 폴더를 만든다
    for b in map(float, a.bits.split(",")):
        name = f"P_INFORMATIVE_{b:g}bit"; pols[name] = out / name
        for F in FL:
            src = out / "P_AMBIGUITY" / F / "run"; dst = out / name / F / "run"; dst.mkdir(parents=True, exist_ok=True)
            inf = out / "P_AMBIGUITY" / F / "informative.json"
            if not inf.exists():
                subprocess.run([sys.executable, "scripts/e4_informative.py", "--loc", str(R / "35_위치추정_축척추정" / F / "loc"), "--run", str(src),
                                "--sem", str(D / F / f"{F}_semantic_map.json"), "--base", str(D / F / "base_map.png"), "--out", str(inf)], check=True,
                               stdout=subprocess.DEVNULL)
            info = json.load(open(inf))
            keep = {(round(x["t"], 3), tuple(int(v) for v in x["picks"][0][:4])) for x in info if x["info_bits"] >= b}
            E = [e for e in json.load(open(src / "events_live_replay.json")) if (round(float(e["t"]), 3), tuple(int(v) for v in e["picks"][0][:4])) in keep]
            Fe = [e for e in json.load(open(src / "facility_events.json")) if (round(float(e["t"]), 3), tuple(int(v) for v in e["picks"][0][:4])) in keep]
            json.dump(E, open(dst / "events_live_replay.json", "w")); json.dump(Fe, open(dst / "facility_events.json", "w"))
    for name in filter(None, a.extra.split(",")):
        pols[name] = out / name
    rows = []
    for name, base in pols.items():
        tot = dict(sign=0, stair=0, tp=0, jn=0, un=0, hit4=0, sec=0, ok3=0, wc=0, amb=0)
        for F in FL:
            run = base / F / "run"
            E = json.load(open(run / "events_live_replay.json")); Fe = json.load(open(run / "facility_events.json"))
            tot["sign"] += len(E); tot["stair"] += sum(1 for e in Fe if e.get("label") == "계단")
            if F in Dj and not F.startswith("_"):
                P = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", [])); jd = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1)))
                for e in E:
                    m = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], e["picks"][0][:4]) > 0.3]
                    if not m or m[0]["id"] not in jd: tot["un"] += 1; continue
                    tot["jn"] += 1; tot["tp"] += m[0]["id"] in P
            if F == "4F":
                hit = set()
                for e in E:
                    g = GT4.get(round(float(e["t"])))
                    if g:
                        hit |= {o for (c, *bb), o in zip(g["items"], g["object_id"]) if c == "plaque" and inside(bb, e["picks"][0])}
                tot["hit4"] = len(hit)
            fb = base / F / a.fb_name
            if not (fb / "timeline.csv").exists():
                subprocess.run([sys.executable, "scripts/obs_feedback.py", "--loc", str(R / "35_위치추정_축척추정" / F / "loc"), "--run", str(run),
                                "--sem", str(D / F / f"{F}_semantic_map.json"), "--base", str(D / F / "base_map.png"), "--out", str(fb),
                                "--use", "facility,plaque", "--fac-kinds", "계단", "--oracle-numbers", str(R / "36_채점용정답" / F),
                                "--tol-per-m", "0.08", "--max-contra", "1"] + a.fb_args.split(), check=True, stdout=subprocess.DEVNULL)
            s = subprocess.run([sys.executable, "scripts/score_loc_pseudo_gt.py", str(R / "36_채점용정답" / F), f"x={fb}"], capture_output=True, text=True).stdout
            m = re.search(r"(\d+)초 \| 1위 중앙 [\d.]+ m, 3 m 안 (\d+)%.*애매 (\d+)%.*확신하며 틀림 = 전체의 (\d+)%", s)
            if m:
                n = int(m.group(1)); tot["sec"] += n; tot["ok3"] += n * int(m.group(2)) / 100; tot["amb"] += n * int(m.group(3)) / 100
                tot["wc"] += n * int(m.group(4)) / 100
        rows.append((name, tot))
    # 기준: 이벤트 반영 없음(위치추정 그대로)
    b = dict(sign=0, stair=0, tp=0, jn=0, un=0, hit4=0, sec=0, ok3=0, wc=0, amb=0)
    for F in FL:
        s = subprocess.run([sys.executable, "scripts/score_loc_pseudo_gt.py", str(R / "36_채점용정답" / F), f"x={R / '35_위치추정_축척추정' / F / 'loc'}"],
                           capture_output=True, text=True).stdout
        m = re.search(r"(\d+)초 \| 1위 중앙 [\d.]+ m, 3 m 안 (\d+)%.*애매 (\d+)%.*확신하며 틀림 = 전체의 (\d+)%", s)
        if m:
            k = int(m.group(1)); b["sec"] += k; b["ok3"] += k * int(m.group(2)) / 100; b["amb"] += k * int(m.group(3)) / 100; b["wc"] += k * int(m.group(4)) / 100
    rows.insert(0, ("(이벤트 반영 없음)", b))
    lines = [f"{'정책':22s} | 요청(표지판+계단) | 표지판 정밀도(판정 층) | 미판정 | 4층 번호판(27) | 번호 읽었다면: 1위 3m안 / 확신틀림 / 애매"]
    for name, t in rows:
        lines.append(f"{name:22s} | {t['sign'] + t['stair']:4d} ({t['sign']}+{t['stair']}) | {100 * t['tp'] / max(t['jn'], 1):5.1f}% ({t['tp']}/{t['jn']}) | {t['un']:3d} | "
                     f"{t['hit4']:2d} | {100 * t['ok3'] / max(t['sec'], 1):.1f}% / {100 * t['wc'] / max(t['sec'], 1):.1f}% / {100 * t['amb'] / max(t['sec'], 1):.1f}%")
    txt = "\n".join(lines); print(txt); (out / f"E4_정책비교{a.tag}.txt").write_text(txt + "\n")
    with open(out / f"E4_trigger_policy_metrics{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["experiment_id", "condition", "requests_sign", "requests_stair", "sign_precision_judged", "judged_n", "unjudged_n",
                                       "recall_4F_plaques_of_27", "oracle_top1_within3m", "oracle_wrong_confident", "oracle_ambiguous", "split"])
        for name, t in rows:
            w.writerow(["E4", name, t["sign"], t["stair"], round(t["tp"] / max(t["jn"], 1), 4), t["jn"], t["un"], t["hit4"],
                        round(t["ok3"] / max(t["sec"], 1), 4), round(t["wc"] / max(t["sec"], 1), 4), round(t["amb"] / max(t["sec"], 1), 4), "PILOT"])


if __name__ == "__main__":
    main()
