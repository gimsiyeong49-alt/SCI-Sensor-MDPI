#!/usr/bin/env python3
"""E4 추가 지표(2026-10-08, 실험 지침서 14.3·14.6·14.7, PILOT). 59_E4_호출정책/paper/ 아래에 만든다.
  tables/E4_rates.md, csv/E4_rates.csv      정책별 멈춤 빈도: 1분당, 10 m 당, 틀린 멈춤 1분당(판정 층), 같은 자리 재멈춤 비율
  tables/E4_confusion.md, figures/E4_fig_confusion   예측 클래스 × 실제(사람 판정·종류, 계단은 의미지도 자동 채점)
  tables/E4_prompts.md, csv/E4_prompts.csv  문구 실험: (가) 판정 사진 230장 문구별 구별력 (나) 4F·7F·G1F 실제 실행 결과(52_ 표)
    python3 scripts/e4_extra.py
"""
import csv, json, math, os, re
from collections import Counter
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import plt, save, INK, INK2, SURF  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper"); J = R / "25_판정"
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)

FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
POLS = os.environ.get("E4_POLS", "P_ALL,P_AMBIGUITY,P_INFORMATIVE_0.5bit,P_INFORMATIVE_1bit,P_INSUFF_WALK,P_INSUFF_NOCAM").split(",")
DUP_M = 1.0


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def auc_pr(S, Y):
    o = np.argsort(-S); tp = np.cumsum(Y[o]); prec = tp / np.arange(1, len(S) + 1); rec = tp / Y.sum()
    ap_ = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0])
    r = np.argsort(np.argsort(S)) + 1; roc = float((r[Y].sum() - Y.sum() * (Y.sum() + 1) / 2) / (Y.sum() * (~Y).sum()))
    return ap_, roc


def rates():
    Mj = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json"))
    sec, dist = {}, {}
    for F in FL:
        tl = list(csv.DictReader(open(R / "35_위치추정_축척추정" / F / "loc" / "timeline.csv")))
        xy = np.array([[float(r["odom_x_m"]), float(r["odom_y_m"])] for r in tl])
        sec[F] = len(tl); dist[F] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    judged_F = [F for F in FL if F in Dj]
    rows = []
    for pol in POLS:
        n = fp = tp = dup = 0
        for F in FL:
            E = json.load(open(P / pol / F / "run" / "events_live_replay.json")); n += len(E)
            seen = []
            for e in E:
                x = np.array(e["xyz"][:2]) if e.get("xyz") else None
                if x is not None and any(np.linalg.norm(x - s) <= DUP_M for s in seen):
                    dup += 1
                if x is not None:
                    seen.append(x)
            if F in Dj:
                Pset = set(Dj[F].get("plaque", [])) | set(Dj[F].get("sign", [])); jd = set(Dj[F].get("judged", range(1, Dj[F].get("judged_max", 0) + 1)))
                for e in E:
                    m = [it for it in Mj[F] if abs(it["t"] - e["t"]) <= 0.6 and iou(it["box"], e["picks"][0][:4]) > 0.3]
                    if m and m[0]["id"] in jd:
                        tp += m[0]["id"] in Pset; fp += m[0]["id"] not in Pset
        S = sum(sec.values()); Dm = sum(dist.values()); Sj = sum(sec[F] for F in judged_F)
        rows.append(dict(condition=pol, sign_stops=n, minutes=round(S / 60, 1), distance_m=round(Dm, 1), stops_per_min=round(n / (S / 60), 2),
                         stops_per_10m=round(10 * n / Dm, 2), judged_floor_minutes=round(Sj / 60, 1), false_stops_per_min_judged=round(fp / (Sj / 60), 2),
                         true_stops_per_min_judged=round(tp / (Sj / 60), 2), repeat_stop_rate=round(dup / max(n, 1), 3), split="PILOT"))
    with open(OUT / "csv/E4_rates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [f"13개 층 위치추정 구간 {rows[0]['minutes']}분, 오돔 이동 {rows[0]['distance_m']:.0f} m. 틀린/맞은 멈춤 1분당은 판정한 층({', '.join(judged_F)}, {rows[0]['judged_floor_minutes']}분)만, "
         f"미판정 멈춤은 빼서 하한값. 재멈춤 = 앞서 멈춘 표지판 3D 위치에서 {DUP_M} m 안에 또 멈춤.", "",
         "| 정책 | 표지판 멈춤 | 1분당 | 10 m 당 | 틀린 멈춤/분(판정 층) | 맞은 멈춤/분(판정 층) | 재멈춤 비율 |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['condition']} | {r['sign_stops']} | {r['stops_per_min']} | {r['stops_per_10m']} | {r['false_stops_per_min_judged']} | {r['true_stops_per_min_judged']} | "
          f"{100 * r['repeat_stop_rate']:.0f}% |" for r in rows]
    (OUT / "tables/E4_rates.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def confusion():
    D = [d for d in csv.DictReader(open(OUT / "csv/E4_detections.csv")) if d["run_id"] == MAIN]
    fpt = {r["event_id"]: r["fp_type"] for r in csv.DictReader(open(OUT / "csv/E4_fp_types.csv"))}
    sign = [d for d in D if d["predicted_class"] == "room_sign"]
    jf = set(json.load(open(J / "판정결과.json")))
    act = Counter()
    for d in sign:
        if d["human_label"] == "TP":
            act["room sign"] += 1
        elif d["human_label"] == "FP":
            act["other sign / lettering" if fpt.get(d["event_id"]) == "E" else "not a sign"] += 1
        elif d["trial_id"][6:] in jf:
            act["unjudged"] += 1
        else:
            act["floor not judged"] += 1
    st = []                                                      # 계단: 57_ 점검.jsonl(의미지도 계단 칸 자동 채점)
    for l in open(R / os.environ.get("E4_AUDIT", "57_바닥값기본_전층") / "점검.jsonl"):
        x = json.loads(l); st += [ok for (_, ok, _) in x["fac"].get("계단", [])]
    cols = ["room sign", "other sign / lettering", "not a sign", "stair", "unjudged", "floor not judged"]
    M = np.array([[act[c] for c in cols[:3]] + [0, act["unjudged"], act["floor not judged"]], [0, 0, sum(not s for s in st), sum(st), 0, 0]])
    L = [f"{MAIN_NAME}, 13개 층. 표지판 행은 사람 판정(판정한 4개 층) + 틀린 종류(사용자 확인), 계단 행은 의미지도 계단 칸 자동 채점(" + os.environ.get("E4_AUDIT", "57_바닥값기본_전층") + " 점검).",
         "놓친 대상(실제는 있는데 예측 없음) 열은 프레임 단위 정답이 4층 60초뿐이라 표에 넣지 않았다(4층: 번호판 27개 중 멈춰서 본 것 4개).", "",
         "| 예측 \\ 실제 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in zip(["room_sign", "stair"], M):
        L.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    (OUT / "tables/E4_confusion.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    fig, a = plt.subplots(figsize=(5.4, 2.0)); a.grid(False)
    frac = M[:, :4] / M[:, :4].sum(1, keepdims=True); M = M[:, :4]; cols = cols[:4]   # 그림은 판정된 것만
    a.imshow(frac, cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6"]), vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]}\n({100 * frac[i, j]:.0f}%)", ha="center", va="center", fontsize=8, color=SURF if frac[i, j] > 0.55 else INK)
    a.set_xticks(range(len(cols))); a.set_xticklabels(cols, fontsize=8); a.set_yticks([0, 1]); a.set_yticklabels(["predicted\nroom sign", "predicted\nstair"])
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title(f"E4 confusion, judged stops only (PILOT, {MAIN_NAME}; % of row)", fontsize=9.5, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_confusion")


def prompts():
    d = np.load(R / "52_문구섞기_3층/문구점수.npz", allow_pickle=True); S, Y, PH = d["S"], d["Y"].astype(bool), list(d["PH"])
    rows = [dict(prompt=p, kind="single", n=len(Y), n_pos=int(Y.sum()), **dict(zip(("pr_auc", "roc_auc"), auc_pr(S[:, i], Y)))) for i, p in enumerate(PH)]
    sets = {"original: small wall sign": ["a small sign mounted on a wall"],
            "room plate + near door": ["a small room name plate", "a small room name plate near a door"],
            "mix1": ["a small sign mounted on a wall", "a small room name plate"],
            "mix2": ["a small sign mounted on a wall", "a small room name plate", "a small sign sticking out from a wall"],
            "mix3": ["a small sign mounted on a wall", "a small room name plate", "a small sign sticking out from a wall", "a small room name plate near a door"],
            "FINAL: room plate + near door + perpendicular": ["a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall"]}
    for k, v in sets.items():
        ap_, roc = auc_pr(S[:, [PH.index(p) for p in v]].max(1), Y)
        rows.append(dict(prompt=k + " = max(" + " | ".join(v) + ")", kind="set", n=len(Y), n_pos=int(Y.sum()), pr_auc=ap_, roc_auc=roc))
    with open(OUT / "csv/E4_prompts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({**r, "pr_auc": round(r["pr_auc"], 4), "roc_auc": round(r["roc_auc"], 4)})
    L = [f"(가) 판정 사진 {len(Y)}장(표지판 {int(Y.sum())})에서 문구별 SigLIP 점수 구별력. 여럿이면 최댓값. 모델 siglip-base-patch16-224.", "",
         "| 문구 | PR-AUC | ROC-AUC |", "|---|---|---|"]
    for r in sorted([r for r in rows if r["kind"] == "set"], key=lambda r: -r["pr_auc"]) + sorted([r for r in rows if r["kind"] == "single"], key=lambda r: -r["pr_auc"]):
        L.append(f"| {r['prompt']} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} |")
    # (나) 실제 실행(52_ 표): 문구만 바꿔 같은 코드로 4F·7F·G1F 를 돌린 결과
    name = {"원래문구": "original: small wall sign", "지금": "room plate + near door", "섞기1": "mix1", "섞기2": "mix2", "섞기3": "mix3",
            "돌출+문근처": "perpendicular + near door", "호실+문근처+돌출": "FINAL prompts (height ≤2.2 m)", "새문구": "FINAL prompts (height ≤2.2 m)",
            "새문구+높이2.8": "FINAL prompts + height ≤2.8 m", "새문구+높이2.8+천장": "FINAL + height ≤2.8 m + ceiling filter (not adopted)"}
    seen, L2 = set(), []
    for t in ("표3.txt", "표2.txt", "표.txt"):
        for l in open(R / "52_문구섞기_3층" / t).read().splitlines()[1:]:
            c = [x.strip() for x in l.split("|")]
            if name.get(c[0], c[0]) in seen:
                continue
            seen.add(name.get(c[0], c[0])); a4 = c[1].split(); m7 = re.match(r"(\d+)\s+(\d+)%\((\d+)/(\d+)\)", c[2]); mg = re.match(r"(\d+)\s+(\d+)%\((\d+)/(\d+)\)", c[3])
            L2.append(f"| {name.get(c[0], c[0])} | {a4[0]} / {a4[1]} | {m7.group(1)} / {m7.group(2)}% | {mg.group(1)} / {mg.group(2)}% | {c[4]} | {c[7]} |")
    L += ["", "(나) 같은 코드에 문구만 바꿔 4F·7F·G1F 를 돌린 결과(52_문구섞기_3층, 2026-10-06, 당시 게이트·기본값). 정밀도는 판정한 멈춤만(미판정 제외).", "",
          "| 문구 | 4F 멈춤 / 번호판 본 수(27) | 7F 멈춤 / 정밀도 | G1F 멈춤 / 정밀도 | 7F+G1F 정밀도 | 재멈춤 |", "|---|---|---|---|---|---|"] + L2
    L += ["", "최종: 세 문구 최댓값 + 높이 0.6~2.8 m + 점수 바닥값 0.078(배관 멈춤 제거, 56_). 천장 거르기는 4층 번호판을 잃어 쓰지 않았다."]
    (OUT / "tables/E4_prompts.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    confusion(); prompts()
    if all((P / p / "1F" / "run").exists() for p in POLS):
        rates()
