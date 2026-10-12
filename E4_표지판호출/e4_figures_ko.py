#!/usr/bin/env python3
"""E4 설명용 한국어 그림(2026-10-08). 논문용(e4_figures.py)보다 단순하게: 그림 하나에 말 하나, 패널 하나.
결과: 59_E4_호출정책/paper/figures_ko/
  K1_규칙비교  K2_상태별  K3_4층흐름  K4_틀린종류  K5_예시사진
    python3 scripts/e4_figures_ko.py
"""
import csv, json, math, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")   # 한글 글리프 포함(이름은 JP 로 잡힘)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import frame  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper"); FIG = OUT / "figures_ko"
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)
METRICS = os.environ.get("E4_METRICS", "E4_trigger_policy_metrics.csv")

BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#b9bec7", "#dce9f9"
INK, INK2, SURF = "#14171c", "#5b6370", "#ffffff"
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG / f"{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig); print(name)


def k1():
    m = {r["condition"]: r for r in csv.DictReader(open(P / METRICS))}
    rows = [("안 멈춤", "(이벤트 반영 없음)"), ("표지판 보이면 다 멈춤", "P_ALL"), ("애매할 때만 멈춤 (10-08 판)", "P_AMBIGUITY")] + ([(MAIN_NAME, MAIN)] if MAIN != "P_AMBIGUITY" else []) + [
            ("애매 + 정보 될 때만", "P_INFORMATIVE_0.5bit")]
    n = [int(m[k]["requests_sign"]) for _, k in rows]; ok = [100 * float(m[k]["oracle_top1_within3m"]) for _, k in rows]
    fig, a = plt.subplots(figsize=(8.5, 3.0)); y = np.arange(len(rows))[::-1]
    a.barh(y, n, height=0.55, color=[BLUE if k == MAIN else GRAY for _, k in rows])
    for yi, v, o in zip(y, n, ok):
        a.text(v + 8, yi, f"{v}번 멈춤   →   위치 맞음 {o:.0f}%", va="center", fontsize=12, color=INK)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows]); a.set_xlim(0, 760); a.set_xticks([])
    a.spines["bottom"].set_visible(False)
    a.set_title("지금 규칙은 덜 멈추는데 효과는 같다" if MAIN == "P_AMBIGUITY" else f"{MAIN_NAME}: 10-08 판보다 덜 멈추고 효과는 더 좋다", loc="left", fontsize=15, fontweight="bold", color=INK, pad=12)
    fig.text(0.01, -0.06, "13개 층 표지판 멈춤 횟수. '위치 맞음' = 멈춘 곳 번호를 읽었다고 치면 위치추정이 3 m 안에 드는 시간 비율.", fontsize=10, color=INK2)
    save(fig, "K1_규칙비교")


def k2():
    T = list(csv.DictReader(open(OUT / "csv/E4_trigger_decisions.csv"))); D = list(csv.DictReader(open(OUT / "csv/E4_detections.csv")))
    cls = [("확실", "GEOMETRIC_CONFIDENT"), ("헷갈림", "GEOMETRIC_AMBIGUITY"), ("헷갈림 + 긴 직선 복도", "GEOMETRIC_INSUFFICIENT")]
    ok, st = [], []
    for _, c in cls:
        v = [int(r["top1_within_3m_gt"]) for r in T if r["uncertainty_class"] == c and r["top1_within_3m_gt"] != ""]
        ok.append(100 * np.mean(v)); st.append(sum(d["run_id"] == MAIN and d["predicted_class"] == "room_sign" and d["uncertainty_class"] == c for d in D))
    fig, a = plt.subplots(figsize=(8.5, 2.6)); y = np.arange(3)[::-1]
    a.barh(y, ok, height=0.55, color=[GRAY, BLUE, BLUE])
    for yi, v, s in zip(y, ok, st):
        a.text(v + 1.5, yi, f"위치 맞음 {v:.0f}%   ·   멈춤 {s}번", va="center", fontsize=12, color=INK)
    a.set_yticks(y); a.set_yticklabels([c[0] for c in cls]); a.set_xlim(0, 145); a.set_xticks([]); a.spines["bottom"].set_visible(False)
    a.set_title("로봇이 '헷갈린다'고 할 때 실제로 위치가 틀렸고, 그때만 멈췄다", loc="left", fontsize=15, fontweight="bold", color=INK, pad=12)
    fig.text(0.01, -0.08, "13개 층 매 초를 위치추정 상태로 나눔. 위치 맞음 = 1위 후보가 정답 3 m 안. 긴 직선 복도 = 벽이 거의 한 방향뿐인 곳.", fontsize=10, color=INK2)
    save(fig, "K2_상태별")


def k3(F="4F"):
    GD = R / "36_채점용정답" / F; ppm = json.load(open(GD / "summary.json"))["ppm"]
    g = list(csv.DictReader(open(GD / "warp_clean" / "gt_warp.csv")))
    gtt = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
    top = {float(r["relative_sec"]): (float(r["x_px"]), float(r["y_px"])) for r in csv.DictReader(open(R / "35_위치추정_축척추정" / F / "loc/topk_series.csv")) if r["rank"] == "1"}
    ts = np.array(sorted(top)); err = np.array([math.hypot(top[t][0] - gxy[np.argmin(abs(gtt - t))][0], top[t][1] - gxy[np.argmin(abs(gtt - t))][1]) / ppm for t in ts])
    G = json.load(open(R / "35_위치추정_축척추정" / F / "모호함_v2.json"))
    fig, a = plt.subplots(figsize=(10, 3.4))
    for x in G:
        if x["amb"]:
            a.axvspan(x["t"] - 0.5, x["t"] + 0.5, color=LIGHT, lw=0)
    a.plot(ts, np.minimum(err, 40), color=INK, lw=1.3)
    E = json.load(open(P / MAIN / F / "run/events_live_replay.json"))
    a.plot([e["t"] for e in E], [42] * len(E), "v", ms=9, color=ORANGE, clip_on=False)
    a.set_ylim(0, 44); a.set_xlim(ts[0], ts[-1]); a.set_ylabel("위치 오차 (m)"); a.set_xlabel("시간 (초)")
    a.text(305, 30, "파란 배경 = 로봇이 '헷갈린다'고 판단한 시간", fontsize=10.5, color=INK2)
    a.text(305, 25, "▼", fontsize=10.5, color=ORANGE); a.text(313, 25, "= 표지판 보려고 멈춘 곳", fontsize=10.5, color=INK2)
    a.text(305, 20, "검은 선 = 위치추정 오차", fontsize=10.5, color=INK2)
    a.set_title("4층: 위치가 크게 틀어진 구간에서만 멈췄다", loc="left", fontsize=15, fontweight="bold", color=INK, pad=16)
    save(fig, "K3_4층흐름")


def k4():
    T = list(csv.DictReader(open(OUT / "csv/E4_fp_types.csv")))
    from collections import Counter
    C = Counter(r["fp_type_ko"] for r in T); ks = sorted(C, key=lambda k: -C[k])
    fig, a = plt.subplots(figsize=(8.5, 3.0)); y = np.arange(len(ks))[::-1]
    a.barh(y, [C[k] for k in ks], height=0.55, color=GRAY)
    for yi, k in zip(y, ks):
        a.text(C[k] + 0.15, yi, f"{C[k]}개", va="center", fontsize=12, color=INK)
    a.set_yticks(y); a.set_yticklabels(ks); a.set_xticks([]); a.spines["bottom"].set_visible(False); a.set_xlim(0, 9)
    a.set_title(f"틀린 멈춤 {len(T)}개: 절반은 표지판처럼 생긴 종이·글씨", loc="left", fontsize=15, fontweight="bold", color=INK, pad=12)
    save(fig, "K4_틀린종류")


def k5():
    D = [d for d in csv.DictReader(open(OUT / "csv/E4_detections.csv")) if d["run_id"] == MAIN and d["human_label"] in ("TP", "FP")]
    fpt = {r["event_id"]: r["fp_type_ko"] for r in csv.DictReader(open(OUT / "csv/E4_fp_types.csv"))}
    pick = {"TP": ["7F_P_AMBIGUITY_0012", None, None, None], "FP": [None] * 4}
    tp = sorted([d for d in D if d["human_label"] == "TP"], key=lambda d: -float(d["siglip_score"]))
    fp = [d for d in D if d["human_label"] == "FP"]
    by = {}
    for d in fp:                                       # 종류마다 하나씩
        by.setdefault(fpt[d["event_id"]], d)
    sel = {"TP": [], "FP": list(by.values())[:4]}
    seenF = set()
    for d in tp:
        if d["trial_id"] not in seenF or len(sel["TP"]) >= 3:
            sel["TP"].append(d); seenF.add(d["trial_id"])
        if len(sel["TP"]) == 4:
            break
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))
    for r, lab in enumerate(("TP", "FP")):
        for c, d in enumerate(sel[lab]):
            F = d["trial_id"][6:]; t = float(d["relative_sec"]); im = frame(F, t)
            x, y, w, h = (float(d[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            cx, cy = x + w / 2, y + h / 2; half = max(w, h, 70) * 1.6
            x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(640, cx + half)), int(min(480, cy + half))
            a = ax[r, c]; a.imshow(im[y0:y1, x0:x1]); a.set_xticks([]); a.set_yticks([])
            a.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, ec=BLUE if lab == "TP" else ORANGE, lw=3))
            a.set_title("표지판 맞음" if lab == "TP" else f"아님: {fpt[d['event_id']]}", fontsize=12, color=BLUE if lab == "TP" else ORANGE, fontweight="bold")
            for s in a.spines.values():
                s.set_visible(False)
    fig.suptitle("로봇이 멈춰서 고른 곳 (위: 맞음, 아래: 틀림)", x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(); save(fig, "K5_예시사진")


if __name__ == "__main__":
    k1(); k2(); k3(); k4(); k5()
