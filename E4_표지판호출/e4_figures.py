#!/usr/bin/env python3
"""E4 논문 그림(2026-10-07, PILOT). e4_paper.py·e4_eval.py·e4_detection.py 결과를 읽어 59_E4_호출정책/paper/figures/ 에 PNG·PDF 로 그린다.
  E4_fig_detection_examples  사람 판정 TP 6개·FP 6개(P_AMBIGUITY 멈춤, 상자·SigLIP 점수·깊이)
  E4_fig_pr_room_sign        표지판 후보 PR 곡선(판정 248개), 기준값 0.078 표시
  E4_fig_policy              정책별 요청 수 / 표지판 정밀도 / 번호 읽었다면 1위 3 m 안(축 하나씩, 세 칸)
  E4_fig_uncertainty_classes 분류별 시간 비율 / 1위 3 m 안 / 멈춤 정밀도
  E4_fig_timeline_<층>       1위 오차(채점용 정답) + 분류 띠 + 멈춤 표시
색: dataviz 기준 팔레트(palette.md 범주색 앞에서부터 순서대로).
    python3 scripts/e4_figures.py [--timeline S6F,4F]
"""
import argparse, csv, json, math, os
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper"); FIG = OUT / "figures"
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)
METRICS = os.environ.get("E4_METRICS", "E4_trigger_policy_metrics.csv")

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLS = ["GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT", "SENSOR_UNRELIABLE"]
CLS_SHORT = {"GEOMETRIC_CONFIDENT": "confident", "GEOMETRIC_AMBIGUITY": "ambiguous", "GEOMETRIC_INSUFFICIENT": "insufficient", "SENSOR_UNRELIABLE": "sensor"}
CLS_COL = dict(zip(CLS, [AQUA, BLUE, ORANGE, YELLOW]))
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight"); fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print("그림", name)


def frame_root(F):
    return Path("/data2/kci_datasets/4F_RGB_FULL_V0_1") if F == "4F" else Path(f"/data3/kci_datasets/{F}_RGB_FULL_5HZ_V0_1")


_idx = {}


def frame(F, t):
    if F not in _idx:
        rows = list(csv.DictReader(open(frame_root(F) / "index.csv")))
        col = "header_relative_sec" if "header_relative_sec" in rows[0] else "bag_relative_sec"
        _idx[F] = (rows, np.array([float(r[col]) for r in rows]))
    rows, tt = _idx[F]
    return cv2.cvtColor(cv2.imread(str(frame_root(F) / rows[int(np.argmin(abs(tt - t)))]["filename"])), cv2.COLOR_BGR2RGB)


def fig_examples(D):
    J = [d for d in D if d["run_id"] == MAIN and d["predicted_class"] == "room_sign" and d["human_label"] in ("TP", "FP")]
    pick = {}
    for lab in ("TP", "FP"):                       # 층이 고르게 섞이도록 층마다 돌아가며, 점수 높은 순
        by = {}
        for d in sorted((d for d in J if d["human_label"] == lab), key=lambda d: -float(d["siglip_score"])):
            by.setdefault(d["trial_id"], []).append(d)
        out, k = [], 0
        while len(out) < 6 and any(by.values()):
            for f in sorted(by):
                if by[f] and len(out) < 6:
                    out.append(by[f].pop(0))
        pick[lab] = out
    fig, ax = plt.subplots(2, 6, figsize=(12, 3.6))
    for r, lab in enumerate(("TP", "FP")):
        for c, d in enumerate(pick[lab]):
            F = d["trial_id"].replace("PILOT_", ""); t = float(d["relative_sec"])
            im = frame(F, t); x, y, w, h = (float(d[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            cx, cy = x + w / 2, y + h / 2; half = max(w, h, 60) * 1.4         # 상자 주변을 크게
            x0, y0 = int(max(0, cx - half)), int(max(0, cy - half)); x1, y1 = int(min(640, cx + half)), int(min(480, cy + half))
            a = ax[r, c]; a.imshow(im[y0:y1, x0:x1]); a.set_xticks([]); a.set_yticks([]); a.grid(False)
            a.add_patch(plt.Rectangle((x - x0, y - y0), w, h, fill=False, ec=AQUA if lab == "TP" else ORANGE, lw=2))
            a.set_title(f"{F} {t:.0f}s  s={float(d['siglip_score']):.3f}  {float(d['depth_m']):.1f} m", fontsize=7.5, color=INK)
            for s in a.spines.values():
                s.set_visible(False)
        ax[r, 0].set_ylabel("room sign (TP)" if lab == "TP" else "not a sign (FP)", fontsize=9)
    fig.suptitle(f"E4 stop boxes judged by a human (PILOT, {MAIN_NAME})", fontsize=10, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_detection_examples")


def fig_pr():
    d = np.load(R / "52_문구섞기_3층/배관검사_점수.npz", allow_pickle=True)
    PH = list(d["PH"]); idx = [PH.index(p) for p in ("a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall")]
    S = d["S"][:, idx].max(1); Y = d["Y"].astype(bool)
    o = np.argsort(-S); tp = np.cumsum(Y[o]); prec = tp / np.arange(1, len(S) + 1); rec = tp / Y.sum()
    ap_ = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0])
    sel = S >= 0.078; p_th, r_th = Y[sel].mean(), Y[sel].sum() / Y.sum()
    fig, a = plt.subplots(figsize=(4.2, 3.4))
    a.plot(rec, prec, lw=2, color=BLUE)
    a.axhline(Y.mean(), ls=(0, (4, 3)), color=INK2, lw=1)
    a.text(0.02, Y.mean() + 0.02, f"base rate {Y.mean():.2f}", color=INK2, fontsize=8)
    a.plot([r_th], [p_th], "o", ms=8, color=ORANGE, mec=SURF, mew=2, zorder=3)
    a.annotate(f"floor 0.078\nP {p_th:.2f}, R {r_th:.2f}", (r_th, p_th), (r_th - 0.42, p_th + 0.22), fontsize=8, color=INK,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    a.set_xlabel("recall"); a.set_ylabel("precision"); a.set_xlim(0, 1.02); a.set_ylim(0, 1.02)
    a.set_title(f"Room-sign candidates (n={len(S)}, {int(Y.sum())} signs), PR-AUC {ap_:.2f}", fontsize=9, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_pr_room_sign")


def fig_policy():
    rows = list(csv.DictReader(open(P / METRICS)))
    name = {"(이벤트 반영 없음)": "no events", "P_ALL": "P_ALL", "P_AMBIGUITY": "P_AMBIGUITY (10-08)", "P_R3B": "C: deployed 10-09",
            "P_INFORMATIVE_0.5bit": "P_INFO ≥0.5 bit", "P_INFORMATIVE_1bit": "P_INFO ≥1 bit"}
    rows = [r for r in rows if r["condition"] in name]
    lab = [name[r["condition"]] for r in rows]; y = np.arange(len(rows))[::-1]
    req = [int(r["requests_sign"]) + int(r["requests_stair"]) for r in rows]
    prec = [100 * float(r["sign_precision_judged"]) if int(r["judged_n"]) else np.nan for r in rows]
    ok = [100 * float(r["oracle_top1_within3m"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for a, v, t, fmt in ((ax[0], req, "camera requests (13 floors)", "{:.0f}"), (ax[1], prec, "room-sign precision, judged stops (%)", "{:.1f}"),
                         (ax[2], ok, "top-1 within 3 m if read (%)", "{:.1f}")):
        a.barh(y, np.nan_to_num(v), height=0.6, color=BLUE, ec=SURF, lw=2)
        for yi, vi in zip(y, v):
            a.text((0 if np.isnan(vi) else vi) + max(np.nanmax(v), 1) * 0.02, yi, "–" if np.isnan(vi) else fmt.format(vi), va="center", fontsize=8, color=INK)
        a.set_title(t, fontsize=9, color=INK); a.grid(axis="y", visible=False); a.set_xlim(0, np.nanmax(v) * 1.18)
    ax[0].set_yticks(y); ax[0].set_yticklabels(lab)
    fig.suptitle("E4 trigger policies (PILOT; precision on 4 judged floors; 'if read' = oracle room numbers)", fontsize=9.5, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_policy")


def fig_classes(T, D):
    sec = {c: 0 for c in CLS}; ok = {c: [0, 0] for c in CLS}
    for r in T:
        c = r["uncertainty_class"]; sec[c] += 1
        if r["top1_within_3m_gt"] != "":
            ok[c][1] += 1; ok[c][0] += int(r["top1_within_3m_gt"])
    tot = sum(sec.values())
    pr = {c: [0, 0] for c in CLS}
    for d in D:
        if d["run_id"] == MAIN and d["predicted_class"] == "room_sign" and d["human_label"] in ("TP", "FP"):
            pr[d["uncertainty_class"]][1] += 1; pr[d["uncertainty_class"]][0] += d["human_label"] == "TP"
    y = np.arange(len(CLS))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.4), sharey=True)
    vals = ([100 * sec[c] / tot for c in CLS], [100 * ok[c][0] / max(ok[c][1], 1) for c in CLS],
            [100 * pr[c][0] / pr[c][1] if pr[c][1] else np.nan for c in CLS])
    txt = ([f"{100 * sec[c] / tot:.1f}% ({sec[c]} s)" for c in CLS], [f"{100 * ok[c][0] / max(ok[c][1], 1):.1f}%" for c in CLS],
           [f"{pr[c][0]}/{pr[c][1]}" if pr[c][1] else "no stops" for c in CLS])
    for a, v, tx, t in zip(ax, vals, txt, ("share of time", "localizer top-1 within 3 m", "sign-stop precision (judged)")):
        a.barh(y, np.nan_to_num(v), height=0.6, color=[CLS_COL[c] for c in CLS], ec=SURF, lw=2)
        for yi, vi, s in zip(y, v, tx):
            a.text((0 if np.isnan(vi) else vi) + 2, yi, s, va="center", fontsize=8, color=INK)
        a.set_xlim(0, 118); a.set_title(t + " (%)", fontsize=9, color=INK); a.grid(axis="y", visible=False)
    ax[0].set_yticks(y); ax[0].set_yticklabels([CLS_SHORT[c] for c in CLS])
    fig.suptitle(f"E4 uncertainty classes over 13 floors ({tot} s, PILOT; insufficient = ambiguous and wall-normal diversity < 0.1)", fontsize=9.5, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_uncertainty_classes")


def fig_timeline(F, T, D):
    GD = R / "36_채점용정답" / F; ppm = json.load(open(GD / "summary.json"))["ppm"]
    g = list(csv.DictReader(open(GD / "warp_clean" / "gt_warp.csv")))
    gtt = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
    top = {}
    for r in csv.DictReader(open(R / "35_위치추정_축척추정" / F / "loc" / "topk_series.csv")):
        if r["rank"] == "1":
            top[float(r["relative_sec"])] = (float(r["x_px"]), float(r["y_px"]))
    ts = np.array(sorted(top)); err = np.array([math.hypot(top[t][0] - gxy[np.argmin(abs(gtt - t))][0], top[t][1] - gxy[np.argmin(abs(gtt - t))][1]) / ppm
                                                 for t in ts])
    rows = [r for r in T if r["trial_id"] == f"PILOT_{F}"]
    ct = np.array([float(r["relative_sec"]) for r in rows]); cc = [r["uncertainty_class"] for r in rows]
    dv = np.array([float(r["normal_diversity"]) if r["normal_diversity"] else np.nan for r in rows])
    fig, ax = plt.subplots(3, 1, figsize=(10, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4, 0.5]))
    a = ax[0]; a.plot(ts, np.minimum(err, 60), lw=1.2, color=INK2)
    a.axhline(3, ls=(0, (4, 3)), color=INK2, lw=0.8); a.text(ts[-1], 3.5, "3 m", ha="right", fontsize=7.5, color=INK2)
    a.set_ylabel("top-1 error (m)\n(capped 60)")
    st = [d for d in D if d["trial_id"] == f"PILOT_{F}" and d["run_id"] == MAIN]
    for kind, mk, col, lab in (("room_sign", "v", BLUE, "sign stop"), ("stair", "^", ORANGE, "stair observation")):
        tt = [float(d["relative_sec"]) for d in st if d["predicted_class"] == kind]
        if tt:
            a.plot(tt, np.full(len(tt), 62), mk, ms=6, color=col, mec=SURF, mew=1, clip_on=False, label=f"{lab} ({len(tt)})")
    a.set_ylim(0, 65); a.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2, bbox_to_anchor=(1, 1.18))
    ax[1].plot(ct, dv, lw=1, color=INK2); ax[1].axhline(0.1, ls=(0, (4, 3)), color=ORANGE, lw=0.8)
    ax[1].set_ylabel("wall-normal\ndiversity"); ax[1].set_ylim(0, 1)
    b = ax[2]; b.grid(False); b.set_yticks([])
    for t, c in zip(ct, cc):
        b.axvspan(t - 0.5, t + 0.5, color=CLS_COL[c], lw=0)
    from matplotlib.patches import Patch
    b.legend([Patch(color=CLS_COL[c]) for c in CLS], [CLS_SHORT[c] for c in CLS], loc="upper center", bbox_to_anchor=(0.5, -1.7), ncol=4, fontsize=7.5, frameon=False)
    b.set_xlabel("time (s)"); b.set_ylabel("class", rotation=0, ha="right", va="center")
    fig.suptitle(f"E4 {F}: localizer error, geometric insufficiency and camera requests (PILOT, {MAIN_NAME})", fontsize=9.5, color=INK)
    fig.tight_layout(); save(fig, f"E4_fig_timeline_{F}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--timeline", default="S6F,4F"); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    T = list(csv.DictReader(open(OUT / "csv" / "E4_trigger_decisions.csv"))); D = list(csv.DictReader(open(OUT / "csv" / "E4_detections.csv")))
    fig_examples(D); fig_pr(); fig_policy(); fig_classes(T, D)
    for F in a.timeline.split(","):
        fig_timeline(F, T, D)


if __name__ == "__main__":
    main()
