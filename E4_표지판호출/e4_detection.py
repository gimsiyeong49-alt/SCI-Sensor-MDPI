#!/usr/bin/env python3
"""E4 검출 정확도(2026-10-07, 실험 지침서 14.6, 파일럿). 사람 판정 멈춤 248개(표지판/아님)의 SigLIP 점수(지금 기본 문구 셋 최댓값)로
PR 곡선·PR-AUC·ROC-AUC, 기준값(바닥값 0.078)에서의 정밀도·재현율을 낸다. 점수는 52_문구섞기_3층/배관검사_점수.npz(phrase_pipe_check.py).
주의: 판정 대상이 '이미 멈춘 상자'라 검출 전체의 재현율이 아니라 멈춤 후보 안에서의 구별력이다. 4층 프레임 단위 재현율은 따로 낸다.
    python3 scripts/e4_detection.py --out <폴더>
"""
import argparse, csv, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); out = Path(a.out)
R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
d = np.load(R / "52_문구섞기_3층/배관검사_점수.npz", allow_pickle=True)
PH = list(d["PH"]); idx = [PH.index(p) for p in ("a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall")]
S = d["S"][:, idx].max(1); Y = d["Y"].astype(bool)
o = np.argsort(-S); tp = np.cumsum(Y[o]); fp = np.cumsum(~Y[o]); prec = tp / np.arange(1, len(S) + 1); rec = tp / Y.sum()
ap_ = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0])
r = np.argsort(np.argsort(S)) + 1; auc = float((r[Y].sum() - Y.sum() * (Y.sum() + 1) / 2) / (Y.sum() * (~Y).sum()))
th = 0.078; sel = S >= th
p_th = float(Y[sel].mean()); r_th = float(Y[sel].sum() / Y.sum())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4)); plt.plot(rec, prec, lw=2, color="#1f6feb"); plt.scatter([r_th], [p_th], color="#d1242f", zorder=3, label=f"floor 0.078 (P {p_th:.2f}, R {r_th:.2f})")
plt.axhline(Y.mean(), ls="--", color="gray", lw=1, label=f"base rate {Y.mean():.2f}")
plt.xlabel("recall"); plt.ylabel("precision"); plt.title(f"room-sign candidates (n={len(S)}), PR-AUC {ap_:.2f}"); plt.legend(loc="lower left", fontsize=8); plt.ylim(0, 1.02); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(out / "E4_PR_room_sign.png", dpi=150)
# 4층 정답 구간 재현율(프레임 단위): 번호판 27개 중 상위 8 벽 구간에 든 것 / 최종 멈춤에 든 것 (45_ 진단 실행, 문구 둘중최대 기준)
st = [ok for l in open(R / "57_바닥값기본_전층/점검.jsonl") for (_, ok, _) in json.loads(l)["fac"].get("계단", [])]   # 2026-10-08: 31/36 은 옮겨 적다 틀린 값
txt = [f"표지판 후보 구별(사람 판정 {len(S)}개, 표지판 {int(Y.sum())}): PR-AUC {ap_:.3f}, ROC-AUC {auc:.3f}, 기준값 0.078 에서 정밀도 {p_th:.3f} 재현율 {r_th:.3f}",
       f"계단 관측 정밀도(자동 채점, 13개 층 최종 설정): {sum(st)}/{len(st)} = {sum(st) / len(st):.3f} (57_바닥값기본_전층/점검.jsonl)",
       "4층 번호판 27개: 상위 8 벽 구간에 든 것 20/27, 최종 멈춤 상자에 든 것 4/27(최종 설정) — 45_문근처_오답분석"]
(out / "E4_검출정확도.txt").write_text("\n".join(txt) + "\n"); print("\n".join(txt))
with open(out / "E4_detection_metrics.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["experiment_id", "class", "n", "n_pos", "pr_auc", "roc_auc", "threshold", "precision_at_th", "recall_at_th", "split", "note"])
    w.writerow(["E4", "room_sign", len(S), int(Y.sum()), round(ap_, 4), round(auc, 4), th, round(p_th, 4), round(r_th, 4), "PILOT", "판정 대상=멈춤 상자"])
    w.writerow(["E4", "stair", len(st), "", "", "", "", round(sum(st) / len(st), 4), "", "PILOT", "자동 채점(의미지도 계단 칸)"])
