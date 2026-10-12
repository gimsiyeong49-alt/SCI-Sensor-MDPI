#!/usr/bin/env python3
"""E4 센서 이상 주입 결과 채점(2026-10-08, PILOT). e4_fault_inject.py 로 튐을 넣은 오돔으로 협업자 위치추정을 다시 돌린 뒤
(paper/fault_inject/4F/loc) 게이트를 같은 코드(make_gate_json)로 만들고 원래와 비교한다.
  튐 순간 감지: 그 초(±1 s)에 sensor_bad 가 켜졌나
  혼동: 튐 뒤 다음 튐 전까지(40 s) '애매'로 바뀐 초 수 — 센서 오류를 장소 애매로 착각해 카메라 멈춤을 허용한 것
  위치추정 영향: 같은 구간 1위 3 m 안 비율(채점용 정답)
  tables/E4_fault_injection.md, csv/E4_fault_injection.csv, figures/E4_fig_fault_injection
    python3 scripts/e4_fault_eval.py
"""
import csv, json, math, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import plt, save, BLUE, ORANGE, AQUA, YELLOW, INK, INK2, SURF  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); OUT = R / "59_E4_호출정책/paper"; FI = OUT / "fault_inject/4F"
INJ = json.load(open("/data/kci_work/e4_fault_inject_20261008/4F/INJECTION.json"))["plan"]


def top1_err(loc, F="4F"):
    GD = R / "36_채점용정답" / F; ppm = json.load(open(GD / "summary.json"))["ppm"]
    g = list(csv.DictReader(open(GD / "warp_clean" / "gt_warp.csv")))
    gtt = np.array([float(r["relative_sec"]) for r in g]); gxy = np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])
    e = {}
    for r in csv.DictReader(open(loc / "topk_series.csv")):
        if r["rank"] == "1":
            t = round(float(r["relative_sec"]), 3); k = int(np.argmin(abs(gtt - t)))
            e[t] = math.hypot(float(r["x_px"]) - gxy[k][0], float(r["y_px"]) - gxy[k][1]) / ppm
    return e


def main():
    gate_f = FI / "모호함_v2.json"
    if not gate_f.exists():
        subprocess.run([sys.executable, "scripts/make_gate_json.py", "--loc", str(FI / "loc"), "--out", str(gate_f)], check=True)
    G0 = {round(x["t"], 3): x for x in json.load(open(R / "35_위치추정_축척추정/4F/모호함_v2.json"))}
    G1 = {round(x["t"], 3): x for x in json.load(open(gate_f))}
    E0 = top1_err(R / "35_위치추정_축척추정/4F/loc"); E1 = top1_err(FI / "loc")
    rows = []
    for k, j in enumerate(INJ):
        t0 = float(j["rel_sec"]); t1 = float(INJ[k + 1]["rel_sec"]) if k + 1 < len(INJ) else 400.0
        ts = [t for t in G1 if t0 <= t < t1 and t in G0]
        det = any(G1.get(round(t0 + d, 3), {}).get("sensor_bad") for d in (-1, 0, 1, 2))
        sb = sum(bool(G1[t].get("sensor_bad")) for t in ts)
        a0 = sum(bool(G0[t]["amb"]) for t in ts); a1 = sum(bool(G1[t]["amb"]) for t in ts)
        newamb = sum(bool(G1[t]["amb"]) and not G0[t]["amb"] for t in ts)
        ok0 = np.mean([E0[t] <= 3 for t in ts if t in E0]); ok1 = np.mean([E1[t] <= 3 for t in ts if t in E1])
        rows.append(dict(rel_sec=t0, jump_m=j["jump_m"], jump_deg=j["jump_deg"], window_s=len(ts), sensor_flag_at_jump=int(det), sensor_bad_seconds=sb,
                         amb_seconds_orig=a0, amb_seconds_injected=a1, new_amb_seconds=newamb, top1_3m_orig=round(float(ok0), 3), top1_3m_injected=round(float(ok1), 3)))
    with open(OUT / "csv/E4_fault_injection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["4층 오돔 사본에 계단형 튐 8개를 넣고 협업자 위치추정(60초 창)을 다시 돌린 결과. 구간 = 그 튐부터 다음 튐 전까지. PILOT, 주입 시험(실제 센서 고장 아님).",
         "센서 이상 기준: 오돔 1초 속도 > 2 m/s 또는 회전 > 120°/s.", "",
         "| 튐 | 튐 순간 센서이상 감지 | 구간 초 | 애매 초 원래 → 주입 | 새로 애매해진 초 (착각) | 1위 3 m 안 원래 → 주입 |", "|---|---|---|---|---|---|"]
    for r in rows:
        nm = " + ".join(x for x in ((f"{r['jump_m']:g} m" if r["jump_m"] else ""), (f"{r['jump_deg']:g}°" if r["jump_deg"] else "")) if x)
        L.append(f"| {r['rel_sec']:.0f}s {nm} | {'예' if r['sensor_flag_at_jump'] else '아니오'} | {r['window_s']} | {r['amb_seconds_orig']} → {r['amb_seconds_injected']} | "
                 f"{r['new_amb_seconds']} | {100 * r['top1_3m_orig']:.0f}% → {100 * r['top1_3m_injected']:.0f}% |")
    S = sum(r["window_s"] for r in rows); N = sum(r["new_amb_seconds"] for r in rows)
    L += ["", f"합계: 튐 순간 감지 {sum(r['sensor_flag_at_jump'] for r in rows)}/{len(rows)}, 구간 {S}초 중 새로 애매해진 초 {N} ({100 * N / S:.0f}%)."]
    (OUT / "tables/E4_fault_injection.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # 그림: 원래 vs 주입 1위 오차 + 분류 띠 두 줄
    def cls(g):
        return "sensor" if g.get("sensor_bad") else ("ambiguous" if g["amb"] else "confident")
    col = {"confident": AQUA, "ambiguous": BLUE, "sensor": YELLOW}
    fig, ax = plt.subplots(3, 1, figsize=(10, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 0.45, 0.45]))
    for E, c, lab in ((E0, INK2, "original odometry"), (E1, ORANGE, "odometry with injected jumps")):
        t = np.array(sorted(E)); ax[0].plot(t, np.minimum([E[x] for x in t], 60), lw=1.1, color=c, label=lab)
    for j in INJ:
        for a in ax:
            a.axvline(j["rel_sec"], color=INK, lw=0.6, ls=(0, (2, 2)))
        nm = "+".join(x for x in ((f"{j['jump_m']:g}m" if j["jump_m"] else ""), (f"{j['jump_deg']:g}°" if j["jump_deg"] else "")) if x)
        ax[0].text(j["rel_sec"] + 1, 62, nm, fontsize=7, color=INK)
    ax[0].axhline(3, ls=(0, (4, 3)), color=INK2, lw=0.8); ax[0].set_ylim(0, 66); ax[0].set_ylabel("top-1 error (m)\n(capped 60)")
    ax[0].legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(0, 1.25), ncol=2)
    for a, G, lab in ((ax[1], G0, "original"), (ax[2], G1, "injected")):
        a.grid(False); a.set_yticks([0.5]); a.set_yticklabels([lab]); a.set_ylim(0, 1)
        for t, g in G.items():
            a.axvspan(t - 0.5, t + 0.5, color=col[cls(g)], lw=0)
    from matplotlib.patches import Patch
    ax[2].legend([Patch(color=col[c]) for c in col], list(col), loc="upper center", bbox_to_anchor=(0.5, -1.6), ncol=3, fontsize=7.5, frameon=False)
    ax[2].set_xlabel("time (s)")
    fig.suptitle("E4 fault injection on 4F odometry: jumps are partly flagged as sensor faults, partly mistaken for ambiguity (PILOT)", fontsize=9.5, color=INK)
    fig.tight_layout(); save(fig, "E4_fig_fault_injection")


if __name__ == "__main__":
    main()
