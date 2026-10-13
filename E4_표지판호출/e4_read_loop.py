#!/usr/bin/env python3
"""E4 운용 흉내(2026-10-13): 첫 진짜 번호판에서 층을 알게 되면 그 뒤로는 그 층 후보만으로 판단한다.
층 모름 E4 는 60초 동안 번호를 하나도 안 읽었다고 치고 매 순간을 셌다. 실제 로봇은 번호판을 읽으면 층이 정해지고, 그 뒤로는 같은 층 안에서만 비교한다.
  읽은 시각 = 채점 결과(채점{SFX}/출발점별_정책별.csv, P_INFORMATIVE)의 first_good_sec: 사람이 표지판이라고 한 멈춤 중 호실 벽 1.5 m 안에 처음 선 시각.
             그 순간 번호를 맞게 읽었다고 친다(판독 성공 상한. 실제 판독 정확도는 E5).
  읽은 뒤 판단 = e4_state_eval.classify 와 같은 기준을 정답 층 후보만으로: 같은 층에 5 m 넘게 떨어진 다른 가설이 Δ 안이거나 1위가 옮겨 간 뒤 10초면 층 내 모호,
             벽 방향 다양성 < 0.1 이면 관측 부족, OdomGuard 사건이면 센서 이상, 아니면 위치 확정. 1위 = 그 층 merit 최고 후보.
  정답(채점용)은 채점과 '읽은 시각'(사람 판정으로 판독 성공을 흉내)에만 쓴다.
    E4_VAR=v8 python3 scripts/e4_read_loop.py
출력: reports/_02_E4_표지판호출/2_본결과_상태판단/번호판읽은뒤{_지도v8}/ (읽기전후.csv, 출발점별.csv, 계산별.csv, 그림)
"""
import csv, json, math, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402,F401
import e4_state_eval as SE  # noqa: E402
import e4_unknown_gate as UG  # noqa: E402

D, K = 0.2, 10.0
SFX = UG.SFX
R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
OUT = Path("reports/_02_E4_표지판호출/2_본결과_상태판단/번호판읽은뒤" + ({"_v8": "_지도v8"}.get(SFX, SFX))); OUT.mkdir(parents=True, exist_ok=True)
wrap = lambda a: (a + 180) % 360 - 180
GREEN, YEL, RED, BLUE, GRAY, INK, INK2 = "#1B7F5B", "#E5B000", "#C0392B", "#2F6DB5", "#B8BFC9", "#1F2328", "#57606A"


def main():
    segs, recs, _ = SE.load(); st0 = SE.classify(recs, D, K, "multi")
    SC = {r["trial_id"]: r for r in csv.DictReader(open(R / "70_층모름E4" / f"채점{SFX}" / "출발점별_정책별.csv")) if r["policy"] == "P_INFORMATIVE"}
    gtc = {}
    def gt(k):
        if k not in gtc:
            F = segs[k]["floor"]; S = json.load(open(R / "36_채점용정답" / F / "summary.json"))
            g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
            gtc[k] = (np.array([float(x["relative_sec"]) for x in g]), np.array([[float(x["x_px"]), float(x["y_px"])] for x in g]), S["ppm"], S["theta_deg"] + S.get("beta_deg", 228.0))
        return gtc[k]
    rows, prev, last_sw = [], {}, {}
    for r, s0 in zip(recs, st0):
        k = r["k"]; T = r["T"]; tr = SC[k]["first_good_sec"]; t_read = float(tr) if tr not in ("", None) else None
        after = t_read is not None and T >= t_read
        if not after:
            st, ok = s0, r["ok"]
        else:
            Pk = [c for c in r["P"] if c["m"] == r["m_true"]]
            if not Pk:
                st, ok = "관측 부족", False
            else:
                top = min(Pk, key=lambda c: c["merit"])
                gs = [c["merit"] - top["merit"] for c in Pk if c is not top and UG.distinct(top, c)]
                if k in prev and UG.distinct(prev[k], top):
                    last_sw[k] = T
                prev[k] = top
                amb = (gs and min(gs) <= D) or (k in last_sw and T - last_sw[k] < K)
                st = "층 내 모호" if amb else "위치 확정"
                if st != "위치 확정" and r["div"] is not None and r["div"] < 0.10:
                    st = "관측 부족"
                if r.get("sensor"):
                    st = "센서 이상"
                tt, gx, ppm, hd = gt(k); pg = gx[int(np.argmin(abs(tt - r["t"])))]
                ok = math.hypot(top["x"] - pg[0], top["y"] - pg[1]) / ppm <= 3.0 and abs(wrap(top["hd"] - hd)) <= 30
        rows.append(dict(출발점=k, 층=r["F"], 시점초=T, 읽은시각초=t_read if t_read is not None else "", 단계="읽은 뒤" if after else "읽기 전",
                         상태=st, 확정=int(st == "위치 확정"), 일위맞음=int(bool(ok)), 관측필요=int(st != "위치 확정" and st != "센서 이상")))
    with open(OUT / "계산별.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def summ(L):
        n = len(L); c = sum(x["확정"] for x in L); a = sum(x["일위맞음"] for x in L)
        right = [x for x in L if x["일위맞음"]]; conf = [x for x in L if x["확정"]]
        return dict(계산수=n, 확정선언=f"{c / n * 100:.0f}% ({c}/{n})" if n else "-", 일위정답=f"{a / n * 100:.0f}% ({a}/{n})" if n else "-",
                    맞는데_관측요청=f"{sum(x['관측필요'] for x in right) / max(len(right), 1) * 100:.0f}% ({sum(x['관측필요'] for x in right)}/{len(right)})",
                    확정일때_맞음=f"{sum(x['일위맞음'] for x in conf) / max(len(conf), 1) * 100:.0f}% ({sum(x['일위맞음'] for x in conf)}/{len(conf)})" if conf else "-",
                    잘못된확신=f"{sum(1 for x in conf if not x['일위맞음']) / max(n, 1) * 100:.1f}% ({sum(1 for x in conf if not x['일위맞음'])}/{n})")
    ph = {"읽기 전": [x for x in rows if x["단계"] == "읽기 전"], "읽은 뒤": [x for x in rows if x["단계"] == "읽은 뒤"], "전체(읽기 없이, 기존 E4)": None}
    base = [dict(x, 확정=int(s == "위치 확정"), 일위맞음=int(bool(r["ok"])), 관측필요=int(s not in ("위치 확정", "센서 이상"))) for x, r, s in zip(rows, recs, st0)]
    tab = [dict(단계=p, **summ(L if L is not None else base)) for p, L in ph.items()]
    nread = sum(1 for k in segs if SC[k]["first_good_sec"] not in ("", None))
    with open(OUT / "읽기전후.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0])); w.writeheader(); w.writerows(tab)
    per = []
    for k in segs:
        L = [x for x in rows if x["출발점"] == k]; A = [x for x in L if x["단계"] == "읽은 뒤"]
        per.append(dict(출발점=k, 층=segs[k]["floor"], 읽은시각초=L[0]["읽은시각초"], 읽은뒤_확정=sum(x["확정"] for x in A), 읽은뒤_계산=len(A),
                        읽은뒤_관측필요=sum(x["관측필요"] for x in A), 읽은뒤_1위맞음=sum(x["일위맞음"] for x in A), 육십초_1위맞음=L[-1]["일위맞음"] if L else ""))
    with open(OUT / "출발점별.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
    # 그림: 읽기 전·후 막대 + 시간축(읽은 시각 기준) 확정·정답 비율
    pct = lambda L, key: 100 * sum(x[key] for x in L) / max(len(L), 1)
    B, A = ph["읽기 전"], ph["읽은 뒤"]
    rightB = [x for x in B if x["일위맞음"]]; rightA = [x for x in A if x["일위맞음"]]
    vals = [("1위가 정답", pct(B, "일위맞음"), pct(A, "일위맞음"), GREEN), ("확정 선언", pct(B, "확정"), pct(A, "확정"), BLUE),
            ("1위가 맞는데 관측 요청", pct(rightB, "관측필요"), pct(rightA, "관측필요"), YEL)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axs[0]; x = np.arange(len(vals)); wd = 0.36
    ax.bar(x - wd / 2, [v[1] for v in vals], wd, color=GRAY, label="번호판 읽기 전(층 모름)")
    ax.bar(x + wd / 2, [v[2] for v in vals], wd, color=[v[3] for v in vals], label="번호판 읽은 뒤(층 고정)")
    for i, v in enumerate(vals):
        ax.text(i - wd / 2, v[1] + 1.5, f"{v[1]:.0f}%", ha="center", fontsize=11, color=INK2); ax.text(i + wd / 2, v[2] + 1.5, f"{v[2]:.0f}%", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([v[0] for v in vals], fontsize=11); ax.set_ylim(0, 110); ax.set_ylabel("계산 순간 비율 (%)")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper right"); ax.set_title(f"번호판을 읽기 전과 후 (읽은 출발점 {nread}/65곳)", loc="left", fontsize=13, fontweight="bold")
    ax = axs[1]; rel = {}
    for x_ in rows:
        if x_["읽은시각초"] == "":
            continue
        d = int(round((x_["시점초"] - x_["읽은시각초"]) / 3.0) * 3)
        rel.setdefault(d, []).append(x_)
    NMIN = 10                                                        # 2026-10-13: 출발점이 10곳 안 되는 시점은 그리지 않음(끝 쪽은 표본이 적음)
    ds = sorted(d for d in rel if -30 <= d <= 45 and len({x["출발점"] for x in rel[d]}) >= NMIN)
    ax.plot(ds, [pct(rel[d], "일위맞음") for d in ds], "-o", color=GREEN, lw=2.5, ms=7, label="1위가 정답", zorder=3)
    ax.plot(ds, [pct(rel[d], "확정") for d in ds], "-s", color=BLUE, lw=2.0, ms=5, label="확정 선언", zorder=4)
    for d in ds[::2]:
        ax.text(d, 92, f"{len({x['출발점'] for x in rel[d]})}", ha="center", fontsize=8.5, color=INK2)
    ax.text(ds[0] - 4.5, 92, "n", ha="center", fontsize=8.5, color=INK2, style="italic")
    ax.axvline(0, color=INK, lw=1.2, ls="--"); ax.text(0.5, 103, "번호판 읽음", fontsize=10.5, color=INK)
    ax.set_xlabel("번호판을 읽은 순간 기준 시간(초)"); ax.set_ylabel("출발점 비율 (%)"); ax.set_ylim(0, 110)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E6E8EB"); ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax.set_title(f"읽은 순간을 기준으로 맞춰 보면 (위 숫자 n = 그 시점 출발점 수, {NMIN}곳 미만 시점은 뺌)", loc="left", fontsize=12, fontweight="bold")
    fig.suptitle("E4 운용 흉내: 번호판 하나를 읽어 층을 정하면 확정 선언이 생기고 불필요한 관측 요청이 준다 (판독 성공 상한)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "읽기전후.png", dpi=130, bbox_inches="tight"); fig.savefig(OUT / "읽기전후.pdf", bbox_inches="tight"); plt.close(fig)
    for t in tab:
        print(t)
    print("읽은 출발점", nread)


if __name__ == "__main__":
    main()
