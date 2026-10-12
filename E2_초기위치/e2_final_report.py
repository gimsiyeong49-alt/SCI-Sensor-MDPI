#!/usr/bin/env python3
"""E2 새 구성 정리(2026-10-10): 입력(A 라이다 / B 라이다+깊이 / C 라이다+깊이+공간 감점) × 축척(탐색·고정 24·정답) × 알고리즘
(벽 겹침 = 우도장 정합 / + 거리 비교 = 광선 거리 비교 / + 정밀 검증 = 미세 정합 + 물리 확인). 평가 전용(정답은 이미 채점된 순위만 읽음).
  벽 겹침: _작업자료/E2_dense_T0<접미사>/E2_top100_per_trial.csv 의 true_rank
  거리 비교: _작업자료/양방향시험/<접미사>.csv 의 rank_twoway (기본 점수 상위 1000개를 협업자 twoway 로 다시 매김)
  정밀 검증: _작업자료/미세정합시험/<접미사>/<과제>.json 의 rank_merit (협업자 metric_fit_v2 후보 처리: 미세 정합 + 물리 확인 + 안 보인 벽)
결과: reports/_01_E2_초기위치/4_결과표와_그림/ (본표_입력x축척x알고리즘.csv, 출발점별_순위.csv, 장소종류별_건물별.csv, 실패원인_정밀검증.csv, 그림)
    python3 scripts/e2_final_report.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

W = Path("reports/_01_E2_초기위치/_작업자료"); OUT = Path("reports/_01_E2_초기위치/4_결과표와_그림"); APP = Path("reports/_01_E2_초기위치/부록")
INP = [("A 라이다", ""), ("B 라이다+깊이", "_depth"), ("C 라이다+깊이+공간 감점", "")]
SRC = {("A 라이다", "탐색"): "T0_nospace", ("A 라이다", "고정 24"): "T0_fixed_nospace", ("A 라이다", "정답(상한)"): "T0_oracle_nospace",
       ("B 라이다+깊이", "탐색"): "T0_depth_nospace", ("B 라이다+깊이", "고정 24"): "T0_fixed_depth_nospace", ("B 라이다+깊이", "정답(상한)"): "T0_oracle_depth_nospace",
       ("C 라이다+깊이+공간 감점", "탐색"): "T0_depth", ("C 라이다+깊이+공간 감점", "고정 24"): "T0_fixed_depth", ("C 라이다+깊이+공간 감점", "정답(상한)"): "T0_oracle_depth"}
REF = {("D 라이다+공간 감점(로봇 원래 설정)", "탐색"): "T0"}
SCALES = ("탐색", "고정 24", "정답(상한)")
ALGO = ("벽 겹침", "+ 거리 비교", "+ 정밀 검증")
KS = (1, 3, 5, 10, 20, 50, 100)
v = lambda x: int(x) if str(x) not in ("", "None") and str(x).isdigit() else None


def ranks(src):
    """과제 → (벽 겹침, 거리 비교, 정밀 검증) 순위와 정답 층 순위. 없으면 None."""
    out = {}
    p1 = W / f"E2_dense_{src}" / "E2_top100_per_trial.csv"
    for r in csv.DictReader(open(p1)):
        out[r["trial_id"]] = {"벽 겹침": v(r["true_rank"]), "벽 겹침_층": v(r["floor_rank"])}
    p2 = W / "양방향시험" / f"{src}.csv"
    if p2.exists():
        for r in csv.DictReader(open(p2)):
            out[r["trial_id"]].update({"+ 거리 비교": v(r["rank_twoway"]), "+ 거리 비교_층": v(r["floor_rank_twoway"])})
    d3 = W / "미세정합시험" / src
    if d3.exists() and len(list(d3.glob("*.json"))) >= 65:
        for f in d3.glob("*.json"):
            j = json.load(open(f)); out[j["task"]].update({"+ 정밀 검증": j["rank_merit"], "+ 정밀 검증_층": j["floor_rank_merit"]})
    return out


def write(name, rows, out=None):
    out = out or OUT; out.mkdir(parents=True, exist_ok=True)
    with open(out / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(W / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(W / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    tasks = list(EV.TASK); R = {k: ranks(s) for k, s in {**SRC, **REF}.items()}
    hit = lambda rr, a, K: sum(1 for t in tasks if rr[t].get(a) is not None and rr[t][a] <= K)
    # 본 표
    tab, ref = [], []
    for (inp, sc), rr in R.items():
        row = {"입력": inp, "축척": sc}
        (ref if (inp, sc) in REF else tab).append(row)
        for a in ALGO:
            done = all(a in rr[t] for t in tasks)
            for K in (1, 10, 100):
                row[f"{a}_Top{K}"] = hit(rr, a, K) if done else ""
            row[f"{a}_Top100_퍼센트"] = round(100 * hit(rr, a, 100) / 65) if done else ""
    write("본표_입력x축척x알고리즘.csv", tab)
    write("로봇원래설정_라이다+공간감점.csv", ref, APP)   # 부록: 로봇 위치추정 원래 설정(D)
    # 입력을 더했을 때(A 대비, 축척 탐색·알고리즘별)
    add = []
    for inp in (i for i, _ in INP):
        row = {"입력": inp}
        for a in ALGO:
            rr, ra = R[(inp, "탐색")], R[("A 라이다", "탐색")]
            if all(a in rr[t] for t in tasks) and all(a in ra[t] for t in tasks):
                n, n0 = hit(rr, a, 100), hit(ra, a, 100)
                row[f"{a}_Top100"] = f"{n} ({round(100 * n / 65)}%)"; row[f"{a}_A대비"] = "기준" if inp == "A 라이다" else f"{n - n0:+d}"
            else:
                row[f"{a}_Top100"] = row[f"{a}_A대비"] = ""
        add.append(row)
    write("입력을_더했을때_축척탐색.csv", add)
    # 정밀 검증 전후 축척 오차(정답 근처 3 m 안 후보들의 중앙값, %): 미세 정합이 축척을 얼마나 맞추나
    sc = []
    for (inp, scl), src in SRC.items():
        if scl != "탐색":
            continue
        d3 = W / "미세정합시험" / src
        J = [json.load(open(f)) for f in d3.glob("*.json")] if d3.exists() else []
        be = [j["true_scale_err_before"] for j in J if j.get("true_scale_err_before") is not None]
        af = [j["true_scale_err_after"] for j in J if j.get("true_scale_err_after") is not None]
        if be:
            sc.append({"입력": inp, "축척": scl, "정답근처후보가_있던_출발점": len(be), "미세정합_전_축척오차_중앙_퍼센트": round(float(np.median(be)), 1),
                       "미세정합_후_축척오차_중앙_퍼센트": round(float(np.median(af)), 1), "오차_10퍼센트_안_전": sum(x <= 10 for x in be), "오차_10퍼센트_안_후": sum(x <= 10 for x in af)})
    if sc:
        write("축척오차_정밀검증_전후.csv", sc)
    # 출발점별 순위
    per = []
    for t in tasks:
        T = EV.TASK[t]; row = {"과제": f"{EV.fl(T['floor'])} 출발점{t.split('_T')[1]}", "출발점종류": EV.TYPE[T["start_type"]]}
        for (inp, sc), rr in R.items():
            if (inp, sc) in REF:
                continue
            for a in ALGO:
                x = rr[t].get(a, "")
                row[f"{inp.split(' ')[0]}·{sc}·{a}"] = x if x != "" and x is not None else ("1000위 밖" if a in rr[t] else "")
        per.append(row)
    write("출발점별_순위.csv", per)
    # 장소 종류별·건물별 (A 라이다·탐색)
    rr = R[("A 라이다", "탐색")]; grp = []
    for name, key in (("출발점 종류", lambda t: EV.TYPE[EV.TASK[t]["start_type"]]), ("건물", lambda t: EV.BNAME[EV.TASK[t]["building"]])):
        for g in dict.fromkeys(key(t) for t in tasks):
            ts = [t for t in tasks if key(t) == g]
            grp.append({"구분": name, "그룹": g, "출발점수": len(ts), **{a: sum(1 for t in ts if rr[t].get(a) and rr[t][a] <= 100) for a in ALGO}})
    write("장소종류별_건물별.csv", grp)
    # 실패 원인(A 라이다·탐색): 벽 겹침 대 정밀 검증, 같은 규칙. 한 출발점에 원인이 여럿일 수 있음
    if all("+ 정밀 검증" in rr[t] for t in tasks):
        ora = R[("A 라이다", "정답(상한)")]; inp = {r["과제"]: r for r in csv.DictReader(open(OUT / "과제별_입력기록.csv", encoding="utf-8-sig"))}
        fl = []
        for a in ("벽 겹침", "+ 정밀 검증"):
            for t in tasks:
                x = rr[t][a]
                if x and x <= 100:
                    continue
                lab = []
                fr = rr[t][a + "_층"]
                if not fr or fr > 100:
                    lab.append("정답 층이 Top-100 에 없음")
                o = ora[t].get(a)
                if o and o <= 100:
                    lab.append("축척 불일치")
                k = f"{EV.fl(EV.TASK[t]['floor'])} 출발점{t.split('_T')[1]}"
                if k in inp and float(inp[k]["실제자리_벽까지거리_m"]) > 0.25:
                    lab.append("지도 불일치")
                if not lab:
                    lab.append("비슷한 다른 자리에 밀림")
                fl.append({"알고리즘": a, "과제": k, "출발점종류": EV.TYPE[EV.TASK[t]["start_type"]], "정답위치_순위": x if x else "1000위 밖",
                           "정답층_최고순위": fr if fr else "1000위 밖", "정답축척일때_순위": o if o else "1000위 밖", "원인": " | ".join(lab)})
        write("실패원인_벽겹침_대_정밀검증.csv", fl)
        from collections import Counter
        for a in ("벽 겹침", "+ 정밀 검증"):
            e = [q for q in fl if q["알고리즘"] == a]; c = Counter(l for q in e for l in q["원인"].split(" | "))
            print("실패", a, len(e), dict(c))
    # 정답 후보(Top-100 안일 때 그 순위의 후보)의 위치·방향·축척 오차: 벽 겹침(축척 목록 그대로) 대 정밀 검증(미세 정합 뒤), A 라이다·축척 탐색
    er = []
    wrap = lambda x: (x + 180) % 360 - 180
    for a, sfx in (("벽 겹침", "_nospace"), ("+ 정밀 검증", "_nospace_정밀검증")):
        pc = W / f"E2_dense_T0{sfx}" / "E2_top100_candidates.csv"
        if not pc.exists():
            continue
        cand = {}
        for r in csv.DictReader(open(pc)):
            cand[(r["trial_id"], int(r["rank"]))] = r
        for t in tasks:
            x = rr[t].get(a)
            if not x or x > 100 or (t, x) not in cand:
                continue
            c = cand[(t, x)]; p_gt, ppm_gt, hd_gt, _, _ = EV.gt_of(t)
            er.append({"알고리즘": a, "과제": f"{EV.fl(EV.TASK[t]['floor'])} 출발점{t.split('_T')[1]}", "정답후보_순위": x,
                       "위치오차_m": round(float(np.hypot(float(c["x_px"]) - p_gt[0], float(c["y_px"]) - p_gt[1]) / ppm_gt), 2),
                       "방향오차_도": round(abs(wrap(float(c["heading_deg"]) - hd_gt)), 1), "축척오차_퍼센트": round(100 * (float(c["ppm"]) - ppm_gt) / ppm_gt, 1)})
    if er:
        write("정답후보_오차_벽겹침_대_정밀검증.csv", er)
        for a in ("벽 겹침", "+ 정밀 검증"):
            e = [q for q in er if q["알고리즘"] == a]
            if e:
                print(a, len(e), "곳 · 위치 중앙", np.median([q["위치오차_m"] for q in e]), "m · 방향 중앙", np.median([q["방향오차_도"] for q in e]),
                      "° · 축척 절대 중앙", np.median([abs(q["축척오차_퍼센트"]) for q in e]), "%")
    # 층별 Top-100 후보 분포(출발점마다 층별 개수): A 라이다 · 알고리즘별 + 참고 정답 축척
    NF = {"kci": 9, "gong": 7, "sw": 8}
    for name, sfx in (("벽겹침", "_nospace"), ("거리비교", "_nospace_거리비교"), ("정밀검증", "_nospace_정밀검증"), ("참고_정답축척_정밀검증", "_oracle_nospace_정밀검증")):
        pc = W / f"E2_dense_T0{sfx}" / "E2_top100_candidates.csv"
        if not pc.exists():
            continue
        cnt = {}
        for r in csv.DictReader(open(pc)):
            if int(r["rank"]) <= 100:
                cnt.setdefault(r["trial_id"], {}).setdefault(int(r["floor_id"]), 0); cnt[r["trial_id"]][int(r["floor_id"])] += 1
        rows = []
        for t in tasks:
            T = EV.TASK[t]; m_true = int(EV.GT[t]["true_map"]); c = cnt.get(t, {}); n = NF[T["building"]]
            row = {"과제": f"{EV.fl(T['floor'])} 출발점{t.split('_T')[1]}", "출발점종류": EV.TYPE[T["start_type"]], "건물": EV.BNAME[T["building"]], "정답층": f"{m_true}층",
                   "정답층_후보수": c.get(m_true, 0), "후보가_있는_층수": sum(1 for v in c.values() if v), "가장_많은_층": "·".join(f"{m}층" for m in sorted(c) if c[m] == max(c.values())) if c else ""}
            for m in range(1, 10):
                row[f"{m}층"] = c.get(m, 0) if m <= n else ""
            rows.append(row)
        write(f"층별_후보분포_{name}.csv", rows)
    figures(R, tasks)
    print("끝")


def figures(R, tasks):
    """옛 판(12조건판, e2_protocol_report.py)과 같은 모양: 결과판(행 = 조건, 칸 = 출발점), Top-K 3칸(축척 탐색·고정 24·정답), 축척 추정."""
    for f in OUT.glob("그림*.png"):
        f.unlink()                                   # 이 스크립트가 만든 그림만 있는 폴더(옛 판 그림은 _작업자료/_옛것)
    SN = {"탐색": "축척 탐색(7개)", "고정 24": "고정 24", "정답(상한)": "정답 축척(참고 상한)"}
    # 결과판 ① 알고리즘별(A 라이다), ② 입력별(+ 정밀 검증)
    board([(("A 라이다", sc), a) for sc in SCALES for a in ALGO], [f"A 라이다 · {SN[sc]} · {a}" for sc in SCALES for a in ALGO], R, tasks,
          "그림1_결과판_알고리즘별.png", "알고리즘별 결과판(A 라이다)")
    board([((inp, sc), "+ 정밀 검증") for inp, _ in INP for sc in SCALES], [f"{inp} · {SN[sc]}" for inp, _ in INP for sc in SCALES], R, tasks,
          "그림2_결과판_입력별.png", "입력별 결과판(+ 정밀 검증)")
    # Top-K: 옛 판 그림2·3 모양
    col_a = {"벽 겹침": EV.GRAY, "+ 거리 비교": EV.BLUE, "+ 정밀 검증": EV.GREEN}; col_i = {"A 라이다": EV.BLUE, "B 라이다+깊이": EV.ORANGE, "C 라이다+깊이+공간 감점": EV.GREEN}
    for fname, title, lines, suf in (
            ("그림3_TopK_정답위치_알고리즘별", "정답 위치(정답 층 + 3 m + 30°)가 Top-K 안에 있는 출발점 비율 · A 라이다, 알고리즘별", [("A 라이다", a, a, col_a[a]) for a in ALGO], ""),
            ("그림4_TopK_정답위치_입력별", "정답 위치(정답 층 + 3 m + 30°)가 Top-K 안에 있는 출발점 비율 · + 정밀 검증, 입력별", [(i, "+ 정밀 검증", i, col_i[i]) for i, _ in INP], ""),
            ("그림5_TopK_정답층_알고리즘별", "(보조) 정답 층 후보가 Top-K 안에 있는 출발점 비율 · A 라이다, 알고리즘별", [("A 라이다", a, a, col_a[a]) for a in ALGO], "_층")):
        fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for a_, sc in zip(ax, SCALES):
            ends = []
            for inp, alg, lab, c in lines:
                rr = R[(inp, sc)]
                if not all(alg in rr[t] for t in tasks):
                    continue
                y = [100 * sum(1 for t in tasks if rr[t].get(alg + suf) and rr[t][alg + suf] <= K) / 65 for K in KS]
                a_.plot(KS, y, marker="o", ms=8, lw=2, color=c, label=lab, markeredgecolor=EV.SURF, markeredgewidth=2); ends.append(y[-1])
            pos = []
            for yy in sorted(ends):                                      # 끝 숫자가 겹치지 않게 최소 4%p 띄움
                pos.append(max(yy, pos[-1] + 4) if pos else yy)
                a_.text(106, pos[-1], f"{yy:.0f}%", va="center", fontsize=10, color=EV.INK)
            a_.set_xscale("log"); a_.set_xticks(KS); a_.set_xticklabels([str(k) for k in KS]); a_.set_ylim(0, 100); a_.grid(alpha=.3); a_.set_xlabel("K")
            a_.set_title(SN[sc], loc="left", fontsize=13, fontweight="bold")
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
        ax[0].set_ylabel("과제 비율 (%)"); ax[0].legend(frameon=False, fontsize=10, loc="upper left")
        fig.suptitle(title + " — 65곳, 제자리 한 번 관측", x=0.01, ha="left", fontsize=15, fontweight="bold"); fig.tight_layout()
        fig.savefig(OUT / f"{fname}.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 축척 추정: 옛 판 그림4 모양(A 라이다 · 축척 탐색, 벽 겹침 단계에서 고른 축척)
    rows = {r["trial_id"]: r for r in csv.DictReader(open(W / "E2_dense_T0_nospace" / "E2_top100_per_trial.csv"))}
    rr = R[("A 라이다", "탐색")]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    a_ = ax[0]; FL = list(dict.fromkeys(t["floor"] for t in EV.TASK.values())); cm = plt.get_cmap("tab20")
    for i, F in enumerate(FL):
        tk = [k for k, t in EV.TASK.items() if t["floor"] == F and rows[k]["truefloor_best_scale"] != ""]
        a_.scatter([float(rows[k]["gt_scale"]) for k in tk], [float(rows[k]["truefloor_best_scale"]) for k in tk], s=70, color=cm(i % 20), edgecolor=EV.SURF, label=EV.fl(F), zorder=3)
    lim = [10, 70]; a_.plot(lim, lim, color=EV.INK2, lw=1, ls=":"); a_.set_xlim(lim); a_.set_ylim(lim); a_.set_aspect("equal"); a_.grid(alpha=.3)
    for sv in sorted({float(r["truefloor_best_scale"]) for r in rows.values() if r["truefloor_best_scale"] != ""}):
        a_.axhline(sv, color="#e3e6ea", lw=0.8, zorder=0)
    a_.set_xlabel("정답 축척 (px/m)"); a_.set_ylabel("추정 축척: 정답 층 최고 후보 (px/m)"); a_.legend(fontsize=8.5, ncol=2, frameon=False, loc="upper left")
    a_.set_title("층마다 추정 축척 vs 정답 축척 (가로 회색 줄 = 고를 수 있는 축척 목록)", loc="left", fontsize=12, fontweight="bold")
    a_ = ax[1]
    for k, r in rows.items():
        if r["truefloor_best_scale_err_pct"] == "" or r["truefloor_best_pos_err_m"] == "":
            continue
        x = rr[k].get("+ 정밀 검증"); c = EV.GREEN if (x and x <= 100) else EV.RED
        a_.scatter(float(r["truefloor_best_scale_err_pct"]), float(r["truefloor_best_pos_err_m"]), s=60, color=c, edgecolor=EV.SURF, zorder=3)
    a_.set_yscale("log"); a_.axhline(3, color=EV.INK2, ls=":", lw=1); a_.text(a_.get_xlim()[0], 3.2, " 3 m (정답 기준)", fontsize=9, color=EV.INK2)
    a_.set_xlabel("추정 축척 오차 (%)"); a_.set_ylabel("정답 층 최고 후보의 위치 오차 (m)"); a_.grid(alpha=.3)
    a_.set_title("축척 오차 대 위치 오차 (초록 = 정밀 검증 뒤 정답 위치 Top-100 안)", loc="left", fontsize=12, fontweight="bold")
    fig.suptitle("축척 추정 (A 라이다 · 축척 탐색, 벽 겹침 단계에서 고른 축척)", x=0.01, ha="left", fontsize=15, fontweight="bold"); fig.tight_layout()
    fig.savefig(OUT / "그림6_축척추정.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def board(keys, names, R, tasks, fname, title):
    """옛 판 그림1(조건별 결과판)과 같은 모양."""
    keep = [i for i, (k, a) in enumerate(keys) if all(a in R[k][t] for t in tasks)]
    keys = [keys[i] for i in keep]; names = [names[i] for i in keep]
    M = np.array([[R[k][t][a] or 5000 for t in tasks] for k, a in keys], float)
    fig, a = plt.subplots(figsize=(20, 0.5 * len(keys) + 2.5))
    colr = lambda v: EV.GREEN if v <= 10 else ("#7fd3b2" if v <= 100 else (EV.YELLOW if v <= 1000 else "#e9ebee"))
    for i in range(len(keys)):
        for j in range(len(tasks)):
            a.add_patch(plt.Rectangle((j, i), 0.92, 0.85, color=colr(M[i, j])))
            if M[i, j] <= 100:
                a.text(j + 0.46, i + 0.43, f"{int(M[i, j])}", ha="center", va="center", fontsize=7, color=EV.INK)
    a.set_xlim(0, len(tasks)); a.set_ylim(len(keys), 0)
    a.set_yticks(np.arange(len(keys)) + 0.43); a.set_yticklabels([nm + f"  ({int((M[i] <= 100).sum())}/65)" for i, nm in enumerate(names)], fontsize=10)
    xt = []; FLs = [EV.TASK[t]["floor"] for t in tasks]
    for F in dict.fromkeys(FLs):
        j = [i for i, x in enumerate(FLs) if x == F]; xt.append((np.mean(j) + 0.46, EV.fl(F))); a.axvline(j[0] - 0.04, color=EV.SURF, lw=3)
    a.set_xticks([x for x, _ in xt]); a.set_xticklabels([n for _, n in xt], fontsize=10); a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title(f"{title}: 칸 하나 = 출발점 하나(층마다 5칸), 숫자 = 정답 위치 순위 · 진초록 Top-10 · 연초록 Top-100 · 노랑 1000위 안 · 회색 밖",
                loc="left", fontsize=13, fontweight="bold")
    fig.savefig(OUT / fname, dpi=140, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    main()
