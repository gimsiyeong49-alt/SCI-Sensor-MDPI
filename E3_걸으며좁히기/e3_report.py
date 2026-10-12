#!/usr/bin/env python3
"""E3 채점 모음·표·그림(평가 전용, 2026-10-10). e3_dense.py 점수판을 e2_dense_score.py 로 채점한 결과(_작업자료/score/<조건>_T<T>)를 모은다.
T0 = E2 같은 계열 조건(T1·T2·T2X 는 E2 A, T2S·T3 는 E2 협업자 설정, T2d 는 E2 B, T3d 는 E2 C)의 제자리 한 번 관측 결과.
결과(reports/_03_E3_걸으며좁히기/3_결과표와_그림/, 한글 파일·열 이름):
  조건시점표.csv            조건 × 시점: 정답 위치 Top-1·10·100, 정답 층 Top-100, 1000위 안, 순위 중앙, 정답 후보 유지(남은 후보 안), 정답 층 유지, 남은 후보 수·층 수 중앙, 1위 오차
  과제시점별_지표.csv        출발점 × 조건 × 시점
  정답후보_사라진사유.csv    T0 또는 앞 시점에 Top-100 이던 정답이 밖으로 나간 경우와 벽 관통 제거 여부
  실제오돔끊김.csv           출발 뒤 60초 안에 실제 오돔 끊김(E4 센서 판단)이 있던 출발점
  그림1~5
    python3 scripts/e3_report.py [--score]     # --score: 채점부터(조건 7 × 시점 5, 8개씩 병렬)
"""
import argparse, csv, json, math, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

PKG = Path("reports/_03_E3_걸으며좁히기"); W = PKG / "_작업자료"; OUT = PKG / "3_결과표와_그림"; SRC = Path("/data2/e3_dense_20261010")
E2W = Path("reports/_01_E2_초기위치/_작업자료")
TS = (6, 15, 30, 45, 60)
TS_FINE = tuple(range(3, 61, 3)); COND_FINE = ("T2", "T3")          # 3초 간격(주요 조건만)
COND = [("T1", "T1 이동만"), ("T2", "T2 시간 누적"), ("T2X", "T2 + 벽 관통 경로 제거"), ("T2S", "T2 + 공간 감점"), ("T3", "T3 물리 조건(관통 제거 + 공간 감점)"),
        ("T2d", "T2 깊이 장면"), ("T3d", "T3 깊이 장면"), ("T3on", "전 칸 누적 + 축척 걸으며 추정")]
T0SRC = {"T1": "E2_dense_T0_nospace", "T2": "E2_dense_T0_nospace", "T2X": "E2_dense_T0_nospace", "T2S": "E2_dense_T0", "T3": "E2_dense_T0",
         "T2d": "E2_dense_T0_depth_nospace", "T3d": "E2_dense_T0_depth", "T3on": "E2_dense_T0"}
COLOR = {"T1": EV.GRAY, "T2": EV.BLUE, "T2X": "#8a5cd1", "T2S": EV.YELLOW, "T3": EV.GREEN, "T2d": EV.ORANGE, "T3d": EV.RED, "T3on": EV.INK}
REAL_GAPS = {"4F": [130.3, 251.8, 269.8], "5F": [260.49], "9F": [404.39], "G1F": [125.7]}   # E4 센서 판단(정상 녹화본) 결과, paper/sensor_check/*.json
rk = lambda v: int(v) if str(v).isdigit() else None
ko_task = lambda k: f"{EV.fl(k.split('_')[0])} 출발점{k.split('_T')[1]}"


def score_all():
    jobs = []
    for c, _ in COND:
        for T in sorted(set(TS) | (set(TS_FINE) if c in COND_FINE else set())):
            res = W / "score" / f"{c}_T{T}"
            if not (res / "E2_top100_per_trial.csv").exists():
                jobs.append(["python3", "-W", "ignore", "scripts/e2_dense_score.py", "--T", str(T), "--src-dir", str(SRC / f"{c}_T{T}"), "--res-dir", str(res)])
    print("채점", len(jobs), flush=True)
    for k in range(0, len(jobs), 8):
        ps = [subprocess.Popen(j, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for j in jobs[k:k + 8]]
        for p in ps:
            p.wait()
        print("  ", min(k + 8, len(jobs)), "/", len(jobs), flush=True)


def load():
    D = {}
    for c, _ in COND:
        D[(c, 0)] = {r["trial_id"]: r for r in csv.DictReader(open(E2W / T0SRC[c] / "E2_top100_per_trial.csv"))}
        for T in sorted(set(TS) | (set(TS_FINE) if c in COND_FINE else set())):
            D[(c, T)] = {r["trial_id"]: r for r in csv.DictReader(open(W / "score" / f"{c}_T{T}" / "E2_top100_per_trial.csv"))}
    return D


def times_of(c):
    return (0,) + (TS_FINE if c in COND_FINE else TS)


def accum_stats(task, T, L=None):
    """출발~T초 누적 관측: 관측 장수, 라이다 구조점 수(전부), 걸은 거리(오돔, m). 위치추정과 같은 창 고르기."""
    import e2_dense as ED
    from scan_timing import window_indices
    t = EV.TASK[task]
    if L is None:
        EV.obs(task); L = ED._ST["cache"][(t["floor"], 1)].L
    ts = float(t["start_sec"]); lst = window_indices(L.T, L.T[0] + ts + T, T, 3.0)
    n_pts = sum(len(L._scan(L.T[i])[0]) for _, i in lst)
    sel = (L.T >= L.T[0] + ts) & (L.T <= L.T[0] + ts + T); x, y = np.asarray(L.X)[sel], np.asarray(L.Y)[sel]
    return dict(frames=len(lst), points=int(n_pts), dist_m=round(float(np.hypot(np.diff(x), np.diff(y)).sum()), 1))


def removed_by_crossing(task, c, T):
    """정답 자리 후보(정답 층·실제 자리 3 m·방향 30° 안, 어느 축척이든)가 이 조건 점수판에서 전부 빠졌나(벽 관통 제거)."""
    t = EV.TASK[task]; F = t["floor"]; m = int(EV.GT[task]["true_map"])
    S = json.load(open(EV.R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]; yaw_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
    gp = list(csv.DictReader(open(EV.R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
    gt = np.array([float(x["relative_sec"]) for x in gp]); gx = np.array([[float(x["x_px"]), float(x["y_px"])] for x in gp])
    pg = gx[int(np.argmin(abs(gt - (float(t["start_sec"]) + T))))]
    d = np.load(SRC / f"{c}_T{T}" / f"{task}.npz"); meta = json.loads(str(d["meta"]))["maps"][str(m)]
    wrap = lambda a: (a + 180) % 360 - 180
    okp = [i for i, p in enumerate(meta["modes"]) if abs(wrap(p + math.degrees(meta["beta"]) - yaw_gt)) <= 30]
    cpx = d[f"cellpx_{m}"]; near = np.flatnonzero(np.hypot(*(cpx - pg).T) / ppm <= 3.0)
    sel = (d["floor"] == m) & np.isin(d["phi_idx"], okp) & np.isin(d["cell"], near)
    if not sel.any():
        return None
    return bool(np.all(~np.isfinite(d["score"][sel])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--score", action="store_true"); a = ap.parse_args()
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(E2W / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(E2W / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    if a.score:
        score_all()
    D = load(); OUT.mkdir(parents=True, exist_ok=True)
    tasks = [k for k in EV.TASK if all(k in D[key] for key in D)]     # E3 점수판이 있는 출발점만(E2 기준값도 같은 출발점으로 맞춤)
    print("출발점", len(tasks), "곳 (빠짐:", ", ".join(ko_task(k) for k in EV.TASK if k not in tasks) or "없음", ")", flush=True)
    hit = lambda r, k: int(rk(r["true_rank"]) is not None and rk(r["true_rank"]) <= k)
    # 1) 조건 × 시점 표
    tab = []
    for c, cn in COND:
        tasks_c = [k for k in EV.TASK if all(k in D[(c, T)] for T in times_of(c))]    # 이 조건이 계산된 출발점(기존 위치추정은 소융대 포함)
        for T in (0,) + TS:
            rr = [D[(c, T)][k] for k in tasks_c]; trs = [rk(r["true_rank"]) for r in rr]
            f = lambda key: [float(r[key]) for r in rr if r.get(key, "") not in ("", None)]
            tab.append({"조건": cn, "시점_초": T, "출발점수": len(rr), "정답위치_Top1": sum(hit(r, 1) for r in rr), "정답위치_Top10": sum(hit(r, 10) for r in rr),
                        "정답위치_Top100": sum(hit(r, 100) for r in rr), "정답층_Top100": sum(int(r["floor_recall_100"]) for r in rr),
                        "정답위치_1000위안": sum(x is not None for x in trs), "정답위치_순위중앙_1000위안": int(np.median([x for x in trs if x])) if any(trs) else "",
                        "정답후보_남은후보안": sum(int(r.get("true_in_eff") or 0) for r in rr) if T else "",
                        "정답층_남은후보안": sum(int(r.get("truefloor_in_eff") or 0) for r in rr) if T else "",
                        "남은후보수_중앙": float(np.median(f("n_eff"))) if T else "", "남은후보_층수_중앙": float(np.median(f("n_eff_floors"))) if T else "",
                        "1위_정답층일때_위치오차중앙_m": round(float(np.median(f("top1_pos_err_m"))), 2) if f("top1_pos_err_m") else "",
                        "1위가_정답층인_출발점": sum(1 for k in tasks_c if str(D[(c, T)][k].get("top1_floor", "")) == EV.GT[k]["true_map"]),
                        "벽관통_제거된_원후보_중앙": float(np.median(f("n_raw_removed"))) if T and f("n_raw_removed") else ""})
    write("조건시점표.csv", tab)
    # 2) 과제 × 조건 × 시점
    per = []
    for k in tasks:
        t = EV.TASK[k]
        for c, cn in COND:
            for T in (0,) + TS:
                r = D[(c, T)][k]
                per.append({"과제": ko_task(k), "층": EV.fl(t["floor"]), "출발점종류": EV.TYPE[t["start_type"]], "조건": cn, "시점_초": T,
                            "정답위치_순위": r["true_rank"].replace(">1000", "1000위 밖"), "정답층_최고순위": r["floor_rank"].replace(">1000", "1000위 밖"),
                            "남은후보수": r.get("n_eff", ""), "남은후보_층수": r.get("n_eff_floors", ""), "정답후보_남은후보안": r.get("true_in_eff", ""),
                            "정답층_남은후보안": r.get("truefloor_in_eff", ""), "1위후보_층": r.get("top1_floor", ""), "1위_위치오차_m(정답층일때)": r.get("top1_pos_err_m", ""),
                            "1위_방향오차_도": r.get("top1_yaw_err_deg", ""), "1위_축척오차_퍼센트(정답층일때)": r.get("top1_scale_err_pct", ""),
                            "정답후보_위치오차_m": r.get("true_cand_pos_err_m", ""), "정답후보_축척오차_퍼센트": r.get("true_cand_scale_err_pct", ""),
                            "벽관통_제거된_원후보": r.get("n_raw_removed", "")})
    write("과제시점별_지표.csv", per)
    # 3) 정답이 Top-100 에서 빠진 경우와 사유
    lost = []
    for k in tasks:
        for c, cn in COND:
            prev = D[(c, 0)][k]
            for T in TS:
                r = D[(c, T)][k]
                if hit(prev, 100) and not hit(r, 100):
                    why = "점수에서 밀림"
                    if c in ("T2X", "T3", "T3d", "T3on"):
                        rm = removed_by_crossing(k, c, T)
                        if rm:
                            why = "벽 관통 경로로 제거"
                    lost.append({"과제": ko_task(k), "조건": cn, "시점_초": T, "앞시점_정답순위": prev["true_rank"], "이시점_정답순위": r["true_rank"].replace(">1000", "1000위 밖"), "사유": why})
                prev = r
    write("정답후보_사라진사유.csv", lost if lost else [{"과제": "", "조건": "", "시점_초": "", "앞시점_정답순위": "", "이시점_정답순위": "", "사유": "없음"}])
    # 4) 실제 오돔 끊김
    real = []
    for k in tasks:
        t = EV.TASK[k]; ts = float(t["start_sec"])
        for g in REAL_GAPS.get(t["floor"], []):
            if ts <= g <= ts + 60:
                real.append({"과제": ko_task(k), "출발시각_초": ts, "끊김시각_초": g, "출발뒤_초": round(g - ts, 1),
                             **{f"{cn}_60초_정답순위": D[(c, 60)][k]["true_rank"] for c, cn in COND[:5]}})
    write("실제오돔끊김.csv", real if real else [{"과제": "없음"}])
    acc = []
    for k in tasks:
        for T in (15, 30, 45, 60):
            st = accum_stats(k, T)
            acc.append({"과제": ko_task(k), "시점_초": T, "관측_장수": st["frames"], "라이다_구조점_개수": st["points"], "걸은거리_m": st["dist_m"],
                        "T2_정답순위": D[("T2", T)][k]["true_rank"].replace(">1000", "1000위 밖"), "T3_정답순위": D[("T3", T)][k]["true_rank"].replace(">1000", "1000위 밖")})
    write("누적관측_통계.csv", acc)
    figures(D, tasks)
    print("끝")


def write(name, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def figures(D, tasks):
    X = (0,) + TS
    hit = lambda r, k: int(rk(r["true_rank"]) is not None and rk(r["true_rank"]) <= k)
    # 그림1 결과판: 행 = 조건 × 시점(T0~60), 열 = 출발점
    keys = [(c, T) for c, _ in COND[:5] for T in X]
    M = np.array([[rk(D[k][t]["true_rank"]) or 5000 for t in tasks] for k in keys], float)
    fig, a = plt.subplots(figsize=(20, 0.32 * len(keys) + 2.5))
    colr = lambda v: EV.GREEN if v <= 10 else ("#7fd3b2" if v <= 100 else (EV.YELLOW if v <= 1000 else "#e9ebee"))
    for i in range(len(keys)):
        for j in range(len(tasks)):
            a.add_patch(plt.Rectangle((j, i), 0.92, 0.85, color=colr(M[i, j])))
            if M[i, j] <= 10:
                a.text(j + 0.46, i + 0.43, f"{int(M[i, j])}", ha="center", va="center", fontsize=6.5, color=EV.INK)
    for i in range(0, len(keys), len(X)):
        a.axhline(i - 0.08, color=EV.INK2, lw=1)
    a.set_xlim(0, len(tasks)); a.set_ylim(len(keys), 0)
    lab = {c: cn for c, cn in COND}
    a.set_yticks(np.arange(len(keys)) + 0.43); a.set_yticklabels([f"{lab[c].split(' ')[0]} {T}초 ({int((M[i] <= 100).sum())})" for i, (c, T) in enumerate(keys)], fontsize=8.5)
    FLs = [EV.TASK[t]["floor"] for t in tasks]; xt = []
    for F in dict.fromkeys(FLs):
        j = [i for i, x in enumerate(FLs) if x == F]; xt.append((np.mean(j) + 0.46, EV.fl(F))); a.axvline(j[0] - 0.04, color=EV.SURF, lw=3)
    a.set_xticks([x for x, _ in xt]); a.set_xticklabels([n for _, n in xt], fontsize=10); a.tick_params(length=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title("E3 결과판: 칸 = 출발점(층마다 5칸), 행 = 조건·시점(괄호 = Top-100 안 수) · 진초록 Top-10(숫자 = 순위) · 연초록 Top-100 · 노랑 1000위 안 · 회색 밖\n"
                "T1 이동만 / T2 시간 누적 / T2X +벽 관통 제거 / T2S +공간 감점 / T3 물리 조건(둘 다)", loc="left", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "그림1_조건시점별_결과판.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 그림2 시간에 따른 Top-K 안 정답 위치·정답 후보 유지
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True)
    for a, (title, fn) in zip(ax, (("Top-10 안 정답 위치", lambda r: hit(r, 10)), ("Top-100 안 정답 위치", lambda r: hit(r, 100)),
                                   ("정답 후보가 남은 후보(1위와 점수 차, 관측 6장 환산 1.5 이하) 안", lambda r: int(r.get("true_in_eff") or 0)))):
        for c, cn in COND:
            Xc = times_of(c); y = [100 * np.mean([fn(D[(c, T)][k]) for k in tasks]) if (T or "남은" not in title) else np.nan for T in Xc]
            a.plot(Xc, y, marker="o", ms=7, lw=2, color=COLOR[c], ls="--" if c.endswith("d") else "-", label=cn, markeredgecolor=EV.SURF, markeredgewidth=1.5)
        a.set_xticks(X); a.set_xlabel("출발 뒤 시간(초)"); a.grid(alpha=.3); a.set_title(title, loc="left", fontsize=12.5, fontweight="bold"); a.set_ylim(0, 100)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("출발점 비율 (%)"); ax[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(f"걸으면서 정답이 올라오나 ({len(tasks)}곳, 0초 = E2 제자리 한 번 관측)", x=0.01, ha="left", fontsize=15, fontweight="bold"); fig.tight_layout()
    fig.savefig(OUT / "그림2_시간별_정답포함.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 그림3 출발점별 시점 순위표(2026-10-11, 선 겹친 그림 대신): 행 = 출발점(층별 묶음), 열 = T2·T3 의 0·15·30·45·60초, 칸 = 정답 순위
    cols = [(c, T) for c in ("T2", "T3") for T in X]
    M = np.array([[rk(D[(c, T)][k]["true_rank"]) or 5000 for (c, T) in cols] for k in tasks], float)
    fig, a = plt.subplots(figsize=(13, 0.36 * len(tasks) + 2.2))
    colr = lambda v: EV.GREEN if v <= 10 else ("#7fd3b2" if v <= 100 else (EV.YELLOW if v <= 1000 else "#e9ebee"))
    for i_ in range(len(tasks)):
        for j_ in range(len(cols)):
            x0 = j_ + (0.6 if j_ >= len(X) else 0)
            a.add_patch(plt.Rectangle((x0, i_), 0.92, 0.84, color=colr(M[i_, j_])))
            v_ = M[i_, j_]; a.text(x0 + 0.46, i_ + 0.42, f"{int(v_)}" if v_ <= 1000 else "밖", ha="center", va="center", fontsize=8.5,
                                  color=EV.INK, fontweight="bold" if v_ <= 100 else "normal")
    FLs = [EV.TASK[k]["floor"] for k in tasks]
    for F in dict.fromkeys(FLs):
        j_ = [i for i, x in enumerate(FLs) if x == F]
        if j_[0]:
            a.axhline(j_[0] - 0.08, color=EV.INK2, lw=1)
    a.set_xlim(-0.1, len(cols) + 0.7); a.set_ylim(len(tasks), -1.3)
    a.set_yticks(np.arange(len(tasks)) + 0.42); a.set_yticklabels([f"{ko_task(k)} · {EV.TYPE[EV.TASK[k]['start_type']]}" for k in tasks], fontsize=8.5)
    a.set_xticks([j_ + (0.6 if j_ >= len(X) else 0) + 0.46 for j_ in range(len(cols))]); a.set_xticklabels([f"{T}초" for _, T in cols], fontsize=9.5)
    a.xaxis.tick_top(); a.tick_params(length=0)
    a.text(len(X) / 2, -1.0, "T2 시간 누적", ha="center", fontsize=12, fontweight="bold"); a.text(len(X) + 0.6 + len(X) / 2, -1.0, "T3 물리 조건(관통 제거 + 공간 감점)", ha="center", fontsize=12, fontweight="bold")
    for sp in a.spines.values():
        sp.set_visible(False)
    n_ = lambda c, T: int(sum(1 for k in tasks if (rk(D[(c, T)][k]["true_rank"]) or 5000) <= 100))
    a.set_title(f"출발점별 정답 위치 순위({len(tasks)}곳) · 0초 = E2 제자리 한 번 · 진초록 Top-10 · 연초록 Top-100 · 노랑 1000위 안 · 회색 밖\n"
                f"Top-100 안: T2 " + " → ".join(str(n_("T2", T)) for T in X) + " · T3 " + " → ".join(str(n_("T3", T)) for T in X), loc="left", fontsize=12, fontweight="bold", pad=34)
    fig.savefig(OUT / "그림3_출발점별_시점순위표.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    # 그림4 남은 후보 수·층 수
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for c, cn in COND:
        for a, key in zip(ax, ("n_eff", "n_eff_floors")):
            Xc = times_of(c)[1:]; y = [np.median([float(D[(c, T)][k][key]) for k in tasks]) for T in Xc]
            a.plot(Xc, y, marker="o", ms=7, lw=2, color=COLOR[c], ls="--" if c.endswith("d") else "-", label=cn, markeredgecolor=EV.SURF)
    ax[0].set_title("남은 후보 수 (1위와 점수 차가 관측 6장 환산 1.5 이하, 중앙값)", loc="left", fontsize=12, fontweight="bold"); ax[0].set_yscale("log")
    ax[1].set_title("남은 후보가 걸친 층 수 (중앙값)", loc="left", fontsize=12, fontweight="bold")
    for a in ax:
        a.set_xticks(TS); a.set_xlabel("출발 뒤 시간(초)"); a.grid(alpha=.3)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[1].legend(fontsize=9, frameon=False)
    fig.suptitle("걸으면서 후보가 줄어드나", x=0.01, ha="left", fontsize=15, fontweight="bold"); fig.tight_layout()
    fig.savefig(OUT / "그림4_남은후보수.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 그림5 축척: 정답 후보의 축척 오차, 1위(정답 층일 때) 축척 오차
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for c in ("T2", "T3"):
        for a, key, tl in zip(ax, ("true_cand_scale_err_pct", "top1_scale_err_pct"), ("정답 후보(1000위 안)의 축척 오차 |%|", "1위가 정답 층일 때 1위의 축척 오차 %")):
            Xc = times_of(c); ys = [[abs(float(D[(c, T)][k][key])) if key.startswith("true") else float(D[(c, T)][k][key]) for k in tasks if D[(c, T)][k].get(key, "") != ""] for T in Xc]
            a.plot(Xc, [np.median(y) if y else np.nan for y in ys], marker="o", ms=7, lw=2, color=COLOR[c], label=dict(COND)[c])
            a.set_title(tl + " (중앙값)", loc="left", fontsize=12, fontweight="bold")
    for a in ax:
        a.set_xticks(X); a.set_xlabel("출발 뒤 시간(초)"); a.grid(alpha=.3); a.axhline(0, color=EV.INK2, lw=0.8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(frameon=False)
    fig.suptitle("축척이 시간에 따라 안정되나", x=0.01, ha="left", fontsize=15, fontweight="bold"); fig.tight_layout()
    fig.savefig(OUT / "그림5_축척.png", dpi=140, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
