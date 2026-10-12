#!/usr/bin/env python3
"""E2 알고리즘 비교 사례 그림(2026-10-10 새 구성). 과제마다 한 장:
  윗줄: 그 자리 카메라 화면 / 그 순간 라이다 벽 점(위에서 본 모양) / 알고리즘별 정답 위치 순위
  아랫줄: 정답 층 지도 위 Top-100 후보 — A 라이다 · 축척 탐색의 벽 겹침 / + 거리 비교 / + 정밀 검증, 참고 정답 축척 + 정밀 검증
결과: _01_E2_초기위치/3_조건비교/<층>_출발점<n>_<종류>.png, 같은 폴더 README.md(과제별 순위·링크)
    python3 scripts/e2_compare_cases.py [--cases 5F_T3,...]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402  (글꼴·색·지도·라이다 표본·정답 위치 함수)
from e2_scene_labels import depth_dist, DEP  # noqa: E402

G = EV.G; OUT = EV.PKG / "3_조건비교"
COND = [("_nospace", "벽 겹침"), ("_nospace_거리비교", "+ 거리 비교"), ("_nospace_정밀검증", "+ 정밀 검증"),
        ("_oracle_nospace_정밀검증", "참고: 정답 축척 + 정밀 검증")]     # 모두 A 라이다, 앞 셋은 축척 탐색(2026-10-10 새 구성)
SCN = {"corridor": "복도", "junction": "갈림길", "dead-end": "막다른 곳"}
rk = lambda v: int(v) if str(v).isdigit() else None


def load():
    res, top = {}, {}
    for sfx, _ in COND:
        d = G / f"E2_dense_T0{sfx}"
        res[sfx] = {r["trial_id"]: r for r in csv.DictReader(open(d / "E2_top100_per_trial.csv"))}
        top[sfx] = {}
        for r in csv.DictReader(open(d / "E2_top100_candidates.csv")):
            top[sfx].setdefault(r["trial_id"], []).append(r)
    return res, top


def depth_panel(a, F, t, lab, used):
    d = np.load(DEP / f"{F}_full.npz"); keys = np.array(sorted(float(k) for k in d.files))
    use = keys[(keys >= t - 1.5) & (keys <= t)]; k = use[-1] if len(use) else keys[int(np.argmin(abs(keys - t)))]
    dep = d[f"{k:.1f}"].astype(float) / 1000.0; Hh, Wd = dep.shape
    f, l, r = (float(used[c]) for c in ("depth_fwd_m", "depth_left_m", "depth_right_m"))   # 장면 판단에 쓴 값(1.5초 안 프레임 중앙값)
    a.imshow(np.where(dep == 0, np.nan, dep), cmap="viridis_r", vmin=0, vmax=12)
    for x0, x1, name, v in ((0, Wd // 4, "왼쪽", l), (2 * Wd // 5, 3 * Wd // 5, "앞", f), (3 * Wd // 4, Wd, "오른쪽", r)):
        a.add_patch(Rectangle((x0, Hh // 3), x1 - x0, Hh // 3, fill=False, ec="white", lw=2))
        a.text((x0 + x1) / 2, Hh // 3 - 8, f"{name} {v:.1f} m", ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
               bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))
    a.axis("off")
    n = len(str(used["depth_keys"]).split(","))
    a.set_title(f"② 깊이 카메라 → 장면 '{SCN[lab]}'" + (f"\n   (거리 = 1.5초 안 프레임 {n}장 중앙값, 그림은 마지막 장)" if n > 1 else ""), loc="left", fontsize=12.5, fontweight="bold")


def case(task, res, top, scn):
    t = EV.TASK[task]; F, b = t["floor"], t["building"]; m_true = int(EV.GT[task]["true_map"]); ts = float(t["start_sec"]); s = scn[task]
    p_gt = EV.gt_of(task)[0]; pts, _ = EV.obs(task)
    fig = plt.figure(figsize=(22, 15)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.25, wspace=0.05)
    a = fig.add_subplot(gs[0, 0]); a.imshow(EV.frame(F, ts)); a.axis("off")
    a.set_title(f"① 카메라 화면 ({EV.fl(F)} {EV.TYPE.get(t['start_type'], '')}, {ts:.0f}초)", loc="left", fontsize=12.5, fontweight="bold")
    a = fig.add_subplot(gs[0, 1]); a.scatter(-pts[:, 1], pts[:, 0], s=1.5, color=EV.INK, alpha=0.6); a.scatter(0, 0, marker="^", s=160, color=EV.RED, zorder=3)
    for rr in (4.5, 9):
        a.add_patch(plt.Circle((0, 0), rr, fill=False, ls=":", color=EV.GRAY, lw=1)); a.text(rr * 0.71, rr * 0.71, f"{rr:g} m", color=EV.INK2, fontsize=9)
    a.set_aspect("equal"); a.set_xlim(-14, 14); a.set_ylim(-10, 16); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"② 그 순간 라이다 벽 점 {len(pts):,}개 (▲ 로봇)", loc="left", fontsize=12.5, fontweight="bold")
    a = fig.add_subplot(gs[0, 2:4]); a.axis("off")
    lines = [f"정답 위치 순위 (A 라이다, 같은 스캔 한 번)"]
    for sfx, name in COND:
        v = res[sfx][task]["true_rank"]; lines.append(f"{name}\n    → {v if rk(v) else '1000 밖'}{'위' if rk(v) else ''}")
    a.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=12.5, linespacing=1.55, transform=a.transAxes)
    a.text(0.02, 0.04, "벽 겹침 = 스캔 점이 지도 벽과 겹치나 · + 거리 비교 = 360° 방향별 벽 거리가 맞나\n+ 정밀 검증 = 위치·방향·축척 미세 조정 뒤 벽에서 먼 점·건물 밖·안 보인 벽 확인",
           fontsize=11, color=EV.INK2, transform=a.transAxes, linespacing=1.6)
    im = EV.mapimg(b, m_true)
    for i, (sfx, name) in enumerate(COND):
        rr, cc = divmod(i, 2); a = fig.add_subplot(gs[1 + rr, 2 * cc:2 * cc + 2]); a.imshow(im, cmap="gray", vmin=0, vmax=255); EV.crop(a, im)
        tp = top[sfx].get(task, []); on = [x for x in tp if int(x["floor_id"]) == m_true]
        for x in sorted(on, key=lambda x: -int(x["rank"])):
            kk = int(x["rank"]); a.scatter(float(x["x_px"]), float(x["y_px"]), s=max(24, 200 - 1.7 * kk), color=EV.BLUE, alpha=0.8, edgecolor=EV.SURF, linewidth=0.5, zorder=3)
        a.scatter(*p_gt, marker="*", s=520, color=EV.RED, edgecolor=EV.SURF, linewidth=1, zorder=5)
        v = res[sfx][task]["true_rank"]; ok = rk(v) and rk(v) <= 100
        f1 = int(sorted(tp, key=lambda x: int(x["rank"]))[0]["floor_id"]) if tp else 0
        a.set_title(f"{name}\n정답 {v if rk(v) else '1000 밖'}{'위' if rk(v) else ''} · Top-100 중 이 층 {len(on)}개 · 다른 층 {len(tp) - len(on)}개 · 1위 후보 {f1}층", loc="left", fontsize=13,
                    fontweight="bold", color=EV.GREEN if ok else EV.INK)
    fig.suptitle(f"E2 알고리즘 비교 — {EV.fl(F)} 출발점{task.split('_T')[1]} ({EV.TYPE.get(t['start_type'], '')}) · 제자리 한 번 관측 · 아랫줄 = 정답 층 지도, 파랑 = Top-100 중 이 층에 있는 후보(나머지는 다른 층), ★ = 실제 위치",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", y=0.995)
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / f"{EV.ko_task(task)}.png", dpi=120, bbox_inches="tight"); plt.close(fig); print(task, flush=True)


def index(res, scn):
    L = ["# E2 알고리즘 비교 사례 (제자리 한 번 관측, 65곳)", "",
         "과제마다 한 장. 윗줄은 카메라 화면과 그 순간 라이다 벽 점, 아랫줄은 같은 스캔으로 순위 매기는 법만 바꾼 4가지의 정답 층 지도 위 Top-100 후보와 실제 위치(★)다(A 라이다). 숫자는 정답 위치 순위, 굵은 것이 Top-100 안.", "",
         "| 과제 | 출발점 | 벽 겹침 | + 거리 비교 | + 정밀 검증 | 참고: 정답 축척 + 정밀 검증 | 그림 |", "|---|---|---|---|---|---|---|"]
    fmt = lambda v: (f"**{v}**" if rk(v) <= 100 else v) if rk(v) else "1000 밖"
    for task, t in EV.TASK.items():
        L.append(f"| {EV.fl(t['floor'])} 출발점{task.split('_T')[1]} | {EV.TYPE.get(t['start_type'], '')} | "
                 + " | ".join(fmt(res[sfx][task]["true_rank"]) for sfx, _ in COND) + f" | [보기]({EV.ko_task(task)}.png) |")
    n = lambda sfx: sum(1 for r in res[sfx].values() if rk(r["true_rank"]) and rk(r["true_rank"]) <= 100)
    L += ["", "Top-100 안 합계: " + " · ".join(f"{name} {n(sfx)}/65" for sfx, name in COND)]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def _init():
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    res, top = load(); scn = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/E2_scene_labels.csv"))}
    return res, top, scn


def _floor_job(cases):
    res, top, scn = _init()
    for c in cases:
        case(c, res, top, scn)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default=""); ap.add_argument("--workers", type=int, default=13); a = ap.parse_args()
    res, top, scn = _init()
    cases = a.cases.split(",") if a.cases else list(EV.TASK)
    groups = {}
    for c in cases:                                  # 같은 층끼리 한 작업자(위치추정 데이터 한 번만 읽기)
        groups.setdefault(EV.TASK[c]["floor"], []).append(c)
    from multiprocessing import Pool
    with Pool(min(a.workers, len(groups))) as pool:
        pool.map(_floor_job, list(groups.values()), chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True); index(res, scn)


if __name__ == "__main__":
    main()
