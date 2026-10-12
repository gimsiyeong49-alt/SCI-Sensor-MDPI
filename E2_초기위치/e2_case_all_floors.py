#!/usr/bin/env python3
"""E2 출발점별 Top-100 그림(2026-10-10 새 구성). A 라이다 · 축척 탐색의 알고리즘 3가지(벽 겹침 / + 거리 비교 / + 정밀 검증) + 참고 정답 축척 × 과제 65개.
한 장(사용자가 고른 F4 형식): 그 자리 카메라 화면 / 정답 층 지도 위 Top-100 후보(파랑, 클수록 높은 순위)·실제 위치(★)·정답 후보 동그라미 / 그 건물 전 층 후보 수·최고 순위
결과: _01_E2_초기위치/2_출발점별_Top100/<폴더>/<층>_출발점<n>_<종류>.png, 폴더마다 README.md(과제별 순위·링크), 2_출발점별_Top100/README.md(과제 × 알고리즘 순위표)
    python3 scripts/e2_case_all_floors.py [--cases 3F_T1] [--conds 1_벽겹침,...] [--workers 13]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

G = EV.G; OUT = EV.PKG / "2_출발점별_Top100"
COND = [  # (폴더, 채점 폴더 접미사, 이름) — 2026-10-10 새 구성: A 라이다 · 축척 탐색에서 순위 매기는 법 3가지 + 참고 상한
    ("1_벽겹침", "_nospace", "A 라이다 · 축척 탐색 · 벽 겹침"),
    ("2_거리비교", "_nospace_거리비교", "A 라이다 · 축척 탐색 · + 거리 비교"),
    ("3_정밀검증", "_nospace_정밀검증", "A 라이다 · 축척 탐색 · + 정밀 검증"),
    ("참고_정답축척_정밀검증", "_oracle_nospace_정밀검증", "참고: A 라이다 · 정답 축척 · + 정밀 검증"),
]
rk = lambda v: int(v) if str(v).isdigit() else None
NMAP = {"kci": 9, "gong": 7, "sw": 8}


def load(sfx):
    d = G / f"E2_dense_T0{sfx}"
    res = {r["trial_id"]: r for r in csv.DictReader(open(d / "E2_top100_per_trial.csv"))}; top = {}
    for r in csv.DictReader(open(d / "E2_top100_candidates.csv")):
        top.setdefault(r["trial_id"], []).append(r)
    return res, top


def draw(task, cname, res, top, out, cache):
    """보내준 F4 형식: 카메라 화면 | 정답 층 지도 위 Top-100 후보·실제 위치 | 그 건물 전 층 후보 수·최고 순위."""
    t = EV.TASK[task]; F, b = t["floor"], t["building"]; m_true = int(EV.GT[task]["true_map"]); n = NMAP[b]
    tp = top.get(task, []); r = res[task]; v = rk(r["true_rank"]); ok = v is not None and v <= 100
    if task not in cache:
        cache[task] = (EV.gt_of(task), EV.frame(F, float(t["start_sec"])))
    (p_gt, ppm_gt, hd_gt, _, _), img = cache[task]
    fig = plt.figure(figsize=(18.5, 5.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.1, 0.85], wspace=0.16, top=0.8, bottom=0.02)
    a = fig.add_subplot(gs[0]); a.imshow(img); a.axis("off")
    a.set_title(f"그 자리 카메라 화면\n{EV.fl(F)} {EV.TYPE.get(t['start_type'], '')} · {float(t['start_sec']):.0f}초", loc="left", fontsize=12)
    a = fig.add_subplot(gs[1]); im = EV.mapimg(b, m_true); a.imshow(im, cmap="gray", vmin=0, vmax=255); EV.crop(a, im)
    on = [x for x in tp if int(x["floor_id"]) == m_true]
    for x in sorted(on, key=lambda x: -int(x["rank"])):
        kk = int(x["rank"]); a.scatter(float(x["x_px"]), float(x["y_px"]), s=max(22, 170 - 1.5 * kk), color=EV.BLUE, alpha=0.75, edgecolor=EV.SURF, linewidth=0.6, zorder=3)
    a.scatter(*p_gt, marker="*", s=460, color=EV.RED, edgecolor=EV.SURF, linewidth=1, zorder=5)
    if ok:                                           # 정답 후보(정답 층 + 3 m + 30° 안 첫 후보)에 동그라미
        tc = next((x for x in sorted(on, key=lambda x: int(x["rank"])) if int(x["rank"]) == v), None)
        if tc:
            a.scatter(float(tc["x_px"]), float(tc["y_px"]), s=700, facecolor="none", edgecolor=EV.GREEN, linewidth=2.5, zorder=6)
            a.annotate(f"정답 후보 {v}위", (float(tc["x_px"]), float(tc["y_px"])), (14, -18), textcoords="offset points", fontsize=11, fontweight="bold",
                       color=EV.GREEN, bbox=dict(facecolor=EV.SURF, edgecolor=EV.GREEN, pad=2), zorder=7)
    a.set_title(f"정답 층({m_true}층) 지도: Top-100 중 이 층 후보 {len(on)}개 (파랑, 클수록 높은 순위) · ★ 실제 위치", loc="left", fontsize=12)
    a = fig.add_subplot(gs[2])
    cnt = [sum(int(x["floor_id"]) == m for x in tp) for m in range(1, n + 1)]
    best = [min([int(x["rank"]) for x in tp if int(x["floor_id"]) == m] or [0]) for m in range(1, n + 1)]
    y = np.arange(n)[::-1]
    a.barh(y, cnt, color=[EV.GREEN if m == m_true else "#b8bec7" for m in range(1, n + 1)], height=0.62)
    xm = max(max(cnt), 1)
    for yi, c, bb, m in zip(y, cnt, best, range(1, n + 1)):
        a.text(c + xm * 0.03, yi, f"{c}개 · 가장 높은 {bb}위" if c else "후보 없음", va="center", fontsize=10,
               color=EV.INK if c else EV.INK2, fontweight="bold" if m == m_true else "normal")
    a.set_yticks(y); a.set_yticklabels([f"{m}층" + (" ★" if m == m_true else "") for m in range(1, n + 1)], fontsize=10.5)
    for lab, m in zip(a.get_yticklabels(), range(1, n + 1)):
        lab.set_color(EV.GREEN if m == m_true else EV.INK); lab.set_fontweight("bold" if m == m_true else "normal")
    a.set_xlim(0, xm * 1.9); a.set_xticks([]); a.set_ylim(-0.6, n - 0.4)
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("층별 Top-100 후보 수 (★ 실제 층)", loc="left", fontsize=12)
    head = f"E2 Top-100 · {EV.fl(F)} 출발점{task.split('_T')[1]} ({EV.TYPE.get(t['start_type'], '')}) · {cname}"
    res_txt = (f"성공: 정답 위치가 {v}위 (Top-100 안)" if ok else ("실패: 정답 위치가 " + (f"{v}위" if v else "1000위 밖") + " (Top-100 밖)")) + \
        f"   ·   정답 층의 가장 높은 후보 {r['floor_rank']}위   ·   1위 후보는 {r.get('top1_floor', '')}층"
    fig.text(0.01, 0.97, head, fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.885, res_txt, fontsize=13, fontweight="bold", va="top", color=EV.GREEN if ok else EV.RED)
    out.mkdir(parents=True, exist_ok=True); fig.savefig(out / f"{EV.ko_task(task)}.png", dpi=120, bbox_inches="tight"); plt.close(fig)


def _job(arg):
    cases, conds = arg
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    data = {c[0]: load(c[1]) for c in COND if c[0] in conds}; cache = {}
    for task in cases:
        for folder, sfx, name in COND:
            if folder in conds:
                draw(task, name, *data[folder], OUT / folder, cache)
        print(task, flush=True)


def index(conds):
    data = {c[0]: load(c[1])[0] for c in COND if c[0] in conds}
    fmt = lambda v: (f"**{v}**" if rk(v) <= 100 else v) if rk(v) else "1000 밖"
    for folder, sfx, name in COND:
        if folder not in conds:
            continue
        res = data[folder]; n = sum(1 for r in res.values() if rk(r["true_rank"]) and rk(r["true_rank"]) <= 100)
        L = [f"# {name}", "", f"출발점마다 한 장. Top-100 안 정답 위치 {n}/65. 굵은 순위가 Top-100 안.", "", "| 과제 | 출발점 | 정답 위치 순위 | 정답 층 최고 순위 | 1위 후보 층 | 그림 |", "|---|---|---|---|---|---|"]
        for task, t in EV.TASK.items():
            r = res[task]; L.append(f"| {EV.fl(t['floor'])} 출발점{task.split('_T')[1]} | {EV.TYPE.get(t['start_type'], '')} | {fmt(r['true_rank'])} | {r['floor_rank']} | {r.get('top1_floor', '')} | [보기]({EV.ko_task(task)}.png) |")
        (OUT / folder / "README.md").write_text("\n".join(L) + "\n")
    L = ["# E2 출발점별 Top-100 그림", "", "모두 A 라이다(제자리 라이다 한 번). 폴더 하나가 순위 매기는 법 하나, 그림 하나가 출발점 하나다. 표의 숫자는 정답 위치 순위(굵은 것이 Top-100 안), 링크는 그 그림이다.", "",
         "| 조건 | 폴더 | Top-100 안 |", "|---|---|---|"]
    for folder, sfx, name in COND:
        if folder in conds:
            n = sum(1 for r in data[folder].values() if rk(r["true_rank"]) and rk(r["true_rank"]) <= 100)
            L.append(f"| {name} | [{folder}/]({folder}/) | {n}/65 |")
    main = [c for c in COND if c[0] in conds]
    L += ["", "## 과제별 정답 위치 순위", "", "| 과제 | 출발점 | " + " | ".join(c[2].split(" · ")[-1] if not c[0].startswith("참고") else "참고: 정답 축척+정밀 검증" for c in main) + " |", "|---|---|" + "---|" * len(main)]
    for task, t in EV.TASK.items():
        L.append(f"| {EV.fl(t['floor'])} 출발점{task.split('_T')[1]} | {EV.TYPE.get(t['start_type'], '')} | " + " | ".join(f"[{fmt(data[c[0]][task]['true_rank'])}]({c[0]}/{EV.ko_task(task)}.png)" for c in main) + " |")
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default=""); ap.add_argument("--conds", default=""); ap.add_argument("--workers", type=int, default=13); a = ap.parse_args()
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    conds = a.conds.split(",") if a.conds else [c[0] for c in COND]
    cases = a.cases.split(",") if a.cases else list(EV.TASK)
    groups = {}
    for c in cases:
        groups.setdefault(EV.TASK[c]["floor"], []).append(c)
    from multiprocessing import Pool
    with Pool(min(a.workers, len(groups))) as pool:
        pool.map(_job, [(g, conds) for g in groups.values()], chunksize=1)
    if not a.cases:
        index(conds)


if __name__ == "__main__":
    main()
