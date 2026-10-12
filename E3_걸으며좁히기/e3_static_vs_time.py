#!/usr/bin/env python3
"""E3 시점별 '정지 대비' 그림·통계(2026-10-11). 15·30·45·60초를 시점마다 따로, 각 그림은 정지(0초 = E2 제자리 한 번)와 그 시점을 나란히.
후보군 비교: 정지 Top-100 후보마다 그 가설대로 걸은 만큼 옮긴 뒤(위치추정 코드와 같은 식: ppm·_A(φ)·(오돔_T − 오돔_0)),
T초 Top-100 과 같은 층·2 m(후보 축척)·30° 안이면 '유지', T초 후보 중 맞는 정지 후보가 없으면 '새 후보'. 정답은 채점·표시에만.
한 장: ① 정지 1위 자세로 얹은 스캔 한 장 ② T초 1위 자세로 얹은 쌓인 라이다 ③ 정답 층 지도: 정지 후보(옮긴 자리, 회색)·T초 후보(파랑)·정답 후보·실제 위치·지나온 길 ④ 층별 후보 수(정지 대 T초)
결과: reports/_03_E3_걸으며좁히기/2_시점별_정지대비/<조건>/<T>초/<층>_출발점<n>_<종류>.png, 폴더마다 README.md
      reports/_03_E3_걸으며좁히기/3_결과표와_그림/정지대비_시점별_통계.csv, 정지대비_출발점별.csv, 정지대비_층별.csv, 정지대비_장소종류별.csv
    python3 scripts/e3_static_vs_time.py [--cases 4F_T1] [--conds T2,T3] [--workers 8] [--no-figs]
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402
import e3_report as E3  # noqa: E402

OUT = E3.PKG; TAB = E3.PKG / "3_결과표와_그림"      # 2026-10-11 폴더 정리: 조건 폴더가 패키지 바로 아래
FOLDER = {"T3on": "2_비교/3_전칸누적_축척걸으며추정", "T2": "2_비교/4_조건쪼개기_축척고정/T2_시간누적", "T3": "2_비교/4_조건쪼개기_축척고정/T3_물리조건"}
TAG = {"T3on": "전칸누적_축척추정", "T2": "T2_시간누적", "T3": "T3_물리조건"}        # 표 파일 이름용
TS = (15, 30, 45, 60); NMAP = {"kci": 9, "gong": 7, "sw": 8}
rk = E3.rk; wrap = lambda a: (a + 180) % 360 - 180


def subpath(k):
    """그림 자리: <건물>/<층>/출발점<n>_<종류>.png (2026-10-11, 시점 폴더 안을 건물·층으로 나눔)."""
    t = EV.TASK[k]; fl = t["floor"].lstrip("GS").replace("F", "층")
    return Path(EV.BNAME[t["building"]]) / fl, f"출발점{k.split('_T')[1]}_{EV.TYPE[t['start_type']].replace(' ', '').replace('·', '')}.png"


def cands(c, T):
    p = (E3.E2W / E3.T0SRC[c] if T == 0 else E3.W / "score" / f"{c}_T{T}") / "E2_top100_candidates.csv"
    out = {}
    for r in csv.DictReader(open(p)):
        if int(r["rank"]) <= 100:
            out.setdefault(r["trial_id"], []).append(r)
    return out


def odom_disp(task, T):
    """출발 관측 프레임 → T초 마지막 프레임의 오돔 이동(m, 오돔 좌표)과 관측 장수·점 수·걸은 거리."""
    import e3_accum_registered as AR
    Z = AR._loc(task); L = Z.L; ts = float(EV.TASK[task]["start_sec"])
    buf = Z.window_scans(L.T[0] + ts + T, T, 3.0)
    d = np.asarray(buf[-1]["xy"], float) - np.asarray(buf[0]["xy"], float)
    sel = (L.T >= L.T[0] + ts) & (L.T <= L.T[0] + ts + T); x, y = np.asarray(L.X)[sel], np.asarray(L.Y)[sel]
    return d, len(buf), float(np.hypot(np.diff(x), np.diff(y)).sum())


def compare(c0, cT, disp):
    """정지 후보를 옮긴 자리와 T초 후보 맞대기 → (옮긴 정지 후보 [(m,x,y,hd,s,유지여부)], T초 후보 유지 여부 목록)."""
    from rt_localizer import _A
    moved = []
    for r in c0:
        m, x, y, hd, s = int(r["floor_id"]), float(r["x_px"]), float(r["y_px"]), float(r["heading_deg"]), float(r["ppm"])
        off = s * (_A(math.radians(hd - 228.0)) @ disp); moved.append([m, x + off[0], y + off[1], hd, s, False])
    keptT = []
    for r in cT:
        m, x, y, hd, s = int(r["floor_id"]), float(r["x_px"]), float(r["y_px"]), float(r["heading_deg"]), float(r["ppm"]); hit = False
        for q in moved:
            if q[0] == m and math.hypot(x - q[1], y - q[2]) / min(s, q[4]) <= 2.0 and abs(wrap(hd - q[3])) <= 30:
                q[5] = True; hit = True
        keptT.append(hit)
    return moved, keptT


def scan_panel(a, task, T, top1, b, m, pg, ppm_gt, gt, gx, ts, label):
    import e3_accum_registered as AR
    m1 = int(top1["floor_id"]); s1 = float(top1["ppm"])
    P, pos, nf, npt, _ = AR.registered(task, T, float(top1["x_px"]), float(top1["y_px"]), float(top1["heading_deg"]), s1)
    a.imshow(EV.mapimg(b, m1), cmap="gray", vmin=0, vmax=255)
    a.scatter(P[:, 0], P[:, 1], s=1.4 if nf > 1 else 5, color=EV.ORANGE, alpha=0.55 if nf > 1 else 0.9, linewidths=0, zorder=3)
    if len(pos) > 1:
        a.plot(pos[:, 0], pos[:, 1], color=EV.INK, lw=1.8, zorder=4)
    a.scatter(*pos[-1], marker="o", s=120, color=EV.BLUE, edgecolor=EV.SURF, zorder=6)
    same = m1 == m; d1 = float(np.hypot(*(pos[-1] - pg)) / ppm_gt) if same else None
    if same:
        sel = (gt >= ts) & (gt <= ts + T)
        if sel.sum() > 1:
            a.plot(gx[sel, 0], gx[sel, 1], color=EV.RED, lw=1.4, ls="--", zorder=4)
        a.scatter(*pg, marker="*", s=280, color=EV.RED, edgecolor=EV.SURF, zorder=7)
    box = np.vstack([pos, pg[None]]) if same else pos; r_ = 13 * s1
    a.set_xlim(box[:, 0].min() - r_, box[:, 0].max() + r_); a.set_ylim(box[:, 1].max() + r_, box[:, 1].min() - r_); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    col = EV.GREEN if (same and d1 <= 3.0) else (EV.ORANGE if same else EV.RED)
    for sp in a.spines.values():
        sp.set_color(col); sp.set_linewidth(2.5)
    a.set_title(f"{label} · 1위 후보 {m1}층" + (" = 정답 층" if same else " (다른 층)") + (f" · 실제 위치에서 {d1:.1f} m" if same else "") +
                f"\n관측 {nf}장 · 점 {npt:,}개 · 벽까지 평균 {EV.wall_dist(b, m1, P, s1):.2f} m", loc="left", fontsize=11, color=col, fontweight="bold")


def draw(task, c, T, D, C, st, od):
    t = EV.TASK[task]; F, b = t["floor"], t["building"]; m = int(EV.GT[task]["true_map"]); ts = float(t["start_sec"]); n = NMAP[b]
    _, ppm_gt, _, gt, gx = EV.gt_of(task); p0 = gx[int(np.argmin(abs(gt - ts)))]; pT = gx[int(np.argmin(abs(gt - (ts + T))))]
    c0 = sorted(C[(c, 0)].get(task, []), key=lambda x: int(x["rank"])); cT = sorted(C[(c, T)].get(task, []), key=lambda x: int(x["rank"]))
    v0, vT = rk(D[(c, 0)][task]["true_rank"]), rk(D[(c, T)][task]["true_rank"])
    fig = plt.figure(figsize=(26, 7.6)); gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.35, 0.62], wspace=0.1, top=0.72, bottom=0.03)
    scan_panel(fig.add_subplot(gs[0]), task, 0, c0[0], b, m, p0, ppm_gt, gt, gx, ts, "정지(0초)")
    scan_panel(fig.add_subplot(gs[1]), task, T, cT[0], b, m, pT, ppm_gt, gt, gx, ts, f"{T}초")
    # ③ 정답 층 지도: 정지 후보(옮긴 자리) 대 T초 후보
    a = fig.add_subplot(gs[2]); im = EV.mapimg(b, m); a.imshow(im, cmap="gray", vmin=0, vmax=255)
    mv = [q for q in st["moved"] if q[0] == m]
    if mv:
        a.scatter([q[1] for q in mv], [q[2] for q in mv], s=70, facecolor="none", edgecolor="#8b929c", linewidth=1.3, zorder=3)
    onT = [(r, k) for r, k in zip(cT, st["keptT"]) if int(r["floor_id"]) == m]
    for r, k in onT:
        a.scatter(float(r["x_px"]), float(r["y_px"]), s=46, color=EV.BLUE if not k else EV.INK, alpha=0.85, edgecolor=EV.SURF, linewidth=0.4, zorder=4)
    sel = (gt >= ts) & (gt <= ts + T); a.plot(gx[sel, 0], gx[sel, 1], color=EV.INK, lw=1.6, zorder=5)
    a.scatter(*p0, marker="o", s=90, color=EV.SURF, edgecolor=EV.RED, linewidth=2, zorder=6); a.scatter(*pT, marker="*", s=320, color=EV.RED, edgecolor=EV.SURF, zorder=7)
    if vT and vT <= 100:
        tc = next((r for r in cT if int(r["rank"]) == vT), None)
        if tc:
            a.scatter(float(tc["x_px"]), float(tc["y_px"]), s=440, facecolor="none", edgecolor=EV.GREEN, linewidth=2.4, zorder=8)
    EV.crop(a, im, pad=30); a.axis("off")
    a.set_title(f"정답 층({m}층) 지도: 정지 후보 {len(mv)}개 → {T}초 후보 {len(onT)}개\n(남은 {sum(k for _, k in onT)}개 · 새로 든 {sum(not k for _, k in onT)}개)", loc="left", fontsize=11.5, fontweight="bold")
    # ④ 층별 후보 수
    a = fig.add_subplot(gs[3]); y = np.arange(n)[::-1]
    n0 = [sum(int(r["floor_id"]) == k for r in c0) for k in range(1, n + 1)]; nT = [sum(int(r["floor_id"]) == k for r in cT) for k in range(1, n + 1)]
    a.barh(y + 0.2, n0, height=0.38, color="#b8bec7", label="정지"); a.barh(y - 0.2, nT, height=0.38, color=EV.BLUE, label=f"{T}초")
    for yi, x0, x1 in zip(y, n0, nT):
        a.text(max(x0, x1) + 1, yi, f"{x0}→{x1}", va="center", fontsize=9.5, color=EV.INK)
    a.set_yticks(y); a.set_yticklabels([f"{k}층" + (" ★" if k == m else "") for k in range(1, n + 1)], fontsize=10)
    for lab, k in zip(a.get_yticklabels(), range(1, n + 1)):
        lab.set_color(EV.GREEN if k == m else EV.INK); lab.set_fontweight("bold" if k == m else "normal")
    a.set_xlim(0, max(max(n0), max(nT), 1) * 1.6); a.set_xticks([]); a.legend(frameon=False, fontsize=10, loc="lower right")
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("층별 Top-100 후보 수 (정지→지금)", loc="left", fontsize=11)
    ok0, okT = bool(v0 and v0 <= 100), bool(vT and vT <= 100); f = lambda v: f"{v}위" if v else "1000위 밖"
    fig.text(0.01, 0.975, f"E3 {T}초 · 정지 대비 · {EV.fl(F)} 출발점{task.split('_T')[1]} ({EV.TYPE[t['start_type']]}) · {dict(E3.COND)[c]}", fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.84, f"정답 층 지도: 회색 원 = 정지 Top-100 을 {T}초까지 걸은 만큼 옮긴 자리 · 검정 점 = 그중 {T}초에도 남은 후보 · 파랑 점 = {T}초에 새로 든 후보 · "
             f"빨간 ○ = 출발 자리 · ★ = {T}초 실제 위치 · 검정 선 = 지나온 길 · 초록 동그라미 = {T}초 정답 후보", fontsize=11, color=EV.INK2, va="top")
    fig.text(0.01, 0.905, f"정답 순위 정지 {f(v0)} → {T}초 {f(vT)}" + ("  (Top-100 안으로 들어옴)" if okT and not ok0 else "  (Top-100 밖으로 빠짐)" if ok0 and not okT else "") +
             f"   ·   {T}초 Top-100 중 정지 때부터 남은 후보 {st['kept']}개 · 새 후보 {st['new']}개   ·   걸은 거리 {st['dist']:.0f} m · 관측 {st['nf']}장",
             fontsize=13, fontweight="bold", va="top", color=EV.GREEN if okT else EV.INK)
    sd, name = subpath(task); (od / sd).mkdir(parents=True, exist_ok=True); fig.savefig(od / sd / name, dpi=100, bbox_inches="tight"); plt.close(fig)


def _setup():
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(E3.E2W / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(E3.E2W / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    return E3.load()


def _job(arg):
    cases, conds, fcs = arg; D = _setup(); C = {(c, T): cands(c, T) for c in conds for T in (0,) + TS}; rows = []
    for k in cases:
        for T in TS:
            disp, nf, dist = odom_disp(k, T)
            for c in conds:
                moved, keptT = compare(C[(c, 0)].get(k, []), C[(c, T)].get(k, []), disp)
                m = int(EV.GT[k]["true_map"]); cT = C[(c, T)].get(k, []); c0 = C[(c, 0)].get(k, [])
                st = dict(moved=moved, keptT=keptT, kept=int(sum(keptT)), new=int(len(keptT) - sum(keptT)), nf=nf, dist=dist)
                rows.append(dict(task=k, c=c, T=T, kept=st["kept"], new=st["new"], dist=round(dist, 1), nf=nf,
                                 tf0=sum(int(r["floor_id"]) == m for r in c0), tfT=sum(int(r["floor_id"]) == m for r in cT),
                                 nfl0=len({r["floor_id"] for r in c0}), nflT=len({r["floor_id"] for r in cT})))
                if c in fcs:
                    draw(k, c, T, D, C, st, OUT / FOLDER[c] / f"{T}초")
        print(k, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default=""); ap.add_argument("--conds", default="T2,T3"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-figs", action="store_true"); ap.add_argument("--fig-conds", default="", help="그림을 그릴 조건만(기본 = --conds 전부)"); a = ap.parse_args()
    D = _setup(); conds = a.conds.split(","); tasks = [k for k in EV.TASK if all(k in D[(c, T)] for c in conds for T in (0,) + TS)]   # 요청한 조건이 다 있는 출발점
    cases = a.cases.split(",") if a.cases else tasks; groups = {}
    for k in cases:
        groups.setdefault(EV.TASK[k]["floor"], []).append(k)
    from multiprocessing import Pool
    with Pool(min(a.workers, len(groups))) as pool:
        fc = a.fig_conds.split(",") if a.fig_conds else conds
        R = [r for rr in pool.map(_job, [(g, conds, [] if a.no_figs else fc) for g in groups.values()], chunksize=1) for r in rr]
    if a.cases:
        for r in R:
            print(r)
        return
    report(R, D, conds, tasks)


def report(R, D, conds, tasks):
    S = {(r["task"], r["c"], r["T"]): r for r in R}; TAB.mkdir(parents=True, exist_ok=True)
    tag = "_".join(TAG[c] for c in conds)                  # 조건마다 따로 저장(조건별로 출발점 수가 다를 수 있음)
    def wr(name, rows):
        name = name.replace(".csv", f"_{tag}.csv")
        with open(TAB / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    v = lambda c, T, k: rk(D[(c, T)][k]["true_rank"]) or 5000
    # 시점별 통계(조건 × 시점, 정지 대비)
    tab = []
    for c in conds:
        for T in TS:
            r0 = [v(c, 0, k) for k in tasks]; rT = [v(c, T, k) for k in tasks]; ss = [S[(k, c, T)] for k in tasks]
            tab.append({"조건": dict(E3.COND)[c], "시점_초": T, "출발점수": len(tasks),
                        "Top100_정지": sum(x <= 100 for x in r0), "Top100_지금": sum(x <= 100 for x in rT),
                        "Top10_정지": sum(x <= 10 for x in r0), "Top10_지금": sum(x <= 10 for x in rT), "Top1_정지": sum(x <= 1 for x in r0), "Top1_지금": sum(x <= 1 for x in rT),
                        "Top100_들어옴": sum(x > 100 and y <= 100 for x, y in zip(r0, rT)), "Top100_빠짐": sum(x <= 100 and y > 100 for x, y in zip(r0, rT)),
                        "순위_오름": sum(y < x for x, y in zip(r0, rT)), "순위_내림": sum(y > x for x, y in zip(r0, rT)), "순위_같음": sum(y == x for x, y in zip(r0, rT)),
                        "정지후보_남은수_중앙": float(np.median([s["kept"] for s in ss])), "새후보수_중앙": float(np.median([s["new"] for s in ss])),
                        "정답층_후보수_중앙_정지": float(np.median([s["tf0"] for s in ss])), "정답층_후보수_중앙_지금": float(np.median([s["tfT"] for s in ss])),
                        "후보_층수_중앙_정지": float(np.median([s["nfl0"] for s in ss])), "후보_층수_중앙_지금": float(np.median([s["nflT"] for s in ss])),
                        "걸은거리_중앙_m": float(np.median([s["dist"] for s in ss])), "관측장수_중앙": float(np.median([s["nf"] for s in ss]))})
    wr("정지대비_시점별_통계.csv", tab)
    per = []
    for k in tasks:
        t = EV.TASK[k]
        for c in conds:
            for T in TS:
                s = S[(k, c, T)]; f = lambda x: x if x <= 1000 else "1000위 밖"
                per.append({"과제": E3.ko_task(k), "출발점종류": EV.TYPE[t["start_type"]], "조건": dict(E3.COND)[c], "시점_초": T, "정답순위_정지": f(v(c, 0, k)), "정답순위_지금": f(v(c, T, k)),
                            "정지후보_남은수": s["kept"], "새후보수": s["new"], "정답층_후보수_정지": s["tf0"], "정답층_후보수_지금": s["tfT"],
                            "후보_층수_정지": s["nfl0"], "후보_층수_지금": s["nflT"], "걸은거리_m": s["dist"], "관측장수": s["nf"]})
    wr("정지대비_출발점별.csv", per)
    for name, key in (("정지대비_층별.csv", lambda k: EV.fl(EV.TASK[k]["floor"])), ("정지대비_장소종류별.csv", lambda k: EV.TYPE[EV.TASK[k]["start_type"]])):
        rows = []
        for c in conds:
            for g in dict.fromkeys(key(k) for k in tasks):
                ks = [k for k in tasks if key(k) == g]
                rows.append({"조건": dict(E3.COND)[c], "그룹": g, "출발점수": len(ks), "Top100_정지": sum(v(c, 0, k) <= 100 for k in ks),
                             **{f"Top100_{T}초": sum(v(c, T, k) <= 100 for k in ks) for T in TS}})
        wr(name, rows)
    # 폴더 README
    for c in conds:
        for T in TS:
            od = OUT / FOLDER[c] / f"{T}초"
            if not od.exists():
                continue
            row = next(x for x in tab if x["조건"] == dict(E3.COND)[c] and x["시점_초"] == T)
            L = [f"# E3 {T}초 · 정지 대비 · {dict(E3.COND)[c]}", "",
                 f"출발점마다 한 장. 정지(0초 = E2 제자리 한 번)와 {T}초를 나란히 놓았다. {len(tasks)}곳 중 Top-100 안 정답 위치: 정지 {row['Top100_정지']} → {T}초 {row['Top100_지금']} "
                 f"(들어옴 {row['Top100_들어옴']}, 빠짐 {row['Top100_빠짐']}). {T}초 Top-100 중 정지 때부터 남은 후보는 중앙 {row['정지후보_남은수_중앙']:.0f}개, 걸은 거리 중앙 {row['걸은거리_중앙_m']:.0f} m.", "",
                 "| 출발점 | 종류 | 정지 순위 | " + f"{T}초 순위 | 남은 후보 | 새 후보 | 정답 층 후보 정지→지금 | 걸은 거리 | 그림 |", "|---|---|---|---|---|---|---|---|---|"]
            fmt = lambda x: (f"**{x}**" if x <= 100 else str(x)) if x <= 1000 else "밖"
            L = L[:4]
            for bn in dict.fromkeys(EV.BNAME[EV.TASK[k]["building"]] for k in tasks):
                kb = [k for k in tasks if EV.BNAME[EV.TASK[k]["building"]] == bn]
                L += [f"## {bn} ({len(kb)}곳 중 Top-100 안: 정지 {sum(v(c, 0, k) <= 100 for k in kb)} → {T}초 {sum(v(c, T, k) <= 100 for k in kb)})", ""]
                for fl in dict.fromkeys(subpath(k)[0].name for k in kb):
                    kf = [k for k in kb if subpath(k)[0].name == fl]
                    L += [f"### {bn} {fl} — [폴더]({bn}/{fl}/)", "", "| 출발점 | 종류 | 정지 순위 | " + f"{T}초 순위 | 남은 후보 | 새 후보 | 정답 층 후보 정지→지금 | 걸은 거리 | 그림 |", "|---|---|---|---|---|---|---|---|---|"]
                    for k in kf:
                        s = S[(k, c, T)]; sd, name = subpath(k)
                        L.append(f"| 출발점{k.split('_T')[1]} | {EV.TYPE[EV.TASK[k]['start_type']]} | {fmt(v(c, 0, k))} | {fmt(v(c, T, k))} | {s['kept']} | {s['new']} | {s['tf0']}→{s['tfT']} | {s['dist']:.0f} m | [보기]({sd.as_posix()}/{name}) |")
                    L.append("")
            (od / "README.md").write_text("\n".join(L) + "\n")
    print("표·README 끝")


if __name__ == "__main__":
    main()
