#!/usr/bin/env python3
"""E3 본 결과 그림(2026-10-11): 노드 그대로 / 연결판(E2 후보 이어받기). 시점마다 폴더, 그 안을 건물 → 층 → 출발점으로 나눈다.
  1_노드그대로/<T>초/<건물>/<층>/출발점<n>_<종류>.png
     ① 판정 층 후보 자세로 쌓인 라이다를 얹은 모습 ② 정답 층 후보 자세로 얹은 모습(★ 실제 위치·빨간 점선 실제 길) ③ 층 순위(거리 비교 점수)
  2_연결판_E2후보이어받기/<T>초/<건물>/<층>/출발점<n>_<종류>.png
     ① 1위 후보 자세로 얹은 모습 ② 정답 층 지도 위 남은 후보·실제 위치·정답 후보 ③ 층별 후보 수(정지 E2 Top-100 → 지금 남은 후보)
얹는 식은 위치추정 코드와 같다(e3_accum_registered). 정답은 표시·채점에만.
    python3 scripts/e3_main_figures.py [--modes node,seeded] [--cases 4F_T1] [--workers 8]
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402

PKG = Path("reports/_03_E3_걸으며좁히기"); W = PKG / "_작업자료"; G = EV.G
NODE = Path("/data2/e3_node_20261011"); TS = (15, 30, 45, 60); NMAP = {"kci": 9, "gong": 7, "sw": 8}
FOLDER = {"node": "2_비교/1_노드그대로_로봇방식", "seeded": "2_비교/2_연결판_E2후보만", "merged": "1_최종_합치기"}
wrap = lambda a: (a + 180) % 360 - 180


def subpath(k):
    t = EV.TASK[k]; fl = t["floor"].lstrip("GS").replace("F", "층")
    return Path(EV.BNAME[t["building"]]) / fl, f"출발점{k.split('_T')[1]}_{EV.TYPE[t['start_type']].replace(' ', '').replace('·', '')}.png"


def walked(k, T):
    import e3_accum_registered as AR
    Z = AR._loc(k); L = Z.L; ts = float(EV.TASK[k]["start_sec"])
    sel = (L.T >= L.T[0] + ts) & (L.T <= L.T[0] + ts + T); x, y = np.asarray(L.X)[sel], np.asarray(L.Y)[sel]
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def scan(a, k, T, m, px, py, hd, s, b, m_true, pg, ppm_gt, gt, gx, ts, title, beta=228.0):
    """후보 자세로 쌓인 라이다를 그 층 지도에 얹기. 테두리: 초록 = 정답(정답 층·3 m 안), 주황 = 정답 층이지만 먼 자리, 빨강 = 다른 층."""
    import e3_accum_registered as AR
    P, pos, nf, npt, _ = AR.registered(k, T, px, py, hd, s, beta_deg=beta)
    a.imshow(EV.mapimg(b, m), cmap="gray", vmin=0, vmax=255)
    a.scatter(P[:, 0], P[:, 1], s=1.4, color=EV.ORANGE, alpha=0.55, linewidths=0, zorder=3)
    a.plot(pos[:, 0], pos[:, 1], color=EV.INK, lw=1.8, zorder=4); a.scatter(*pos[-1], marker="o", s=120, color=EV.BLUE, edgecolor=EV.SURF, zorder=6)
    same = m == m_true; d = float(np.hypot(*(pos[-1] - pg)) / ppm_gt) if same else None
    if same:
        sel = (gt >= ts) & (gt <= ts + T)
        a.plot(gx[sel, 0], gx[sel, 1], color=EV.RED, lw=1.4, ls="--", zorder=4); a.scatter(*pg, marker="*", s=280, color=EV.RED, edgecolor=EV.SURF, zorder=7)
    box = np.vstack([pos, pg[None]]) if same else pos; r_ = 13 * s
    a.set_xlim(box[:, 0].min() - r_, box[:, 0].max() + r_); a.set_ylim(box[:, 1].max() + r_, box[:, 1].min() - r_); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    col = EV.GREEN if (same and d <= 3.0) else (EV.ORANGE if same else EV.RED)
    for sp in a.spines.values():
        sp.set_color(col); sp.set_linewidth(2.5)
    a.set_title(title + f"\n{m}층" + (" = 정답 층" if same else " (다른 층)") + (f" · 실제 위치에서 {d:.1f} m" if same else "") +
                f" · 벽까지 평균 {EV.wall_dist(b, m, P, s):.2f} m", loc="left", fontsize=11, color=col, fontweight="bold")
    return d


def draw_node(k, T, od):
    t = EV.TASK[k]; F, b = t["floor"], t["building"]; m_true = int(EV.GT[k]["true_map"]); ts = float(t["start_sec"]); n = NMAP[b]
    _, ppm_gt, hd_gt, gt, gx = EV.gt_of(k); pg = gx[int(np.argmin(abs(gt - (ts + T))))]
    R = {}
    for f in (NODE / k).glob("*.json"):
        r = json.load(open(f))["rows"].get(str(T))
        if r and "twoway" in r:
            R[int(f.stem)] = r
    order = sorted(R.values(), key=lambda r: -r["twoway"]); top = order[0]; tf = R.get(m_true)
    fr = next((i + 1 for i, r in enumerate(order) if r["map"] == m_true), None)
    fig = plt.figure(figsize=(25, 7.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.55], wspace=0.1, top=0.76, bottom=0.03)
    d1 = scan(fig.add_subplot(gs[0]), k, T, top["map"], *top["pos"], top["phi"] + top["beta_deg"], top["ppm"], b, m_true, pg, ppm_gt, gt, gx, ts,
              "① 판정 층(거리 비교 1위) 후보 자세로 쌓인 라이다", beta=top["beta_deg"])
    if tf:
        d2 = scan(fig.add_subplot(gs[1]), k, T, m_true, *tf["pos"], tf["phi"] + tf["beta_deg"], tf["ppm"], b, m_true, pg, ppm_gt, gt, gx, ts,
                  f"② 정답 층 후보 자세로 쌓인 라이다 (층 순위 {fr}위)", beta=tf["beta_deg"])
        yaw_ok = abs(wrap(tf["phi"] + tf["beta_deg"] - hd_gt)) <= 30
    a = fig.add_subplot(gs[2]); y = np.arange(n)[::-1]; sc = [R[m]["twoway"] if m in R else np.nan for m in range(1, n + 1)]
    lo = np.nanmin(sc); cols = [EV.GREEN if m == m_true else (EV.BLUE if m == top["map"] else "#b8bec7") for m in range(1, n + 1)]
    a.barh(y, [v - lo + 0.05 for v in sc], color=cols, height=0.62)
    for yi, v, m in zip(y, sc, range(1, n + 1)):
        rk = next((i + 1 for i, r in enumerate(order) if r["map"] == m), None)
        a.text(v - lo + 0.08, yi, f"{rk}위 ({v:.2f})", va="center", fontsize=10, color=EV.INK, fontweight="bold" if m in (m_true, top["map"]) else "normal")
    a.set_yticks(y); a.set_yticklabels([f"{m}층" + (" ★" if m == m_true else "") for m in range(1, n + 1)], fontsize=10.5)
    for lab, m in zip(a.get_yticklabels(), range(1, n + 1)):
        lab.set_color(EV.GREEN if m == m_true else EV.INK)
    a.set_xticks([]); a.set_xlim(0, (np.nanmax(sc) - lo + 0.05) * 1.6)
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("③ 층 순위: 지금 스캔 거리 비교 점수\n(파랑 = 판정 층, 초록 ★ = 정답 층)", loc="left", fontsize=11)
    ok1 = top["map"] == m_true and d1 is not None and d1 <= 3.0 and abs(wrap(top["phi"] + top["beta_deg"] - hd_gt)) <= 30
    fig.text(0.01, 0.975, f"E3 노드 그대로 · {T}초 · {EV.fl(F)} 출발점{k.split('_T')[1]} ({EV.TYPE[t['start_type']]}) · 로봇 실험 모드와 같은 계산(floor_id, 창 {T}초)", fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.905, f"판정 {top['map']}층 · 정답 {m_true}층(층 순위 {fr}위) · " + ("1위 위치 맞음" if ok1 else "1위 위치 틀림") +
             (f" · 층을 안다면 위치 {'맞음' if (d2 <= 3.0 and yaw_ok) else '틀림'}({d2:.1f} m)" if tf else "") + f" · 걸은 거리 {walked(k, T):.0f} m",
             fontsize=13, fontweight="bold", va="top", color=EV.GREEN if ok1 else EV.INK)
    sd, name = subpath(k); (od / sd).mkdir(parents=True, exist_ok=True); fig.savefig(od / sd / name, dpi=100, bbox_inches="tight"); plt.close(fig)


def draw_seeded(k, T, od, C, E2c, E2r, head="E3 연결판", how="E2 정지 Top-100 을 이어받아 노드와 같은 부품으로 다시 매김"):
    t = EV.TASK[k]; F, b = t["floor"], t["building"]; m_true = int(EV.GT[k]["true_map"]); ts = float(t["start_sec"]); n = NMAP[b]
    _, ppm_gt, hd_gt, gt, gx = EV.gt_of(k); pg = gx[int(np.argmin(abs(gt - (ts + T))))]
    cs = sorted(C.get((k, T), []), key=lambda r: int(r["rank"])); top = cs[0]
    okp = lambda c: int(c["floor_id"]) == m_true and math.hypot(float(c["x_px"]) - pg[0], float(c["y_px"]) - pg[1]) / ppm_gt <= 3.0 and abs(wrap(float(c["heading_deg"]) - hd_gt)) <= 30
    tr = next((int(c["rank"]) for c in cs if okp(c)), None)
    fig = plt.figure(figsize=(25, 7.6)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.3, 0.55], wspace=0.1, top=0.76, bottom=0.03)
    scan(fig.add_subplot(gs[0]), k, T, int(top["floor_id"]), float(top["x_px"]), float(top["y_px"]), float(top["heading_deg"]), float(top["ppm"]), b, m_true, pg, ppm_gt, gt, gx, ts,
         "① 1위 후보 자세로 쌓인 라이다")
    a = fig.add_subplot(gs[1]); im = EV.mapimg(b, m_true); a.imshow(im, cmap="gray", vmin=0, vmax=255)
    on = [c for c in cs if int(c["floor_id"]) == m_true]
    for c in sorted(on, key=lambda c: -int(c["rank"])):
        a.scatter(float(c["x_px"]), float(c["y_px"]), s=max(30, 160 - 1.3 * int(c["rank"])), color=EV.BLUE, alpha=0.8, edgecolor=EV.SURF, linewidth=0.5, zorder=3)
    sel = (gt >= ts) & (gt <= ts + T); a.plot(gx[sel, 0], gx[sel, 1], color=EV.INK, lw=1.6, zorder=4)
    a.scatter(*pg, marker="*", s=320, color=EV.RED, edgecolor=EV.SURF, zorder=6)
    if tr:
        tc = next(c for c in cs if int(c["rank"]) == tr)
        a.scatter(float(tc["x_px"]), float(tc["y_px"]), s=460, facecolor="none", edgecolor=EV.GREEN, linewidth=2.4, zorder=7)
    EV.crop(a, im, pad=30); a.axis("off")
    a.set_title(f"② 정답 층({m_true}층) 지도: 남은 후보 {len(cs)}개 중 이 층 {len(on)}개 (파랑, 클수록 높은 순위)\n★ 지금 실제 위치 · 검정 선 지나온 길 · 초록 동그라미 정답 후보",
                loc="left", fontsize=11)
    a = fig.add_subplot(gs[2]); y = np.arange(n)[::-1]
    n0 = [sum(int(c["floor_id"]) == m for c in E2c.get(k, [])) for m in range(1, n + 1)]; nT = [sum(int(c["floor_id"]) == m for c in cs) for m in range(1, n + 1)]
    a.barh(y + 0.2, n0, height=0.38, color="#b8bec7", label="정지(E2 Top-100)"); a.barh(y - 0.2, nT, height=0.38, color=EV.BLUE, label=f"{T}초 남은 후보")
    for yi, x0, x1 in zip(y, n0, nT):
        a.text(max(x0, x1) + 1, yi, f"{x0}→{x1}", va="center", fontsize=9.5, color=EV.INK)
    a.set_yticks(y); a.set_yticklabels([f"{m}층" + (" ★" if m == m_true else "") for m in range(1, n + 1)], fontsize=10.5)
    for lab, m in zip(a.get_yticklabels(), range(1, n + 1)):
        lab.set_color(EV.GREEN if m == m_true else EV.INK)
    a.set_xlim(0, max(max(n0), max(nT), 1) * 1.6); a.set_xticks([]); a.legend(frameon=False, fontsize=10, loc="lower right")
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("③ 층별 후보 수 (정지 → 지금)", loc="left", fontsize=11)
    r0 = E2r.get(k); f_ = lambda v: f"{v}위" if v else "남은 후보 밖"
    fig.text(0.01, 0.975, f"{head} · {T}초 · {EV.fl(F)} 출발점{k.split('_T')[1]} ({EV.TYPE[t['start_type']]}) · {how}",
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.01, 0.905, f"정답 순위 정지(E2) {f_(r0)} → {T}초 {f_(tr)} · 남은 후보 {len(cs)}개 · 걸은 거리 {walked(k, T):.0f} m",
             fontsize=13, fontweight="bold", va="top", color=EV.GREEN if tr == 1 else EV.INK)
    sd, name = subpath(k); (od / sd).mkdir(parents=True, exist_ok=True); fig.savefig(od / sd / name, dpi=100, bbox_inches="tight"); plt.close(fig)


def _setup():
    EV.TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
    EV.GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}


def _job(arg):
    cases, modes = arg; _setup(); C = {}; E2c = {}; E2r = {}
    CM = {}
    if "merged" in modes:                                   # 2026-10-12 합치기(E2 후보 + 15초마다 노드 재탐색)
        for r in csv.DictReader(open(W / "연결판_재탐색" / "연결판_남은후보.csv")):
            CM.setdefault((r["trial_id"], int(r["T"])), []).append(r)
    if "seeded" in modes or "merged" in modes:
        if "seeded" in modes:
            for r in csv.DictReader(open(W / "연결판" / "연결판_남은후보.csv")):
                C.setdefault((r["trial_id"], int(r["T"])), []).append(r)
        for r in csv.DictReader(open(G / "E2_dense_T0_nospace_정밀검증" / "E2_top100_candidates.csv")):
            E2c.setdefault(r["trial_id"], []).append(r)
        for r in csv.DictReader(open(G / "E2_dense_T0_nospace_정밀검증" / "E2_top100_per_trial.csv")):
            E2r[r["trial_id"]] = int(r["true_rank"]) if str(r["true_rank"]).isdigit() and int(r["true_rank"]) <= 100 else None
    for k in cases:
        for T in TS:
            if "node" in modes:
                draw_node(k, T, PKG / FOLDER["node"] / f"{T}초")
            if "seeded" in modes:
                draw_seeded(k, T, PKG / FOLDER["seeded"] / f"{T}초", C, E2c, E2r)
            if "merged" in modes:
                draw_seeded(k, T, PKG / FOLDER["merged"] / f"{T}초", CM, E2c, E2r, head="E3 합치기(최종)",
                            how="E2 정지 Top-100 + 15초마다 노드가 새로 찾은 후보를 함께 이어받아 다시 매김")
        print(k, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modes", default="node,seeded"); ap.add_argument("--cases", default=""); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); _setup(); modes = a.modes.split(",")
    cases = a.cases.split(",") if a.cases else list(EV.TASK); groups = {}
    for k in cases:
        groups.setdefault(EV.TASK[k]["floor"], []).append(k)
    from multiprocessing import Pool
    with Pool(min(a.workers, len(groups))) as pool:
        pool.map(_job, [(g, modes) for g in groups.values()], chunksize=1)


if __name__ == "__main__":
    main()
