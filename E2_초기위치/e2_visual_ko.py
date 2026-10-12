#!/usr/bin/env python3
"""E2 이해용 시각 자료(2026-10-10). 막대 대신 지도·사진 위에서 보이게 한다. 결과: _01_E2_초기위치/시각자료/
  V1_흐름_<과제>      한 출발점: 카메라 화면 → 그 순간 본 벽 점(위에서 본 모양) → 9개 층 지도의 Top-100 후보 → 정답 층 확대
  V2_왜헷갈리나_<과제> 같은 벽 점을 실제 위치와 1등 후보(다른 층·다른 자리)에 얹은 모습
  V3_전층결과지도      13개 층 지도에 출발점 5곳을 결과 색(Top-100 안·1000 안·밖)과 순위로
  V5_축척_<과제>       실제 위치에서 정답 축척과 목록 축척으로 얹은 벽 점(꼭대기가 어긋나는 이유)
벽 점을 지도에 얹는 식은 협업자 rt_localizer.localize_buf 와 같다: hd = 방향 + 오돔 yaw + 장착각, 지도 = 칸 + R(hd)·점·축척(y 아래).
    python3 scripts/e2_visual_ko.py [--cases 3F_T1,4F_T3]
"""
import argparse, csv, json, math, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import frame  # noqa: E402
import e2_dense as ED  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); G = Path("reports/_01_E2_초기위치/_작업자료"); PKG = Path("reports/_01_E2_초기위치"); OUT = PKG / "1_시각자료"
MAPS = {"kci": Path("/data/kci_work/floor_id_pkg/floor_id/maps/kci_maps_perfloor"),
        "gong": Path("/data/kci_work/floor_id_pkg/floor_id/maps/gong_maps_perfloor"),
        "sw": Path("/data/kci_work/floor_id_pkg/floor_id/maps/sw_maps_perfloor")}
BLUE, ORANGE, GREEN, RED, GRAY, YELLOW, INK, INK2, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9aa3b0", "#eda100", "#14171c", "#5b6370", "#ffffff"
TYPE = {"S1": "긴 직선 복도", "S2": "꺾이는 곳", "S3": "갈림길", "S4": "계단·엘리베이터 앞", "S5": "막다른 곳"}
NAME = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}
BNAME = {"kci": "정보관", "gong": "공학관", "sw": "소융대"}
fl = lambda F: NAME.get(F, F.replace("F", "층"))
FNAME = {"G1F": "공학관1층", "G2F": "공학관2층", "S1F": "소융대1층", "S6F": "소융대6층"}
TYPE_FILE = {"S1": "긴직선복도", "S2": "꺾이는곳", "S3": "갈림길", "S4": "계단엘리베이터앞", "S5": "막다른곳"}


def ko_task(task):
    """파일 이름용 한글 과제 이름: 3F_T1 → 3층_출발점1_갈림길 (TASK 가 채워져 있어야 종류가 붙음)."""
    F, n = task.split("_"); t = TASK.get(task) if isinstance(globals().get("TASK"), dict) else None
    return f"{FNAME.get(F, F.replace('F', '층'))}_출발점{n[1:]}" + (f"_{TYPE_FILE[t['start_type']]}" if t else "")
wrap = lambda d: (d + 180) % 360 - 180
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight"); plt.close(fig); print(name)


def mapimg(building, m):
    p = MAPS[building] / f"{m}F" / "maps" / "base_map.png"
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return 255 - (255 - im) * 0.5


def crop(ax, im, pad=40):
    ys, xs = np.where(im < 200); ax.set_xlim(xs.min() - pad, xs.max() + pad); ax.set_ylim(ys.max() + pad, ys.min() - pad); ax.axis("off")


def obs(task):
    """그 출발 시각의 구조점 표본(로봇 좌표, 미러 적용)과 오돔 yaw."""
    t = TASK[task]; F = t["floor"]; b = t["building"]
    import floor_id as FI
    if ED._ST.get("building") != b:
        cfg = FI.load_cfg(); bb = FI.setup_env(cfg, b); ED._ST.update(building=b, b=bb, cache={})
    import rt_localizer as RT, run_floor_id as Rr
    key = (F, 1)
    if key not in ED._ST["cache"]:
        ED._ST["cache"] = {key: RT.RTLocalizer(lam_sem=0.0, w_nogo=3.0, verbose=False, floor=1, ppm=Rr.MID, data_dir=ED.DATA.get(F, f"/data2/loc_inputs_20261005/{F}"), bag=None)}
    L = ED._ST["cache"][key].L; tt = L.T[0] + float(t["start_sec"])
    pts, yaw = L._scan(tt)
    return np.asarray(pts, float) * [1, 1], float(yaw)


def project(pts, cell_px, hd_rad, ppm):
    c, s = math.cos(hd_rad), math.sin(hd_rad)
    return np.c_[cell_px[0] + (pts[:, 0] * c - pts[:, 1] * s) * ppm, cell_px[1] - (pts[:, 0] * s + pts[:, 1] * c) * ppm]


def gt_of(task):
    t = TASK[task]; F = t["floor"]; S = json.load(open(R / "36_채점용정답" / F / "summary.json"))
    gp = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
    gtime = np.array([float(x["relative_sec"]) for x in gp]); gxy = np.array([[float(x["x_px"]), float(x["y_px"])] for x in gp])
    ts = float(t["start_sec"]); p = gxy[int(np.argmin(abs(gtime - ts)))]
    return p, S["ppm"], S["theta_deg"] + S.get("beta_deg", 228.0), gtime, gxy


def wall_dist(b, m, P, ppm):
    """얹은 점에서 가장 가까운 벽까지 거리(m): 가까운 70% 점의 평균(협업자 점수와 같은 70% 잘라내기, 설명용)."""
    im = cv2.imread(str(MAPS[b] / f"{m}F" / "maps" / "base_map.png"), cv2.IMREAD_GRAYSCALE)
    dt = cv2.distanceTransform((im > 128).astype(np.uint8), cv2.DIST_L2, 5)
    c = np.clip(np.round(P[:, 0]).astype(int), 0, im.shape[1] - 1); r = np.clip(np.round(P[:, 1]).astype(int), 0, im.shape[0] - 1)
    d = np.sort(dt[r, c]) / ppm
    return float(d[:int(len(d) * 0.7)].mean())


def v1(task):
    t = TASK[task]; F, b = t["floor"], t["building"]; m_true = int(GT[task]["true_map"]); n = {"kci": 9, "gong": 7, "sw": 8}[b]
    tp = [x for x in TOP if x["trial_id"] == task]; rank = RES[task]["true_rank"]
    pts, yaw = obs(task); p_gt = gt_of(task)[0]
    fig = plt.figure(figsize=(22, 13.5)); gs = fig.add_gridspec(3, 5, width_ratios=[1.05, 1, 1, 1, 0.02], hspace=0.3, wspace=0.06, bottom=0.03, top=0.88)
    a = fig.add_subplot(gs[0, 0]); a.imshow(frame(F, float(t["start_sec"]))); a.axis("off")
    a.set_title(f"① 그 자리 카메라 화면 ({fl(F)} {TYPE.get(t['start_type'], '')})", loc="left", fontsize=13, fontweight="bold")
    a = fig.add_subplot(gs[1:3, 0]); a.scatter(-pts[:, 1], pts[:, 0], s=2, color=INK, alpha=0.6); a.scatter(0, 0, marker="^", s=220, color=RED, zorder=3)
    for rr in (5, 10, 15):
        a.add_patch(plt.Circle((0, 0), rr, fill=False, ls=":", color=GRAY, lw=1)); a.text(rr * 0.71, rr * 0.71, f"{rr} m", color=INK2, fontsize=9)
    a.set_aspect("equal"); a.set_xlim(-16, 16); a.set_ylim(-16, 16); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color("#d8dce2")
    a.set_title(f"② 그 순간 라이다가 본 벽 점 {len(pts):,}개\n   (▲ 로봇, 위쪽이 앞 · 위에서 내려다본 모양)", loc="left", fontsize=13, fontweight="bold")
    for k, m in enumerate(range(1, n + 1)):
        rr, cc = divmod(k, 3); a = fig.add_subplot(gs[rr, 1 + cc]); im = mapimg(b, m); a.imshow(im, cmap="gray", vmin=0, vmax=255); crop(a, im)
        on = [x for x in tp if int(x["floor_id"]) == m]
        for x in sorted(on, key=lambda x: -int(x["rank"])):
            kk = int(x["rank"]); a.scatter(float(x["x_px"]), float(x["y_px"]), s=max(18, 160 - 1.4 * kk), color=BLUE, alpha=0.8, edgecolor=SURF, linewidth=0.6, zorder=3)
        if m == m_true:
            a.scatter(*p_gt, marker="*", s=420, color=RED, edgecolor=SURF, linewidth=1, zorder=5)
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_color(GREEN); sp.set_linewidth(3)
            a.axis("on"); a.set_xticks([]); a.set_yticks([])
        best = min([int(x["rank"]) for x in on] or [0])
        a.set_title(f"{m}층 · 후보 {len(on)}개" + (f" · 가장 높은 {best}위" if on else "") + ("   ← 실제 층" if m == m_true else ""), fontsize=11.5, loc="left",
                    color=GREEN if m == m_true else INK, fontweight="bold" if m == m_true else "normal")
    if str(RES[task].get("raw_candidates", "")).isdigit():
        how = f"③ {BNAME[b]} {n}개 층 전체에서 이 벽 점이 잘 맞는 자리 {int(RES[task]['raw_candidates']):,}개를 점수순으로 세워, 서로 다른 상위 100개만 표시"
    else:
        how = f"③ {BNAME[b]} {n}개 층 전체를 벽 겹침으로 훑은 뒤 " + ("거리 비교로" if "거리" in METHOD else "정밀 검증(미세 정합 + 물리 확인)으로") + " 다시 순위를 매겨, 서로 다른 상위 100개만 표시"
    fig.text(0.31, 0.945, how, fontsize=13.5, fontweight="bold")
    res = f"정답 위치는 {rank}위" + (" → Top-100 안" if rank.isdigit() and int(rank) <= 100 else " → Top-100 밖")
    fig.text(0.31, 0.922, f"파란 원 = 후보(클수록 높은 순위) · ★ = 실제 위치 · {res}", fontsize=12.5, color=INK2)
    fig.suptitle(f"E2 초기 위치: 제자리에서 한 번 보고 후보 만들기 — {fl(F)} 출발점{task.split('_T')[1]} (A 라이다 · 축척 탐색 · {METHOD})", x=0.01, ha="left", fontsize=18, fontweight="bold", y=0.99)
    save(fig, f"1_한출발점흐름_{ko_task(task)}")


def v2(task):
    t = TASK[task]; F, b = t["floor"], t["building"]; m_true = int(GT[task]["true_map"])
    tp = sorted([x for x in TOP2 if x["trial_id"] == task], key=lambda x: int(x["rank"]))
    pts, yaw = obs(task); p_gt, ppm_gt, hd_gt, gtime, gxy = gt_of(task)
    panels = [(m_true, p_gt, math.radians(hd_gt) + yaw, ppm_gt, GREEN, f"실제 위치 ({fl(F)})", "★")]
    t1 = tp[0]; oth = next((x for x in tp if int(x["floor_id"]) not in (m_true, int(t1["floor_id"]))), tp[1])
    for x in (t1, oth):
        panels.append((int(x["floor_id"]), (float(x["x_px"]), float(x["y_px"])), math.radians(float(x["heading_deg"])) + yaw, float(x["ppm"]), ORANGE,
                       f"{x['rank']}위 후보 ({int(x['floor_id'])}층)" + (" — 다른 층에서 가장 높은 것" if x is oth else ""), "●"))
    fig, ax = plt.subplots(1, 3, figsize=(21, 6.6))
    for a, (m, pos, hd, ppm, col, title, mk) in zip(ax, panels):
        P = project(pts, pos, hd, ppm); im = mapimg(b, m); a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.scatter(P[:, 0], P[:, 1], s=3, color=col, alpha=0.85)
        a.scatter(*pos, marker="*" if mk == "★" else "o", s=380 if mk == "★" else 200, color=RED if mk == "★" else col, edgecolor=SURF, linewidth=1.2, zorder=5)
        a.annotate("", xy=(pos[0] + 2.5 * ppm * math.cos(hd), pos[1] - 2.5 * ppm * math.sin(hd)), xytext=pos, arrowprops=dict(arrowstyle="-|>", color=INK, lw=2), zorder=6)
        r_ = 17 * ppm; a.set_xlim(pos[0] - r_, pos[0] + r_); a.set_ylim(pos[1] + r_ * 0.75, pos[1] - r_ * 0.75); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(col); sp.set_linewidth(3)
        a.set_title(title, fontsize=14, fontweight="bold", loc="left", color=col)
        a.text(0.02, 0.03, f"벽까지 평균 {wall_dist(b, m, P, ppm):.2f} m", transform=a.transAxes, fontsize=12.5, fontweight="bold", color=INK,
               bbox=dict(facecolor=SURF, edgecolor=col, pad=4))
    fig.suptitle(f"왜 헷갈리나 — 같은 벽 점 {len(pts):,}개를 실제 위치와 다른 층 후보에 얹어 보면 ({fl(F)} {TYPE.get(t['start_type'], '')}, 벽 겹침만으로 정답 {RES2[task]['true_rank']}위)",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.01, 0.0, "점 색 = 그 자리·방향(화살표)에 얹은 벽 점, 회색 = 지도의 벽. '벽까지 평균' = 벽에 가까운 70% 점이 지도 벽에서 떨어진 평균 거리(작을수록 잘 맞음).\n"
             "벽 겹침 점수도 이렇게 가장 먼 30% 점을 버리고 매기므로, 일부 점이 허공에 떠도 나머지가 벽에 붙으면 다른 층·다른 자리가 실제 위치보다 높게 나올 수 있다.",
             fontsize=11.5, color=INK2, va="top", linespacing=1.6)
    save(fig, f"2_왜헷갈리나_{ko_task(task)}")


def v3(RES, method, fname):
    FL = list(dict.fromkeys(t["floor"] for t in TASK.values()))
    fig, ax = plt.subplots(7, 2, figsize=(17, 25)); ax = ax.ravel()
    col_of = lambda v: GREEN if (v and v <= 100) else (YELLOW if v else RED)
    for i, F in enumerate(FL):
        a = ax[i]; tk = sorted([t for t in TASK.values() if t["floor"] == F], key=lambda t: t["start_type"]); b = tk[0]["building"]; m = int(GT[tk[0]["task_id"]]["true_map"])
        im = mapimg(b, m); a.imshow(im, cmap="gray", vmin=0, vmax=255)
        _, _, _, gtime, gxy = gt_of(tk[0]["task_id"]); a.plot(gxy[:, 0], gxy[:, 1], color="#c3c8d0", lw=2.5, zorder=1)
        parts = []
        for t in tk:
            p = gt_of(t["task_id"])[0]; rk = RES[t["task_id"]]["true_rank"]; v = int(rk) if rk.isdigit() else None
            a.scatter(*p, s=420, color=col_of(v), edgecolor=SURF, linewidth=2, zorder=4)
            a.text(*p, t["start_type"][1], ha="center", va="center", fontsize=10, fontweight="bold", color=SURF, zorder=5)
            parts.append(f"{t['start_type']} {str(v) + '위' if v else '1000위 밖'}")
        ys, xs = np.where(im < 200); pad = 60; a.set_xlim(xs.min() - pad, xs.max() + pad); a.set_ylim(ys.max() + pad, ys.min() - pad); a.axis("off")
        nin = sum(1 for t in tk if RES[t["task_id"]]["true_rank"].isdigit() and int(RES[t["task_id"]]["true_rank"]) <= 100)
        a.set_title(f"{fl(F)} — Top-100 안 {nin}/5", loc="left", fontsize=13, fontweight="bold")
        a.text(0.0, -0.02, "  ·  ".join(parts), transform=a.transAxes, fontsize=10.5, color=INK2, va="top")
    ax[13].axis("off")
    h = [Line2D([], [], marker="o", ls="", color=c, ms=16) for c in (GREEN, YELLOW, RED)]
    ax[13].legend(h, ["정답 위치가 Top-100 안", "100위 밖 · 1000위 안", "1000위 밖"], loc="upper center", frameon=False, fontsize=14)
    ax[13].text(0.5, 0.3, "원 안 숫자 = 출발점 종류\n1 긴 직선 복도 · 2 꺾이는 곳 · 3 갈림길\n4 계단·엘리베이터 앞 · 5 막다른 곳\n회색 선 = 그 층 주행 경로",
                ha="center", va="center", fontsize=12, color=INK2, transform=ax[13].transAxes, linespacing=1.7)
    tot = sum(1 for r in RES.values() if r["true_rank"].isdigit() and int(r["true_rank"]) <= 100)
    fig.suptitle(f"13개 층 65개 출발점: 제자리 한 번 관측으로 정답 위치가 몇 위였나 (A 라이다 · 축척 탐색 · {method}, Top-100 안 {tot}/65)", x=0.01, ha="left", fontsize=19, fontweight="bold", y=1.0)
    fig.tight_layout(h_pad=2.5); save(fig, fname)


def v5(task):
    t = TASK[task]; F, b = t["floor"], t["building"]; m_true = int(GT[task]["true_map"]); p_gt, ppm_gt, hd_gt, _, _ = gt_of(task)
    d = json.loads(str(np.load(ED.OUT / "T0" / f"{task}.npz")["meta"])); sc = d["maps"][str(m_true)]["scales"]; near = min(sc, key=lambda s: abs(s - ppm_gt))
    pts, yaw = obs(task)
    fig, a = plt.subplots(figsize=(10, 7)); im = mapimg(b, m_true); a.imshow(im, cmap="gray", vmin=0, vmax=255)
    for s, col, lab in ((ppm_gt, GREEN, f"정답 축척 {ppm_gt:.1f} px/m"), (near, ORANGE, f"목록에서 가장 가까운 축척 {near:.1f} px/m ({100 * (near - ppm_gt) / ppm_gt:+.0f}%)")):
        P = project(pts, p_gt, math.radians(hd_gt) + yaw, s); a.scatter(P[:, 0], P[:, 1], s=3, color=col, alpha=0.8, label=lab)
    a.scatter(*p_gt, marker="*", s=300, color=RED, edgecolor=SURF, zorder=5)
    r_ = 16 * ppm_gt; a.set_xlim(p_gt[0] - r_, p_gt[0] + r_); a.set_ylim(p_gt[1] + r_ * 0.7, p_gt[1] - r_ * 0.7); a.axis("off")
    a.legend(loc="lower left", fontsize=11, markerscale=5, frameon=True)
    a.set_title(f"실제 자리·방향에서도 축척이 조금 다르면 먼 벽이 어긋난다 ({fl(F)} 출발점{task.split('_T')[1]})", loc="left", fontsize=13, fontweight="bold")
    save(fig, f"5_축척어긋남_{ko_task(task)}")


def startmaps():
    """5_출발점지도: 층마다 정답 경로(회색) 위 출발점 5곳(원 안 번호)과 장소 종류·출발 시각. 축척은 가로·세로 같음."""
    out = PKG / "5_출발점지도"; out.mkdir(parents=True, exist_ok=True)
    cols = [BLUE, ORANGE, GREEN, "#8a5cd1", RED]
    for F in dict.fromkeys(t["floor"] for t in TASK.values()):
        tk = sorted([t for t in TASK.values() if t["floor"] == F], key=lambda t: int(t["task_id"].split("_T")[1])); b = tk[0]["building"]
        m = int(GT[tk[0]["task_id"]]["true_map"]); im = mapimg(b, m); _, ppm, _, _, gxy = gt_of(tk[0]["task_id"])
        H, W = im.shape; fig, a = plt.subplots(figsize=(18, 18 * H / W + 1.4)); a.imshow(im, cmap="gray", vmin=0, vmax=255)
        a.plot(gxy[:, 0], gxy[:, 1], color="#c3c8d0", lw=2.5, zorder=1)
        for i, t in enumerate(tk):
            p = gt_of(t["task_id"])[0]; n = t["task_id"].split("_T")[1]
            a.scatter(*p, s=520, color=cols[i % 5], edgecolor=SURF, linewidth=2, zorder=4)
            a.text(*p, n, ha="center", va="center", fontsize=12, fontweight="bold", color=SURF, zorder=5)
            a.annotate(f"출발점{n} {TYPE.get(t['start_type'], '')} ({float(t['start_sec']):.0f}초)", p, (14, -14), textcoords="offset points", fontsize=12, fontweight="bold",
                       color=INK, bbox=dict(facecolor=SURF, edgecolor=cols[i % 5], alpha=0.9, pad=2), zorder=6)
        ys, xs = np.where(im < 200); pad = 60; a.set_xlim(xs.min() - pad, xs.max() + pad); a.set_ylim(ys.max() + pad, ys.min() - pad); a.axis("off")
        a.set_title(f"{fl(F)} 출발점 5곳 · 회색 선 = 그 층 주행 경로 · 지도 실제 크기 약 {W / ppm:.0f} m × {H / ppm:.0f} m (가로·세로 같은 축척)", loc="left",
                    fontsize=14, fontweight="bold")
        fig.savefig(out / f"{FNAME.get(F, F.replace('F', '층'))}.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("출발점지도", F)


def main():
    global TASK, GT, RES, TOP, RES2, TOP2, METHOD
    ap = argparse.ArgumentParser(); ap.add_argument("--startmaps-only", action="store_true"); ap.add_argument("--cases", default="3F_T1,S1F_T1,6F_T4,7F_T1"); ap.add_argument("--scale-cases", default="S6F_T2,3F_T1")
    ap.add_argument("--src", default="_nospace_정밀검증", help="흐름·전층 그림의 결과(2026-10-10 새 구성: A 라이다 · 축척 탐색 · + 정밀 검증)")
    ap.add_argument("--v2-src", default="_nospace", help="'왜 헷갈리나' 그림의 결과(벽 겹침)"); a = ap.parse_args()
    TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}; GT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    ld = lambda sfx: ({r["trial_id"]: r for r in csv.DictReader(open(G / f"E2_dense_T0{sfx}/E2_top100_per_trial.csv"))}, list(csv.DictReader(open(G / f"E2_dense_T0{sfx}/E2_top100_candidates.csv"))))
    RES, TOP = ld(a.src); RES2, TOP2 = ld(a.v2_src)
    METHOD = {"_nospace": "벽 겹침", "_nospace_거리비교": "+ 거리 비교", "_nospace_정밀검증": "+ 정밀 검증"}.get(a.src, a.src)
    startmaps()
    if a.startmaps_only:
        return
    v3(RES, METHOD, "4_전층결과지도_정밀검증"); v3(RES2, "벽 겹침", "4_전층결과지도_벽겹침")
    for c in a.cases.split(","):
        v1(c); v2(c)
    for c in a.scale_cases.split(","):
        v5(c)


if __name__ == "__main__":
    main()
