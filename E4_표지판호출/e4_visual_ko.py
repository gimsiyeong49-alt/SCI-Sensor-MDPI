#!/usr/bin/env python3
"""E4 이해용 시각 자료(2026-10-08). 사용자가 PPT 그림을 직접 만들기 전에 '무슨 실험을 어떻게 했고 전층에서 어떻게 나왔나'를
지도·사진 위에서 보이게 한다. 결과: 59_E4_호출정책/paper/visual_ko/
  V1_흐름_4F243   한 순간(4층 243초)을 따라가며 실험 단계: 카메라 화면 → 위치 후보 → 표지판 찾기 → 읽으면 가려지나 → 채점
  V2_확실vs헷갈림 같은 4층에서 확실한 순간과 헷갈리는 순간의 후보 분포
  V3_전층지도     13개 층 지도에 실제 경로(상태별 색)와 멈춘 곳
  V4_규칙비교_4F  4층 같은 경로에서 규칙별 멈춘 곳
  V5_층별효과     층마다 위치 맞음: 멈춤 없음 → 지금 규칙(번호 읽었다면)
  V6_오돔튐       오돔 튐 8개: 원래 vs 튐 넣은 뒤 상태 띠
    python3 scripts/e4_visual_ko.py
"""
import csv, json, math, os, re, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
sys.path.insert(0, str(Path(__file__).parent))
from e4_figures import frame  # noqa: E402
from obs_feedback import A  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper") / "visual_ko"
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)
FB = os.environ.get("E4_FB", "feedback_oracle")      # 채점 폴더(feedback_oracle_dedup = 같은 번호 한 번만)

D = Path("/data/kci_work/maps_perfloor_20261001"); L = R / "35_위치추정_축척추정"
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
NAME = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}
BLUE, ORANGE, GREEN, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, SURF = "#14171c", "#5b6370", "#ffffff"
STC = {"GEOMETRIC_CONFIDENT": GREEN, "GEOMETRIC_AMBIGUITY": BLUE, "GEOMETRIC_INSUFFICIENT": ORANGE, "SENSOR_UNRELIABLE": YELLOW}
STK = {"GEOMETRIC_CONFIDENT": "확실", "GEOMETRIC_AMBIGUITY": "헷갈림", "GEOMETRIC_INSUFFICIENT": "헷갈림 + 긴 직선 복도", "SENSOR_UNRELIABLE": "센서 이상"}
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fl(F):
    return NAME.get(F, F.replace("F", "층"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig); print(name)


def basemap(F):
    im = cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE)
    return 255 - (255 - im) * 0.45                                   # 지도는 옅게


def content_box(F, pad=40):
    im = cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE); ys, xs = np.where(im < 200)
    return xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad


def gt(F):
    g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
    return np.array([float(r["relative_sec"]) for r in g]), np.array([[float(r["x_px"]), float(r["y_px"])] for r in g])


def gt_at(F, t, _c={}):
    if F not in _c:
        _c[F] = gt(F)
    tt, xy = _c[F]
    return xy[int(np.argmin(abs(tt - t)))]


def cands(F, t):
    c = [r for r in csv.DictReader(open(L / F / "loc/topk_series.csv")) if abs(float(r["relative_sec"]) - t) < 1e-6]
    return sorted(c, key=lambda r: -float(r["score"]))


def draw_cands(a, F, t, near=1.5, label=True):
    c = cands(F, t); best = float(c[0]["score"])
    for r in c:
        s = float(r["score"]); w = math.exp(s - best); isn = s >= best - near
        a.scatter(float(r["x_px"]), float(r["y_px"]), s=40 + 260 * w, color=BLUE if isn else "#9aa3b0", alpha=0.85 if isn else 0.5,
                  edgecolor=SURF, linewidth=1.2, zorder=3)
    g = gt_at(F, t); a.scatter(*g, marker="*", s=420, color=RED, edgecolor=SURF, linewidth=1.2, zorder=5)
    return c


def states():
    T = list(csv.DictReader(open(P / "paper/csv/E4_trigger_decisions.csv")))
    out = {}
    for r in T:
        out.setdefault(r["trial_id"][6:], {})[float(r["relative_sec"])] = r["uncertainty_class"]
    return out


def legend_states(ax_or_fig, **kw):
    h = [Line2D([], [], color=STC[k], lw=5) for k in STC if k != "SENSOR_UNRELIABLE"] + [Line2D([], [], color="#c3c8d0", lw=5)]
    return ax_or_fig.legend(h, [STK[k] for k in STC if k != "SENSOR_UNRELIABLE"] + ["위치추정 시작 전"], frameon=False, **kw)


def path_by_state(a, F, S, lw=3):
    """실제 경로 전체를 옅은 회색으로 깔고(위치추정 시작 전 20초 포함), 그 위에 초마다 상태 색을 이어서 칠한다.
    오돔이 1초 넘게 빈 곳은 잇지 않는다(4층 3곳)."""
    tt, xy = gt(F); st = S[F]; ks = np.array(sorted(st))
    gap = np.where(np.diff(tt) > 1.0)[0]
    for seg in np.split(np.arange(len(tt)), gap + 1):
        a.plot(xy[seg, 0], xy[seg, 1], color="#c3c8d0", lw=lw * 0.8, solid_capstyle="round", zorder=1)
    for t0 in ks:
        i0 = int(np.searchsorted(tt, t0 - 0.5)); i1 = int(np.searchsorted(tt, t0 + 0.5))   # 다음 조각 첫 점까지 넣어 잇는다
        i1 = min(i1 + 1, len(tt))
        if i1 - i0 >= 2 and not np.any(np.diff(tt[i0:i1]) > 1.0):
            a.plot(xy[i0:i1, 0], xy[i0:i1, 1], color=STC[st[t0]], lw=lw, solid_capstyle="round", zorder=2)


def stops(pol, F):
    return json.load(open(P / pol / F / "run/events_live_replay.json"))


def v1():
    F, t = "4F", 243.0
    E = [e for e in stops(MAIN, F) if abs(e["t"] - t) < 1e-6][0]
    info = [x for x in json.load(open(P / MAIN / F / "informative.json")) if abs(x["t"] - t) < 1e-6 and x["kind"] == "번호판"][0]
    sem = json.load(open(D / F / f"{F}_semantic_map.json")); rooms = [s for s in sem["spaces"] if s.get("type") == "room" and s.get("polygon_px")]
    tl = {float(r["relative_sec"]): r for r in csv.DictReader(open(L / F / "loc/timeline.csv"))}[t]
    ppm = float(tl["ppm_est"]); o = np.array([float(tl["odom_x_m"]), float(tl["odom_y_m"])]); xy = np.array(E["xyz"][:2])
    c = cands(F, t); near = [r for r in c if float(r["score"]) >= float(c[0]["score"]) - 1.5]
    im = frame(F, t); bx = E["picks"][0]; M = basemap(F); g = gt_at(F, t); cb = content_box(F)
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], hspace=0.22, wspace=0.06)
    a = fig.add_subplot(gs[0, 0]); a.imshow(im); a.axis("off")
    a.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=ORANGE, lw=3))
    a.text(bx[0] - 10, bx[3] + 26, f"표지판 후보 · 점수 {E['cur_sim']:.3f} · 거리 {E['z'][0]:.1f} m", color=ORANGE, fontsize=11.5, fontweight="bold",
           bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.9))
    a.set_title("① RealSense 화면에서 표지판 후보를 찾음", loc="left", fontsize=14, fontweight="bold")
    a = fig.add_subplot(gs[0, 1]); a.imshow(M, cmap="gray", vmin=0, vmax=255); a.set_xlim(cb[0], cb[1]); a.set_ylim(cb[3], cb[2]); a.axis("off")
    draw_cands(a, F, t)
    a.set_title("② 같은 순간 위치추정 후보: 점수 비슷한 후보(파란 점)가 복도 여러 곳에 흩어짐\n    = '헷갈림' → 멈춰서 표지판을 봐도 됨   (★ = 실제 위치)", loc="left", fontsize=14, fontweight="bold")
    a = fig.add_subplot(gs[1, 1]); a.imshow(M, cmap="gray", vmin=0, vmax=255); a.set_xlim(cb[0], cb[1]); a.set_ylim(cb[3], cb[2]); a.axis("off")
    labs = {}
    for r, ans in zip(near, info["answers"]):
        px = np.array([float(r["x_px"]), float(r["y_px"])]); mp = px + ppm * A(math.radians(float(r["phi_deg"]))) @ (xy - o)
        a.scatter(*px, s=90, color=BLUE, edgecolor=SURF, zorder=3)
        a.add_patch(FancyArrowPatch(px, mp, arrowstyle="-|>", mutation_scale=13, color=ORANGE if ans else "#9aa3b0", lw=1.8 if ans else 1.2, zorder=2))
        if ans:
            labs.setdefault(ans, []).append(mp)
    for k, (ans, mps) in enumerate(sorted(labs.items())):
        mp = np.mean(mps, 0)
        a.annotate(f"{rooms[ans - 1].get('room_number', '?')}호 자리", mp, (mp[0] - 120 + 260 * k, mp[1] - 170), fontsize=13, fontweight="bold", color=ORANGE,
                   arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1), bbox=dict(facecolor=SURF, edgecolor=ORANGE, pad=3))
    a.scatter(*g, marker="*", s=380, color=RED, edgecolor=SURF, zorder=5)
    n0 = sum(1 for v in info["answers"] if v == 0)
    a.set_title(f"③ 후보마다 '내가 여기라면 그 표지판은 지도에서 어디?'를 계산 (화살표)\n    {', '.join(rooms[v - 1].get('room_number', '?') + '호' for v in sorted(labs))} 로 갈림 (회색 {n0}개는 호실 없는 벽) → 읽으면 후보가 가려짐",
                loc="left", fontsize=14, fontweight="bold")
    a = fig.add_subplot(gs[1, 0]); cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; h = max(bx[2] - bx[0], bx[3] - bx[1]) * 0.75
    X0, Y0 = int(max(0, cx - h)), int(max(0, cy - h)); cr = im[Y0:int(min(480, cy + h)), X0:int(min(640, cx + h))]
    a.imshow(cr); a.axis("off"); a.add_patch(plt.Rectangle((bx[0] - X0, bx[1] - Y0), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=ORANGE, lw=3))
    a.set_title("④ 채점: 멈춘 상자 안에 실제 번호판이 있나 → 있음\n    (4층은 프레임마다 정답 표시, 다른 층은 사람이 판정)", loc="left", fontsize=14, fontweight="bold")
    fig.suptitle("E4 실험이 한 순간에 하는 일 (4층 243초)", x=0.01, ha="left", fontsize=18, fontweight="bold")
    save(fig, "V1_흐름_4F243")


def v2():
    F = "4F"; M = basemap(F)
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    for a, t, title in ((ax[0], 60.0, "확실한 순간 (4층 60초): 비슷한 후보가 한 곳에 모여 있음 → 안 멈춤"),
                        (ax[1], 243.0, "헷갈리는 순간 (4층 243초): 비슷한 후보가 복도 여러 곳에 흩어짐 → 멈춤 허용")):
        a.imshow(M, cmap="gray", vmin=0, vmax=255); a.axis("off"); draw_cands(a, F, t); cb = content_box(F); a.set_xlim(cb[0], cb[1]); a.set_ylim(cb[3], cb[2])
        a.set_title(title, loc="left", fontsize=14, fontweight="bold", color=GREEN if t == 60 else BLUE)
    h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=11), Line2D([], [], marker="o", ls="", color="#9aa3b0", ms=8), Line2D([], [], marker="*", ls="", color=RED, ms=16)]
    fig.legend(h, ["1등과 점수가 비슷한 후보 (클수록 높음)", "나머지 후보", "실제 위치"], loc="lower center", ncol=3, frameon=False, fontsize=12)
    fig.suptitle("'헷갈림' 판단 = 점수 비슷한 후보가 5 m 넘게 떨어진 곳에도 있나", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "V2_확실vs헷갈림")


def floor_scores():
    cache = OUT / "floor_scores.json"
    if cache.exists():
        return json.load(open(cache))
    pat = re.compile(r"(\d+)초 \| 1위 중앙 [\d.]+ m, 3 m 안 (\d+)%")
    res = {}
    for F in FL:
        r = {}
        for k, d in (("none", L / F / "loc"), ("main", P / MAIN / F / FB), ("P_ALL", P / "P_ALL" / F / FB)):
            s = subprocess.run([sys.executable, "scripts/score_loc_pseudo_gt.py", str(R / "36_채점용정답" / F), f"x={d}"], capture_output=True, text=True).stdout
            m = pat.search(s); r[k] = int(m.group(2)) if m else None
        res[F] = r
    OUT.mkdir(parents=True, exist_ok=True); json.dump(res, open(cache, "w"), ensure_ascii=False, indent=1)
    return res


def v3(S, SC):
    fig, ax = plt.subplots(7, 2, figsize=(16, 22)); ax = ax.ravel()
    for i, F in enumerate(FL):
        a = ax[i]; M = basemap(F); a.imshow(M, cmap="gray", vmin=0, vmax=255); a.axis("off")
        path_by_state(a, F, S, lw=2.6)
        E = stops(MAIN, F); p = np.array([gt_at(F, e["t"]) for e in E]) if E else np.zeros((0, 2))
        if len(p):
            a.scatter(p[:, 0], p[:, 1], marker="v", s=70, color=INK, edgecolor=SURF, linewidth=0.8, zorder=4)
        tt, xy = gt(F); pad = 80
        a.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); a.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
        st = list(S[F].values()); conf = 100 * st.count("GEOMETRIC_CONFIDENT") / len(st)
        a.set_title(f"{fl(F)}   멈춤 {len(E)}번 · 확실 {conf:.0f}% · 위치 맞음 {SC[F]['none']}% → {SC[F]['main']}%", loc="left", fontsize=12, fontweight="bold")
    ax[13].axis("off")
    h = [Line2D([], [], color=STC[k], lw=5) for k in ("GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT")] + \
        [Line2D([], [], color="#c3c8d0", lw=5), Line2D([], [], marker="v", ls="", color=INK, ms=10)]
    ax[13].legend(h, ["확실한 구간", "헷갈리는 구간", "헷갈림 + 긴 직선 복도", "위치추정 시작 전 (처음 20초)", "표지판 보려고 멈춘 곳"], loc="center", frameon=False, fontsize=14)
    ax[13].text(0.5, 0.12, "위치 맞음 = 위치추정이 3 m 안인 시간 비율\n(멈춤 없음 → 멈춘 곳 번호를 읽었다면)", ha="center", fontsize=11, color=INK2, transform=ax[13].transAxes)
    fig.suptitle(f"13개 층 전부: 실제 걸은 경로를 그 순간 상태로 칠하고, 멈춘 곳을 찍음 ({MAIN_NAME})", x=0.01, ha="left", fontsize=18, fontweight="bold", y=1.0)
    fig.tight_layout(); save(fig, "V3_전층지도")


def v4(S):
    F = "4F"; M = basemap(F); tt, xy = gt(F); pad = 80
    pols = [("P_ALL", "표지판 보이면 다 멈춤"), ("P_AMBIGUITY", "헷갈릴 때만 멈춤 (10-08 판)")] + ([(MAIN, MAIN_NAME)] if MAIN != "P_AMBIGUITY" else [("P_INFORMATIVE_0.5bit", "헷갈림 + 읽으면 가려질 때만")])
    fig, ax = plt.subplots(3, 1, figsize=(15, 11.5))
    for a, (pol, nm) in zip(ax, pols):
        a.imshow(M, cmap="gray", vmin=0, vmax=255); a.axis("off"); path_by_state(a, F, S, lw=3)
        E = stops(pol, F); p = np.array([gt_at(F, e["t"]) for e in E])
        conf = sum(S[F].get(min(S[F], key=lambda k: abs(k - e["t"]))) == "GEOMETRIC_CONFIDENT" for e in E)
        a.scatter(p[:, 0], p[:, 1], marker="v", s=120, color=INK, edgecolor=SURF, linewidth=1, zorder=4)
        a.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad); a.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
        a.set_title(f"{nm}:  멈춤 {len(E)}번 (그중 확실한 구간에서 {conf}번)", loc="left", fontsize=14, fontweight="bold")
    legend_states(fig, loc="lower center", ncol=4, fontsize=12)
    fig.suptitle("같은 4층 경로, 규칙만 바꿨을 때 멈춘 곳 (▼)", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "V4_규칙비교_4F")


def v5(SC):
    order = sorted(FL, key=lambda F: SC[F]["main"] - SC[F]["none"])
    fig, a = plt.subplots(figsize=(11, 7.5)); y = np.arange(len(order))
    for yi, F in zip(y, order):
        n0, n1 = SC[F]["none"], SC[F]["main"]
        a.plot([n0, n1], [yi, yi], color="#c9ced6", lw=4, zorder=1, solid_capstyle="round")
        a.scatter(n0, yi, s=110, color="#8e96a3", zorder=2); a.scatter(n1, yi, s=150, color=BLUE, zorder=3)
        a.text(max(n0, n1) + 2, yi, f"+{n1 - n0}%p" if n1 >= n0 else f"{n1 - n0}%p", va="center", fontsize=11, color=INK)
    a.set_yticks(y); a.set_yticklabels([fl(F) for F in order], fontsize=12); a.set_xlim(0, 112); a.set_xlabel("위치추정이 3 m 안인 시간 비율 (%)")
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.grid(axis="x", color="#eceef1"); a.set_axisbelow(True)
    h = [Line2D([], [], marker="o", ls="", color="#8e96a3", ms=10), Line2D([], [], marker="o", ls="", color=BLUE, ms=12)]
    a.legend(h, ["멈춤 없음 (위치추정만)", f"{MAIN_NAME}로 멈춘 곳 번호를 읽었다면"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=11)
    a.set_title("층마다 표지판을 읽으면 위치추정이 얼마나 좋아지나", loc="left", fontsize=16, fontweight="bold", pad=34)
    save(fig, "V5_층별효과")


def v6():
    FI = P / "paper/fault_inject"
    rowsG = [("원래 오돔", json.load(open(L / "4F/모호함_v2.json")), L / "4F/loc"),
             ("튐 넣음 · 고치기 전", json.load(open(FI / "4F/모호함_v2.json")), FI / "4F/loc"),
             ("튐 넣음 · 고친 뒤", json.load(open(FI / "4F_repaired/모호함_v3.json")), FI / "4F_repaired/loc")]
    inj = json.load(open("/data/kci_work/e4_fault_inject_20261008/4F/INJECTION.json"))["plan"]
    fx = {int(float(r["rel_sec"])): r for r in csv.DictReader(open(P / "paper/csv/E4_fault_fix.csv"))}
    col = lambda g: YELLOW if g.get("sensor_bad") else (BLUE if g["amb"] else GREEN)
    fig = plt.figure(figsize=(15, 8.2)); gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 0.5, 0.5, 0.5], hspace=0.35)
    a0 = fig.add_subplot(gs[0]); ax = [fig.add_subplot(gs[i], sharex=a0) for i in (1, 2, 3)]
    GD = R / "36_채점용정답/4F"; ppm = json.load(open(GD / "summary.json"))["ppm"]; gtt, gxy = gt("4F")
    for (nm, G, loc), c, lw in zip(rowsG, ("#9aa3b0", RED, BLUE), (2.4, 1.3, 1.8)):
        top = {float(r["relative_sec"]): (float(r["x_px"]), float(r["y_px"])) for r in csv.DictReader(open(loc / "topk_series.csv")) if r["rank"] == "1"}
        ts = np.array(sorted(top)); err = [math.hypot(top[t][0] - gxy[np.argmin(abs(gtt - t))][0], top[t][1] - gxy[np.argmin(abs(gtt - t))][1]) / ppm for t in ts]
        a0.plot(ts, np.minimum(err, 40), color=c, lw=lw, label=nm)
    a0.set_ylim(0, 44); a0.set_ylabel("위치 오차 (m)"); a0.legend(loc="upper left", frameon=False, fontsize=11, ncol=3)
    for s_ in ("top", "right"):
        a0.spines[s_].set_visible(False)
    for a, (nm, G, _) in zip(ax, rowsG):
        for g in G:
            a.axvspan(g["t"] - 0.5, g["t"] + 0.5, color=col(g), lw=0)
        a.set_yticks([]); a.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=11)
        for s_ in a.spines.values():
            s_.set_visible(False)
    for j in inj:
        r = fx[int(j["rel_sec"])]
        nm = "+".join(x for x in ((f"{j['jump_m']:g}m" if j["jump_m"] else ""), (f"{j['jump_deg']:g}°" if j["jump_deg"] else "")) if x)
        for a in [a0] + ax:
            a.axvline(j["rel_sec"], color=INK, lw=1)
        a0.text(j["rel_sec"], 1.02, f"{nm}\n전 {'잡음' if r['detected_before'] == '1' else '못 잡음'} → 후 잡음", transform=a0.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.5, color=INK if r["detected_before"] == "1" else RED, fontweight="bold")
    ax[-1].set_xlabel("시간 (초)"); a0.set_xlim(20, 400)
    for a in [a0] + ax[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    fig.legend([Patch(color=GREEN), Patch(color=BLUE), Patch(color=YELLOW)], ["확실", "헷갈림", "센서 이상"], loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=12)
    fig.suptitle("4층 오돔에 일부러 튐 8개: 고친 뒤 8개 다 잡고, 튄 걸음을 고쳐 위치추정이 원래대로 돌아옴", x=0.01, ha="left", fontsize=15, fontweight="bold", y=1.0)
    save(fig, "V6_오돔튐")

if __name__ == "__main__":
    S = states(); SC = floor_scores()
    v1(); v2(); v3(S, SC); v4(S); v5(SC); v6()
