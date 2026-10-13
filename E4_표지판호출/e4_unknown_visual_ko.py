#!/usr/bin/env python3
"""E4 층 모름(지침서판, 합치기 후보) 이해용 시각 자료(2026-10-12). 배포판 C 판(e4_visual_ko.py V1~V6)과 같은 모양으로 만든다.
  1_한순간흐름   멈춘 한 순간: 카메라 화면 → 여러 층에 걸친 후보 → 가설마다 그 표지판이 몇 층 어느 호실인지 → 사람 판정과 효과
  2_확실vs헷갈림 위치 확정 순간과 층 간 모호 순간: 층마다 가장 잘 맞는 후보의 점수 차와 1위 층 지도
  3_전층지도     13개 층에 출발점 구간 실제 경로(상태별 색)와 멈춘 곳(사람 판정 색)
  4_규칙비교_4층 4층 출발점 5곳에서 멈춤 방식별 멈춘 곳
  5_층별효과     층마다 60초 위치 맞음(5곳 중): 걷기만 → 번호 읽으면 → 층을 알면
  6_오돔튐       오돔 튐 크기별 탐지: 1초 규칙 vs OdomGuard
결과: reports/_02_E4_표지판호출/1_본결과_층모름/시각자료/
    E4_VAR=merged python3 scripts/e4_unknown_visual_ko.py
"""
import csv, json, math, os, sys
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
os.environ.setdefault("E4_VAR", "merged")
from e4_figures import frame  # noqa: E402
import e2_visual_ko as EV  # noqa: E402
import e4_unknown_gate as UG  # noqa: E402
import e4_unknown_informative as UI  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); RUN = R / "70_층모름E4"; J = R / "25_판정"
SFX = {"merged": "_merged", "v8": "_v8"}.get(os.environ["E4_VAR"], "")
W = Path("/data2/e4_unknown_20261011"); GATE = W / f"gate{SFX}_d0.2"
OUT = Path("reports/_02_E4_표지판호출/1_본결과_층모름/시각자료" + ("_지도v8" if SFX == "_v8" else ""))
D = Path("/data/kci_work/maps_perfloor_20261001")
FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
NAME = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}
BLD = {"kci": "정보관", "gong": "공학관", "sw": "소융대"}
BLUE, ORANGE, GREEN, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#9aa3b0"
INK, INK2, SURF = "#14171c", "#5b6370", "#ffffff"
STC = {"GEOMETRIC_CONFIDENT": GREEN, "GEOMETRIC_AMBIGUITY": BLUE, "GEOMETRIC_INSUFFICIENT": ORANGE, "SENSOR_UNRELIABLE": YELLOW}
STK = {"GEOMETRIC_CONFIDENT": "확정 선언(위치 확정)", "GEOMETRIC_AMBIGUITY": "헷갈림(층 간·층 내)", "GEOMETRIC_INSUFFICIENT": "헷갈림 + 긴 직선 복도", "SENSOR_UNRELIABLE": "센서 이상"}
POLS = [("P_ALL", "표지판 보이면 다 멈춤", RUN / "P_ALL"), ("P_AMBIGUITY", "헷갈릴 때만 멈춤", RUN / f"P_AMBIGUITY_gate{SFX}_d0.2"), ("P_INFORMATIVE", "헷갈림 + 읽으면 갈릴 때만", None)]
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 11, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fl(F):
    return NAME.get(F, F.replace("F", "층"))


def flm(b, m):
    return f"{BLD[b]} {m}층"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / f"{name}.png", dpi=130, bbox_inches="tight"); plt.close(fig); print(name)


def faint(im):
    return 255 - (255 - im.astype(float)) * 0.45


def basemap(F):
    return faint(cv2.imread(str(D / F / "base_map.png"), cv2.IMREAD_GRAYSCALE))


def gt(F, _c={}):
    if F not in _c:
        g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        _c[F] = (np.array([float(r["relative_sec"]) for r in g]), np.array([[float(r["x_px"]), float(r["y_px"])] for r in g]))
    return _c[F]


def gt_at(F, t):
    tt, xy = gt(F); return xy[int(np.argmin(abs(tt - t)))]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


class Data:
    def __init__(self):
        self.segs = {s["task_id"]: s for s in csv.DictReader(open(W / "segments.csv"))}
        self.task = {t["task_id"]: t for t in csv.DictReader(open(EV.G / "tasks/E2_tasks.csv"))}
        self.gtt = {r["task_id"]: r for r in csv.DictReader(open(EV.G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
        self.sc = {(r["trial_id"], r["policy"]): r for r in csv.DictReader(open(RUN / f"채점{SFX}" / "출발점별_정책별.csv"))}
        self.st = {}
        for r in csv.DictReader(open(RUN / f"채점{SFX}" / "매초상태4종_d0.2.csv")):
            self.st.setdefault(r["trial_id"], {})[float(r["scene_sec"])] = r["state"]
        self.M = json.load(open(J / "판정목록.json")); Dj = json.load(open(J / "판정결과.json")); X = json.load(open(J / "층모름합치기_판정기록.json"))
        self.yes = {F: set(Dj.get(F, {}).get("plaque", [])) | set(Dj.get(F, {}).get("sign", [])) | set(X["맞음"].get(F, [])) for F in FL}
        self.judged = {F: set(Dj.get(F, {}).get("judged", range(1, Dj.get(F, {}).get("judged_max", 0) + 1))) | set(X["시트"].get(F, [])) for F in FL}

    def label(self, F, e):
        m = [it for it in self.M[F] if abs(it["t"] - float(e["t"])) <= 0.6 and iou(it["box"], e["picks"][0][:4]) > 0.3]
        if not m or m[0]["id"] not in self.judged[F]:
            return None
        return m[0]["id"] in self.yes[F]

    def stops(self, pol, k):
        if pol == "P_INFORMATIVE":
            keep = {round(x["t"], 3) for x in json.load(open(RUN / f"P_INFORMATIVE{SFX}_d0.2" / f"{k}.json")) if x["informative"] and x["kind"] == "번호판"}
            return [e for e in self.stops("P_AMBIGUITY", k) if round(float(e["t"]), 3) in keep]
        base = dict((p, b) for p, _, b in POLS)[pol]
        return json.load(open(base / k / "run/events_live_replay.json"))

    def hyp_all(self, k, T):
        J_ = [json.load(open(p)) for p in (UG.SEEDED / k).glob("*.json")]
        return UG.hypotheses(J_, T)


def floor_gaps(P):
    best = {}
    for c in P:
        best.setdefault(c["m"], c["merit"])
    top = P[0]["merit"]
    return {m: v - top for m, v in best.items()}


def u1(Dt):
    """멈춘 한 순간: 사람 판정 표지판, 정보성 있음, 남은 가설이 여러 층, 그 출발점은 번호 읽으면 맞고 걷기만 하면 틀림."""
    best = None
    for k, sg in Dt.segs.items():
        F = sg["floor"]; s_inf = Dt.sc.get((k, "P_INFORMATIVE"), {})
        if s_inf.get("oracle_ok_60") != "1" or s_inf.get("top1_ok_60") == "1":
            continue
        inf = {round(x["t"], 3): x for x in json.load(open(RUN / f"P_INFORMATIVE{SFX}_d0.2" / f"{k}.json")) if x["kind"] == "번호판" and x["informative"]}
        for e in Dt.stops("P_AMBIGUITY", k):
            x = inf.get(round(float(e["t"]), 3))
            if x and Dt.label(F, e) and e.get("xyz"):
                n = len(set(x["floors"]))
                if best is None or (n, -float(e["t"])) > (best[0], -float(best[2]["t"])):
                    best = (n, k, e, x)
    n, k, e, x = best
    sg = Dt.segs[k]; F = sg["floor"]; b = sg["building"]; ts = float(sg["start_sec"]); off = float(sg["off"]); m_true = int(Dt.gtt[k]["true_map"])
    hyp = json.load(open(GATE / f"{k}_가설.json")); h = [hh for hh in hyp if hh["avail"] <= float(e["t"]) + 1e-6][-1]
    OT, OXY, _ = UI.odom(F); o_now = OXY[int(np.argmin(abs(OT - (ts + h["T"]))))]; xy = np.array(e["xyz"][:2])
    near = sorted(h["near"], key=lambda c: c["merit"])
    fls = list(dict.fromkeys(c["m"] for c in near))[:6]
    im = frame(F, float(e["t"])); bx = e["picks"][0]
    fig = plt.figure(figsize=(19, 17)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.9], hspace=0.3, wspace=0.45)
    a = fig.add_subplot(gs[0, 0:2]); a.imshow(im); a.axis("off")
    a.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=ORANGE, lw=3))
    a.text(bx[0] - 10, bx[3] + 26, f"표지판 후보 · 거리 {e['z'][0]:.1f} m", color=ORANGE, fontsize=12, fontweight="bold", bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.9))
    a.set_title(f"① RealSense 화면에서 표지판 후보를 찾음 ({fl(F)} 출발점{k.split('_T')[1]}, 출발 뒤 {float(e['t']) - float(sg['seg_lo']):.0f}초)", loc="left", fontsize=14, fontweight="bold")
    a = fig.add_subplot(gs[0, 2:4]); P = Dt.hyp_all(k, h["T"]); gp = floor_gaps(P); ms = sorted(gp)
    yv = np.arange(len(ms))[::-1]
    for yi, m in zip(yv, ms):
        a.barh(yi, max(gp[m], 0.004), color=(BLUE if gp[m] <= 0.2 else "#c9ced6"), height=0.6)
        a.text(max(gp[m], 0.004) + 0.01, yi, f"{gp[m]:.2f}", va="center", fontsize=10.5, color=INK)
    a.axvline(0.2, color=INK2, ls="--", lw=1.2); a.text(0.205, yv[0] + 0.45, "Δ 0.2", fontsize=10, color=INK2)
    a.set_yticks(yv); a.set_yticklabels([flm(b, m) + ("  ★실제 층" if m == m_true else "") for m in ms], fontsize=11)
    for lab, m in zip(a.get_yticklabels(), ms):
        lab.set_color(GREEN if m == m_true else INK)
    for s_ in ("top", "right"):
        a.spines[s_].set_visible(False)
    a.set_xlabel("그 층에서 가장 잘 맞는 후보의 점수 − 1위 점수 (작을수록 비슷하게 맞음)")
    a.set_title(f"② 같은 순간 후보: {len(set(c['m'] for c in near))}개 층이 1위와 비슷하게 맞음(파랑) → '층 간 모호'", loc="left", fontsize=14, fontweight="bold")
    inner = gs[1, :].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.06, hspace=0.18)
    ans_of = {}
    for c, an in zip(h["near"], x["answers"]):
        ans_of.setdefault(c["m"], []).append((c, an))
    for j, m in enumerate(fls):
        a = fig.add_subplot(inner[j // 3, j % 3]); mi = faint(EV.mapimg(b, m)); a.imshow(mi, cmap="gray", vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
        pts = []
        for c, an in ans_of.get(m, []):
            p = np.array([c["x"], c["y"]]); mp = p + c["s"] * (UI.A(math.radians(c["hd"]) - UI.YAW_OFF) @ (xy - o_now)); pts += [p, mp]
            a.scatter(*p, s=90, color=BLUE, edgecolor=SURF, zorder=3)
            a.add_patch(FancyArrowPatch(p, mp, arrowstyle="-|>", mutation_scale=12, color=ORANGE if an != "0" else GRAY, lw=1.8, zorder=2))
        if m == m_true:
            g = gt_at(F, float(e["t"]) - off); a.scatter(*g, marker="*", s=300, color=RED, edgecolor=SURF, zorder=5); pts.append(g)
        pts = np.array(pts); s0 = np.median([c["s"] for c, _ in ans_of.get(m, [])] or [30]); r_ = 6 * s0
        a.set_xlim(pts[:, 0].min() - r_, pts[:, 0].max() + r_); a.set_ylim(pts[:, 1].max() + r_, pts[:, 1].min() - r_)
        an = [an for _, an in ans_of.get(m, [])]
        a.set_title(f"{flm(b, m)}" + (" ★" if m == m_true else "") + f"\n표지판 → {'호실 앞' if any(v != '0' for v in an) else '호실 없음'}", fontsize=11.5,
                    color=GREEN if m == m_true else INK, fontweight="bold")
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(GREEN if m == m_true else "#c9ced6"); sp.set_linewidth(3 if m == m_true else 1)
    a = fig.add_subplot(inner[:, 3]); cx, cy = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2; hh = max(bx[2] - bx[0], bx[3] - bx[1]) * 0.9
    X0, Y0 = int(max(0, cx - hh)), int(max(0, cy - hh)); cr = im[Y0:int(min(480, cy + hh)), X0:int(min(640, cx + hh))]
    a.imshow(cr); a.axis("off"); a.add_patch(plt.Rectangle((bx[0] - X0, bx[1] - Y0), bx[2] - bx[0], bx[3] - bx[1], fill=False, ec=ORANGE, lw=3))
    s_inf = Dt.sc[(k, "P_INFORMATIVE")]
    a.set_title("④ 사람 판정: 표지판 맞음\n번호를 읽으면 층이 정해짐 →\n60초 위치 " + ("맞음" if s_inf["oracle_ok_60"] == "1" else "틀림") +
                f" (걷기만 하면 {'맞음' if s_inf['top1_ok_60'] == '1' else '틀림'})", loc="left", fontsize=12.5, fontweight="bold", color=GREEN)
    fig.text(0.01, 0.592, "③ 멈추기 전에, 후보마다 '내가 여기라면 그 표지판은 지도에서 어디?'를 RealSense 깊이·상자 방향으로 계산(주황 = 호실 앞에 찍힘). 글자는 아직 안 읽음.\n    층이 다르면 번호 앞자리가 다르므로 읽으면 층이 갈린다 → 멈출 가치가 있음. (실제 판독과 팬틸트 방향으로 영역을 줄이는 것은 현장 실험 E5·E6)",
             fontsize=13.5, fontweight="bold")
    fig.suptitle("E4 층 모름: 표지판을 보려고 멈춘 한 순간에 하는 일", x=0.01, ha="left", fontsize=18, fontweight="bold")
    save(fig, "1_한순간흐름")


def u2(Dt):
    rows = list(csv.DictReader(open("reports/_02_E4_표지판호출/2_본결과_상태판단/계산별_상태.csv")))
    conf = [r for r in rows if r["상태"] == "위치 확정" and r["1위맞음"] == "1" and r["층"] == "4F"] or [r for r in rows if r["상태"] == "위치 확정" and r["1위맞음"] == "1"]
    amb = [r for r in rows if r["상태"] == "층 간 모호" and r["층"] == conf[0]["층"]]
    pick = [conf[len(conf) // 2], max(amb, key=lambda r: -float(r["다른층최소차"] or 9))]
    fig, ax = plt.subplots(2, 2, figsize=(16, 11), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for row, r in zip(ax, pick):
        k = r["출발점"]; T = int(r["시점초"]); sg = Dt.segs[k]; F = sg["floor"]; b = sg["building"]; m_true = int(Dt.gtt[k]["true_map"])
        P = Dt.hyp_all(k, T); gp = floor_gaps(P); ms = sorted(gp); yv = np.arange(len(ms))[::-1]; a = row[0]
        for yi, m in zip(yv, ms):
            a.barh(yi, max(gp[m], 0.004), color=BLUE if gp[m] <= 0.2 else "#c9ced6", height=0.6)
            a.text(max(gp[m], 0.004) + 0.01, yi, f"{gp[m]:.2f}", va="center", fontsize=10)
        a.axvline(0.2, color=INK2, ls="--", lw=1.2); a.set_yticks(yv); a.set_yticklabels([flm(b, m) + (" ★" if m == m_true else "") for m in ms])
        for lab, m in zip(a.get_yticklabels(), ms):
            lab.set_color(GREEN if m == m_true else INK)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
        a.set_xlabel("층별 최고 후보 점수 − 1위 (Δ 0.2 안 = 파랑)")
        top = P[0]; a2 = row[1]; mi = faint(EV.mapimg(b, top["m"])); a2.imshow(mi, cmap="gray", vmin=0, vmax=255); a2.axis("off")
        on = [c for c in P if c["m"] == top["m"]]
        for c in sorted(on, key=lambda c: -c["merit"]):
            isn = c["merit"] <= top["merit"] + 0.2
            a2.scatter(c["x"], c["y"], s=60 + (200 if c is top else 60), color=BLUE if isn else GRAY, alpha=0.9 if isn else 0.5, edgecolor=SURF, zorder=3)
        if top["m"] == m_true:
            g = gt_at(F, float(sg["start_sec"]) + T); a2.scatter(*g, marker="*", s=380, color=RED, edgecolor=SURF, zorder=5)
        EV.crop(a2, mi.astype(np.uint8), pad=30)
        okc = r["상태"] == "위치 확정"
        row[0].set_title(("확실한 순간: 다른 층은 모두 Δ 밖 → 안 멈춤" if okc else "헷갈리는 순간: 여러 층이 Δ 안 → 표지판이 보이면 멈춤"),
                         loc="left", fontsize=14, fontweight="bold", color=GREEN if okc else BLUE)
        a2.set_title(f"{fl(F)} 출발점{k.split('_T')[1]} · 출발 뒤 {T}초 · 1위 층({flm(b, top['m'])}) 지도의 후보  (★ = 실제 위치)", loc="left", fontsize=12)
    fig.suptitle("층 모름 '헷갈림' 판단 = 1위와 비슷하게 맞는 후보가 다른 층(또는 같은 층 5 m 밖)에도 있나", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "2_확실vs헷갈림")


def path_states(a, Dt, k, lw=3):
    sg = Dt.segs[k]; F = sg["floor"]; off = float(sg["off"]); tt, xy = gt(F); st = Dt.st.get(k, {})
    lo, hi = float(sg["start_sec"]), float(sg["start_sec"]) + 60
    sel = (tt >= lo) & (tt <= hi); a.plot(xy[sel, 0], xy[sel, 1], color="#c3c8d0", lw=lw * 0.8, zorder=1, solid_capstyle="round")
    for t0, s in st.items():
        tl = t0 - off; i0 = int(np.searchsorted(tt, tl - 0.5)); i1 = min(int(np.searchsorted(tt, tl + 0.5)) + 1, len(tt))
        if i1 - i0 >= 2:
            a.plot(xy[i0:i1, 0], xy[i0:i1, 1], color=STC[s], lw=lw, solid_capstyle="round", zorder=2)


COK, CFL, CNO = "#1B7F5B", "#E5B000", "#C0392B"


def path_correct(a, k, lw=3):
    """2026-10-12: 경로를 '그 시점 1위가 맞았나'로 칠한다(웹 화면처럼). 초록 = 1위가 정답(층 + 3 m + 30°), 노랑 = 1위 층만 맞음, 빨강 = 1위 층부터 틀림. 3초 계산마다 그 앞 3초 구간."""
    sg = DtG.segs[k]; F = sg["floor"]; tt, xy = gt(F); lo = float(sg["start_sec"])
    sel = (tt >= lo) & (tt <= lo + 60); a.plot(xy[sel, 0], xy[sel, 1], color="#c3c8d0", lw=lw * 0.8, zorder=1, solid_capstyle="round")
    for r in sorted((r for r in ACC if r["출발점"] == k), key=lambda r: float(r["시점초"])):
        T = float(r["시점초"]); i0 = int(np.searchsorted(tt, lo + T - 3)); i1 = min(int(np.searchsorted(tt, lo + T)) + 1, len(tt))
        c = COK if r["1위맞음"] == "1" else (CFL if r["1위층맞음"] == "1" else CNO)
        if i1 - i0 >= 2:
            a.plot(xy[i0:i1, 0], xy[i0:i1, 1], color=c, lw=lw, solid_capstyle="round", zorder=2)


def stop_marks2(a, Dt, k, pol, s=80):
    sg = Dt.segs[k]; F = sg["floor"]; off = float(sg["off"]); n = [0, 0]
    for e in Dt.stops(pol, k):
        lab = Dt.label(F, e); p = gt_at(F, float(e["t"]) - off)
        a.scatter(*p, marker="v", s=s, facecolor=INK if lab else "white", edgecolor=INK, linewidth=1.0, zorder=4)
        n[0] += 1; n[1] += bool(lab)
    return n


def full_path(a, F):
    """녹화 전체 경로를 옅은 회색으로(출발점 구간 밖 포함). 1초 넘게 빈 곳은 잇지 않는다."""
    tt, xy = gt(F)
    for seg in np.split(np.arange(len(tt)), np.where(np.diff(tt) > 1.0)[0] + 1):
        a.plot(xy[seg, 0], xy[seg, 1], color="#d9dde3", lw=2.0, zorder=0, solid_capstyle="round")
    return xy


def stop_marks(a, Dt, k, pol, s=90):
    sg = Dt.segs[k]; F = sg["floor"]; off = float(sg["off"]); n = [0, 0]
    for e in Dt.stops(pol, k):
        lab = Dt.label(F, e); p = gt_at(F, float(e["t"]) - off)
        a.scatter(*p, marker="v", s=s, color=GREEN if lab else (RED if lab is False else GRAY), edgecolor=INK, linewidth=0.6, zorder=4)
        n[0] += 1; n[1] += bool(lab)
    return n


def legend_u3(ax, **kw):
    h = [Line2D([], [], color=STC[k], lw=5) for k in ("GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT")] + [Line2D([], [], color="#d9dde3", lw=5)] + \
        [Line2D([], [], marker="v", ls="", color=GREEN, markeredgecolor=INK, ms=10), Line2D([], [], marker="v", ls="", color=RED, markeredgecolor=INK, ms=10)]
    return ax.legend(h, [STK[k] for k in ("GEOMETRIC_CONFIDENT", "GEOMETRIC_AMBIGUITY", "GEOMETRIC_INSUFFICIENT")] + ["출발점 구간 밖 주행(실험에 안 씀)", "멈춘 곳: 사람 판정 표지판", "멈춘 곳: 표지판 아님"], frameon=False, **kw)


ACC = list(csv.DictReader(open(RUN / f"E4_상태판단{SFX}" / "계산별_상태.csv"))) if (RUN / f"E4_상태판단{SFX}" / "계산별_상태.csv").exists() else []


DtG = None


def u3(Dt):
    global DtG
    DtG = Dt
    # 2026-10-13: 칸 높이를 층 지도 모양(세로/가로)에 맞춰 빈 공간을 줄임. 두 층씩 한 줄, 줄 높이 = 그 줄 두 층 중 큰 세로 비율
    pad = 80; box = {}
    for F in FL:
        xy = gt(F)[1]; box[F] = (xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    asp = [(box[F][3] - box[F][2]) / (box[F][1] - box[F][0]) for F in FL] + [0.5]
    hr = [max(asp[i], asp[i + 1]) for i in range(0, len(asp), 2)]
    H = 8 * sum(hr) + 0.75 * len(hr) + 1.0                    # 줄마다 두 줄 제목 자리 0.75 in, 맨 위 큰 제목 1 in
    fig = plt.figure(figsize=(16, H)); gs = fig.add_gridspec(len(hr), 2, height_ratios=hr, hspace=0.75 * len(hr) / (8 * sum(hr)), wspace=0.06, top=1 - 1.0 / H, bottom=0.005, left=0.01, right=0.99)
    ax = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(FL) + 1)]
    for i, F in enumerate(FL):
        a = ax[i]; a.imshow(basemap(F), cmap="gray", vmin=0, vmax=255); a.axis("off")
        ks = [k for k, s in Dt.segs.items() if s["floor"] == F]; n = [0, 0]; pts = full_path(a, F)
        for k in ks:
            path_correct(a, k, lw=2.6); m_ = stop_marks2(a, Dt, k, "P_INFORMATIVE", s=70); n[0] += m_[0]; n[1] += m_[1]
        a.set_xlim(box[F][0], box[F][1]); a.set_ylim(box[F][3], box[F][2]); a.set_anchor("W")
        st = [s for k in ks for s in Dt.st.get(k, {}).values()]; conf = 100 * st.count("GEOMETRIC_CONFIDENT") / max(len(st), 1)
        acc = 100 * np.mean([r["1위맞음"] == "1" for r in ACC if r["층"] == F]) if ACC else float("nan")   # 2026-10-12: 확정 선언과 1위 정답을 같이
        w0 = sum(Dt.sc[(k, "P_INFORMATIVE")]["top1_ok_60"] == "1" for k in ks); w1 = sum(Dt.sc[(k, "P_INFORMATIVE")]["oracle_ok_60"] == "1" for k in ks)
        a.set_title(f"{fl(F)}  멈춤 {n[0]}번(표지판 {n[1]})\n확정 선언 {conf:.0f}% · 1위 정답 {acc:.0f}% · 60초 위치 맞음 {w0}/5 → 읽으면 {w1}/5", loc="left", fontsize=12, fontweight="bold")
    ax[len(FL)].axis("off")
    h = [Line2D([], [], color=c, lw=5) for c in (COK, CFL, CNO)] + [Line2D([], [], color="#d9dde3", lw=5), Line2D([], [], marker="v", ls="", markerfacecolor=INK, markeredgecolor=INK, ms=10), Line2D([], [], marker="v", ls="", markerfacecolor="white", markeredgecolor=INK, ms=10)]
    ax[len(FL)].legend(h, ["1위가 정답(층 + 3 m + 30°)", "1위 층만 맞음(위치 틀림)", "1위 층부터 틀림", "출발점 구간 밖 주행(실험에 안 씀)", "멈춘 곳: 사람 판정 표지판", "멈춘 곳: 표지판 아님"], frameon=False, loc="upper left", fontsize=13)
    ax[len(FL)].text(0.02, 0.02, "층마다 출발점 5곳에서 각각 막 켠 상태로 60초씩 걸은 구간만 실험에 씀(색 칠한 부분)\n멈춤 = '헷갈림 + 읽으면 갈릴 때만' 방식", ha="left", va="bottom", fontsize=11, color=INK2, transform=ax[len(FL)].transAxes)
    fig.suptitle("13개 층 전부(층 모름): 출발점 구간 경로를 그 시점 1위가 맞았는지로 칠하고(초록 = 층·위치까지 정답), 멈춘 곳을 사람 판정으로 찍음", x=0.01, ha="left", fontsize=18, fontweight="bold", y=0.995)
    save(fig, "3_전층지도")


def u4(Dt):
    F = "4F"; ks = [k for k, s in Dt.segs.items() if s["floor"] == F]
    fig, ax = plt.subplots(3, 1, figsize=(15, 13))
    for a, (pol, nm, _) in zip(ax, POLS):
        a.imshow(basemap(F), cmap="gray", vmin=0, vmax=255); a.axis("off"); n = [0, 0]; pts = full_path(a, F)
        for k in ks:
            path_states(a, Dt, k, lw=3); m_ = stop_marks(a, Dt, k, pol, s=120); n[0] += m_[0]; n[1] += m_[1]
        pad = 80; a.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad); a.set_ylim(pts[:, 1].max() + pad, pts[:, 1].min() - pad)
        a.set_title(f"{nm}:  멈춤 {n[0]}번 (사람 판정 표지판 {n[1]})", loc="left", fontsize=14, fontweight="bold")
    legend_u3(fig, loc="lower center", ncol=6, fontsize=10.5)
    fig.suptitle("같은 4층 출발점 5곳(각 60초), 멈춤 방식만 바꿨을 때 멈춘 곳 (▼)", x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "4_규칙비교_4층")


def u5(Dt):
    rows = {F: [0, 0, 0] for F in FL}
    for (k, pol), r in Dt.sc.items():
        if pol == "P_INFORMATIVE":
            F = r["floor"]; rows[F][0] += int(r["top1_ok_60"]); rows[F][1] += int(r["oracle_ok_60"]); rows[F][2] += int(r["truefloor_ok_60"])
    order = sorted(FL, key=lambda F: (rows[F][1] - rows[F][0], rows[F][1]))
    fig, a = plt.subplots(figsize=(11, 7.8)); y = np.arange(len(order))
    for yi, F in zip(y, order):
        n0, n1, n2 = rows[F]
        a.plot([n0, max(n1, n2)], [yi, yi], color="#c9ced6", lw=4, zorder=1, solid_capstyle="round")
        a.scatter(n2, yi, s=170, facecolor="none", edgecolor=GREEN, linewidth=2, zorder=2)
        a.scatter(n0, yi, s=110, color="#8e96a3", zorder=3); a.scatter(n1, yi, s=150, color=BLUE, zorder=4)
        a.text(5.25, yi, f"{n0} → {n1}  (층 알면 {n2})", va="center", fontsize=11, color=INK)
    a.set_yticks(y); a.set_yticklabels([fl(F) for F in order], fontsize=12); a.set_xlim(-0.3, 7.2); a.set_xticks(range(6)); a.set_xlabel("60초 때 위치 맞은 출발점 (5곳 중)")
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    a.grid(axis="x", color="#eceef1"); a.set_axisbelow(True)
    h = [Line2D([], [], marker="o", ls="", color="#8e96a3", ms=10), Line2D([], [], marker="o", ls="", color=BLUE, ms=12),
         Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=GREEN, ms=13)]
    a.legend(h, ["걷기만 (합치기, 3초 갱신)", "첫 진짜 번호판 번호를 읽었다면", "층을 처음부터 안다면(상한)"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=11)
    tot = np.array(list(rows.values())).sum(0)
    a.set_title(f"층 모름: 층마다 번호판 하나 읽으면 60초 위치가 얼마나 맞나 (65곳 합 {tot[0]} → {tot[1]}, 상한 {tot[2]})", loc="left", fontsize=15, fontweight="bold", pad=34)
    save(fig, "5_층별효과")


def u6():
    rows = list(csv.DictReader(open("reports/_02_E4_표지판호출/2_본결과_상태판단/센서주입_출발점별.csv")))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    for a, kind, unit in ((ax[0], "이동(m)", "순간 이동 크기 (m)"), (ax[1], "회전(°)", "순간 회전 크기 (°)")):
        sizes = sorted({float(r["size"]) for r in rows if r["kind"] == kind})
        for lab, key, filt, col, ls in (("예전 1초 규칙(2 m/s·120°/s)", "rule", lambda r: True, GRAY, "-"), ("OdomGuard (전체 65곳)", "guard", lambda r: True, BLUE, "-"),
                                        ("OdomGuard (자이로 있는 층)", "guard", lambda r: r["gyro"] == "1", BLUE, "--")):
            v = [100 * np.mean([int(r[key]) for r in rows if r["kind"] == kind and float(r["size"]) == s and filt(r)]) for s in sizes]
            a.plot(sizes, v, color=col, ls=ls, lw=2.4, marker="o", label=lab)
        a.set_ylim(-3, 103); a.set_xlabel(unit); a.set_ylabel("튐을 잡은 출발점 (%)"); a.grid(color="#eceef1")
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    ax[0].legend(frameon=False, fontsize=11, loc="lower right")
    base = [r for r in rows if r["kind"] == "정상"]
    fig.suptitle(f"출발점 구간 30초에 오돔 튐을 일부러 넣었을 때 센서 이상으로 잡나 (정상 주행 오탐: 1초 규칙 {sum(int(r['rule']) for r in base)}번, "
                 f"OdomGuard {sum(int(r['guard']) for r in base)}번 = 모두 실제 오돔 끊김)", x=0.01, ha="left", fontsize=14, fontweight="bold")
    save(fig, "6_오돔튐")


if __name__ == "__main__":
    Dt = Data()
    for f in (u1, u2, u3, u4, u5):
        f(Dt)
    u6()
