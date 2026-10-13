#!/usr/bin/env python3
"""E2·E3 평가 지표와 그림(2026-10-12, 지도 8차 결과). 새로 돌리지 않고 저장된 결과에서 계산한다. 정답(채점용, LiDAR 정합 기반)은 채점에만 쓴다.
  E2: Top-K 포함률(1·5·10·20·50·100), 정답 층 포함률@K, 1위 위치·방향·축척 오차, 정답 순위, 후보 층 분포, 처리 시간, 놓친 이유, 임계값 민감도
  E3: 시점별 1위·상위10·후보 안 포함(0초 = E2), E2 성공(A)·실패(B) 구간 나누기, 정답 생존·소실·복구(복구가 노드 재탐색에서 왔나), 수렴 시간, 후보 수, 처리 시간
    python3 scripts/e2e3_eval_v8.py
출력: reports/_11_E2E3_지도v8_평가_20261012/{E2,E3}/ (CSV·그림)
"""
import csv, json, math, sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV  # noqa: E402,F401  (글꼴)

G = Path("reports/_01_E2_초기위치/_작업자료"); R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/36_채점용정답")
E2J = G / "미세정합시험/T0_nospace_후보_지도v8"; E3D = Path("reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색_지도v8")
OUT = Path("reports/_11_E2E3_지도v8_평가_20261012"); (OUT / "E2").mkdir(parents=True, exist_ok=True); (OUT / "E3").mkdir(parents=True, exist_ok=True)
TASK = {t["task_id"]: t for t in csv.DictReader(open(G / "tasks/E2_tasks.csv"))}
GTT = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
wrap = lambda a: (a + 180) % 360 - 180
GREEN, AMBER, RED, GRAYC, BLUE, INK, INK2 = "#2E8B57", "#E8A33D", "#C0392B", "#C9CED6", "#3B6FB6", "#1F2328", "#57606A"
BN = {"kci": "정보관", "gong": "공학관", "sw": "소융대"}
FLOORS = [f"{i}F" for i in range(1, 10)] + ["G1F", "G2F", "S1F", "S6F"]
TYPES = ["S1", "S2", "S3", "S4", "S5"]
FNAME = {f: (f"정보관 {f[:-1]}층" if f[0].isdigit() else ("공학관 " if f[0] == "G" else "소융대 ") + f[1:-1] + "층") for f in FLOORS}
KS = [1, 5, 10, 20, 50, 100]
SHORT = {"S1": "직선복도", "S2": "꺾임", "S3": "갈림길", "S4": "계단·엘베", "S5": "막다른곳"}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def gt_at(k, dt=0.0):
    F = TASK[k]["floor"]; S = json.load(open(R / F / "summary.json"))
    gp = list(csv.DictReader(open(R / F / "warp_clean/gt_warp.csv"))); gt = np.array([float(q["relative_sec"]) for q in gp])
    gx = np.array([[float(q["x_px"]), float(q["y_px"])] for q in gp]); ts = float(TASK[k]["start_sec"])
    return gx[int(np.argmin(abs(gt - ts - dt)))], S["ppm"], S["theta_deg"] + S.get("beta_deg", 228.0)


# ---------------- E2 ----------------
e2 = {}
for k in TASK:
    j = json.load(open(E2J / f"{k}.json")); tops = j["top100"]["merit"]; m_true = int(GTT[k]["true_map"]); pg, ppm, yaw = gt_at(k)
    def ok(c, r=3.0, y=30.0, m_true=m_true, pg=pg, ppm=ppm, yaw=yaw):     # 기본값으로 묶어 늦은 참조 방지
        return int(c[0]) == m_true and math.hypot(c[1] - pg[0], c[2] - pg[1]) / ppm <= r and abs(wrap(c[3] - yaw)) <= y
    t1 = tops[0]; same = int(t1[0]) == m_true
    e2[k] = dict(j=j, tops=tops, ok=ok, m_true=m_true, pg=pg, ppm=ppm, yaw=yaw)
    rk = next((i + 1 for i, c in enumerate(tops) if ok(c)), None)
    tc = [c for c in tops if ok(c)]
    reason = "Top-100 안" if rk else ("벽 겹침 단계 1000위 밖(후보가 거의 안 생김)" if not j["rank_orig"] else "1000위 안에 있었으나 정밀 검증 후 100위 밖")
    e2[k]["row"] = dict(출발점=k, 건물=BN[TASK[k]["building"]], 층=TASK[k]["floor"], 장소=EV.TYPE[TASK[k]["start_type"]],
                        정답순위=rk or "", 정답층_첫순위=next((i + 1 for i, c in enumerate(tops) if int(c[0]) == m_true), ""),
                        벽겹침단계_정답순위=j["rank_orig"] or "1000밖", **{f"Top{K}": int(bool(rk and rk <= K)) for K in KS},
                        **{f"층맞음@{K}": int(any(int(c[0]) == m_true for c in tops[:K])) for K in KS},
                        일위_층맞음=int(same), 일위_위치오차_m=round(math.hypot(t1[1] - pg[0], t1[2] - pg[1]) / ppm, 2) if same else "",
                        일위_방향오차_도=round(abs(wrap(t1[3] - yaw)), 1) if same else "", 일위_축척오차_퍼센트=round(abs(t1[4] - ppm) / ppm * 100, 1),
                        정답후보_축척오차_퍼센트=round(abs(tc[0][4] - ppm) / ppm * 100, 1) if tc else "", 상위100_층수=len({int(c[0]) for c in tops}),
                        정밀검증_계산초=j["sec"], 놓친이유=reason)
rows = [e2[k]["row"] for k in TASK]; write(OUT / "E2/E2_출발점별.csv", rows)
n = len(rows); hit = {K: sum(r[f"Top{K}"] for r in rows) for K in KS}; fhit = {K: sum(r[f"층맞음@{K}"] for r in rows) for K in KS}
sens = []
for rr in (1.0, 2.0, 3.0):
    for yy in (10.0, 20.0, 30.0):
        h1 = sum(1 for k in TASK if e2[k]["ok"](e2[k]["tops"][0], rr, yy)); h100 = sum(1 for k in TASK if any(e2[k]["ok"](c, rr, yy) for c in e2[k]["tops"]))
        sens.append(dict(위치_m=rr, 방향_도=yy, Top1=h1, Top100=h100))
write(OUT / "E2/E2_임계값민감도.csv", sens)
write(OUT / "E2/E2_TopK.csv", [dict(K=K, 정답포함=hit[K], 정답층포함=fhit[K], 분모=n) for K in KS])
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(KS, [fhit[K] / n * 100 for K in KS], "-o", color=BLUE, lw=2, ms=8, label="정답 층 후보가 있음(층만)")
ax.plot(KS, [hit[K] / n * 100 for K in KS], "-o", color=GREEN, lw=2.5, ms=8, label="정답 후보가 있음(층 + 3 m + 30°)")
for K in KS:
    ax.annotate(f"{hit[K]}", (K, hit[K] / n * 100), textcoords="offset points", xytext=(0, -16), ha="center", color=GREEN, fontsize=10)
    ax.annotate(f"{fhit[K]}", (K, fhit[K] / n * 100), textcoords="offset points", xytext=(0, 8), ha="center", color=BLUE, fontsize=10)
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(K) for K in KS]); ax.set_ylim(0, 105); ax.set_xlabel("상위 K개 후보"); ax.set_ylabel("출발점 비율 (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#E5E7EB"); ax.legend(frameon=False, loc="upper left")
ax.set_title(f"E2 Top-K: 상위 K개 안에 정답이 있는 출발점 (65곳, 숫자 = 곳 수)", loc="left", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "E2/E2_TopK곡선.png", dpi=110); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for a, key in zip(axs, ("Top1", "Top100")):
    M = np.array([[next(s[key] for s in sens if s["위치_m"] == rr and s["방향_도"] == yy) for yy in (10, 20, 30)] for rr in (1, 2, 3)])
    a.imshow(M, cmap="Greens", vmin=0, vmax=65)
    for i in range(3):
        for jj in range(3):
            a.text(jj, i, f"{M[i, jj]}곳\n{M[i, jj] / 65 * 100:.0f}%", ha="center", va="center", color="white" if M[i, jj] > 35 else INK, fontsize=11)
    a.set_xticks(range(3)); a.set_xticklabels(["10°", "20°", "30°"]); a.set_yticks(range(3)); a.set_yticklabels(["1 m", "2 m", "3 m"])
    a.set_xlabel("방향 허용"); a.set_ylabel("위치 허용"); a.set_title(f"{'1위가 정답' if key == 'Top1' else '상위 100 안에 정답'}", loc="left", fontsize=12, fontweight="bold")
fig.suptitle("E2 정답 기준을 바꾸면 (기본 3 m·30°)", x=0.01, ha="left", fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "E2/E2_임계값민감도.png", dpi=110); plt.close(fig)
# E2 65칸 표: 놓친 이유
fig, ax = plt.subplots(figsize=(10, 11))
col = {"1위": GREEN, "Top-100 안": "#7FBF8E", "정밀 검증 후 100위 밖": AMBER, "1000위 밖": GRAYC}
cnt = Counter()
for i, F in enumerate(FLOORS):
    for jj in range(5):
        k = f"{F}_T{jj + 1}"; r = e2[k]["row"]; ty = TASK[k]["start_type"]
        c = "1위" if r["정답순위"] == 1 else ("Top-100 안" if r["정답순위"] else ("정밀 검증 후 100위 밖" if r["벽겹침단계_정답순위"] != "1000밖" else "1000위 밖")); cnt[c] += 1
        ax.add_patch(plt.Rectangle((jj, -i), 0.94, 0.9, color=col[c]))
        txt = f"{r['정답순위']}위" if r["정답순위"] else (f"벽겹침 {r['벽겹침단계_정답순위']}위" if c != "1000위 밖" else "후보 없음")
        ax.text(jj + 0.47, -i + 0.52, txt, ha="center", va="center", fontsize=10, fontweight="bold", color="white" if c == "1위" else INK)
        ax.text(jj + 0.47, -i + 0.17, SHORT[ty], ha="center", va="center", fontsize=7.5, color="white" if c == "1위" else INK2)
    ax.text(-0.08, -i + 0.45, FNAME[F], ha="right", va="center", fontsize=11)
for jj in range(5):
    ax.text(jj + 0.47, 1.05, f"출발점 {jj + 1}", ha="center", va="bottom", fontsize=10.5)
ax.set_xlim(-2.0, 5.1); ax.set_ylim(-12.3, 2.0); ax.axis("off")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in col.values()], labels=[f"{k_} {cnt[k_]}곳" for k_ in col], loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False, fontsize=10.5)
ax.set_title("E2 65개 출발점: 정답이 몇 위였나, 놓쳤다면 어디서 놓쳤나\n(벽겹침 n위 = 첫 단계에선 n위였는데 정밀 검증 뒤 100위 밖으로 밀림, 후보 없음 = 첫 단계부터 1000위 밖)", loc="left", fontsize=12.5, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "E2/E2_출발점별_정답순위.png", dpi=100); plt.close(fig)
e2sum = dict(n=n, hit=hit, fhit=fhit, cnt=dict(cnt), t1_pos=np.median([r["일위_위치오차_m"] for r in rows if r["일위_위치오차_m"] != ""]),
             t1_yaw=np.median([r["일위_방향오차_도"] for r in rows if r["일위_방향오차_도"] != ""]), t1_floor=sum(r["일위_층맞음"] for r in rows),
             sec=np.median([r["정밀검증_계산초"] for r in rows]), nfl=np.median([r["상위100_층수"] for r in rows]),
             scale_t1=np.median([r["일위_축척오차_퍼센트"] for r in rows]), scale_true=np.median([r["정답후보_축척오차_퍼센트"] for r in rows if r["정답후보_축척오차_퍼센트"] != ""]))

# ---------------- E3 ----------------
SC = {(r["trial_id"], int(r["T"])): r for r in csv.DictReader(open(E3D / "연결판_채점.csv"))}
CA = {}
for r in csv.DictReader(open(E3D / "연결판_남은후보.csv")):
    CA[(r["trial_id"], int(r["T"]), int(r["rank"]))] = r
TS = [15, 30, 45, 60]
A = [k for k in TASK if e2[k]["row"]["정답순위"]]; B = [k for k in TASK if not e2[k]["row"]["정답순위"]]
e3rows = []
for k in TASK:
    r0 = e2[k]["row"]; row = dict(출발점=k, 건물=r0["건물"], 층=r0["층"], 장소=r0["장소"], 집단="A(E2 성공)" if k in A else "B(E2 실패)",
                                   E2_정답순위=r0["정답순위"], E2_후보수=100)
    conv = ""
    for T in TS:
        s = SC[(k, T)]; tr = int(s["true_rank"]) if s["true_rank"] else None
        src = ""
        if tr:
            sid = CA[(k, T, tr)]["sid"]; src = "노드 재탐색" if sid.startswith("N") else f"E2 후보({sid}위)"
        row.update({f"{T}초_후보수": s["n_left"], f"{T}초_정답순위": tr or "", f"{T}초_1위정답": s["top1_ok"], f"{T}초_상위10": s["top10_ok"],
                    f"{T}초_후보안": int(bool(tr)), f"{T}초_정답출처": src})
    for T in TS:
        if all(SC[(k, t)]["top1_ok"] == "1" for t in TS if t >= T):
            conv = T; break
    row["수렴시점_초"] = conv; e3rows.append(row)
write(OUT / "E3/E3_출발점별_시점별.csv", e3rows)
def cnt3(ks, key, T):
    return sum(1 for r in e3rows if r["출발점"] in ks and str(r[f"{T}초_{key}"]) == "1")
tab = []
for grp, ks in (("전체", list(TASK)), ("A(E2 성공)", A), ("B(E2 실패)", B)):
    d0 = dict(집단=grp, 분모=len(ks), 시점="0초(E2)", 일위정답=sum(1 for k in ks if e2[k]["row"]["정답순위"] == 1), 상위10=sum(1 for k in ks if e2[k]["row"]["Top10"]),
              후보안=sum(1 for k in ks if e2[k]["row"]["Top100"]), 후보수_중앙=100)
    tab.append(d0)
    for T in TS:
        tab.append(dict(집단=grp, 분모=len(ks), 시점=f"{T}초", 일위정답=cnt3(ks, "1위정답", T), 상위10=cnt3(ks, "상위10", T), 후보안=cnt3(ks, "후보안", T),
                        후보수_중앙=int(np.median([int(r[f"{T}초_후보수"]) for r in e3rows if r["출발점"] in ks]))))
write(OUT / "E3/E3_시점별_집단별.csv", tab)
# 생존·소실(A)·복구(B)
surv = []
for T in TS:
    a_keep = sum(1 for r in e3rows if r["출발점"] in A and r[f"{T}초_후보안"] == 1)
    b_rec = sum(1 for r in e3rows if r["출발점"] in B and r[f"{T}초_후보안"] == 1)
    b_node = sum(1 for r in e3rows if r["출발점"] in B and r[f"{T}초_후보안"] == 1 and r[f"{T}초_정답출처"] == "노드 재탐색")
    a_node = sum(1 for r in e3rows if r["출발점"] in A and r[f"{T}초_후보안"] == 1 and r[f"{T}초_정답출처"] == "노드 재탐색")
    surv.append(dict(시점=f"{T}초", A_분모=len(A), A_정답유지=a_keep, A_그중_노드재탐색으로이어짐=a_node, B_분모=len(B), B_정답복구=b_rec, B_그중_노드재탐색에서=b_node))
write(OUT / "E3/E3_정답생존_복구.csv", surv)
# 처리 시간
nsec = [json.load(open(f)).get("sec", 0) for f in Path("/data2/e3_node_v8_20261012").glob("*/*.json")]
msec = [json.load(open(f)).get("sec", 0) for f in Path("/data2/e3_merged_v8_20261012").glob("*/*.json")]
e3time = dict(노드_과제x층_중앙초=float(np.median(nsec)) if nsec else None, 합치기_과제x층_중앙초=float(np.median(msec)) if msec else None)
# 그림 1: 시점별 곡선(전체·A·B)
fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
X = [0] + TS
for a, (grp, ks) in zip(axs, (("전체 65곳", list(TASK)), (f"A: E2 성공 {len(A)}곳", A), (f"B: E2 실패 {len(B)}곳", B))):
    rr = [t for t in tab if t["집단"].startswith(grp[0]) or (grp.startswith("전체") and t["집단"] == "전체")]
    rr = [t for t in tab if t["집단"] == ("전체" if grp.startswith("전체") else ("A(E2 성공)" if grp.startswith("A") else "B(E2 실패)"))]
    for key, c, lab in (("후보안", BLUE, "후보 안에 정답"), ("상위10", AMBER, "상위 10 안"), ("일위정답", GREEN, "1위가 정답")):
        v = [t[key] / len(ks) * 100 for t in rr]; a.plot(X, v, "-o", color=c, lw=2.5, ms=7, label=lab)
        for x_, t in zip(X, rr):
            a.annotate(str(t[key]), (x_, t[key] / len(ks) * 100), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5, color=c)
    a.set_xticks(X); a.set_xticklabels(["0초\n(E2)"] + [f"{T}초" for T in TS]); a.set_ylim(0, 105); a.grid(axis="y", color="#E5E7EB")
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_title(grp, loc="left", fontsize=13, fontweight="bold")
axs[0].set_ylabel("출발점 비율 (%)"); axs[0].legend(frameon=False, loc="lower right")
fig.suptitle("E3 합치기: 걸으면서 정답을 지키고(A) 놓친 정답을 되찾나(B) — 숫자 = 곳 수", x=0.01, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "E3/E3_시점별_A_B.png", dpi=105); plt.close(fig)
# 그림 2: B 구간 복구 출처
fig, ax = plt.subplots(figsize=(9, 4.6))
bn = [s["B_그중_노드재탐색에서"] for s in surv]; be = [s["B_정답복구"] - s["B_그중_노드재탐색에서"] for s in surv]
ax.bar(range(4), bn, color=BLUE, width=0.55, label="노드 재탐색이 새로 찾은 후보"); ax.bar(range(4), be, bottom=bn, color=AMBER, width=0.55, label="E2 후보가 걸으며 다듬어져 정답 조건(3 m·30°)에 들어옴")
for i, s in enumerate(surv):
    ax.text(i, s["B_정답복구"] + 0.5, f"{s['B_정답복구']}/{len(B)}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([s["시점"] for s in surv]); ax.set_ylim(0, len(B) + 3); ax.set_ylabel("정답을 되찾은 출발점 수")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="upper left"); ax.set_title(f"E3 B 구간(E2에서 정답을 놓친 {len(B)}곳): 걸으면서 정답이 후보로 돌아왔나, 어디서 왔나", loc="left", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "E3/E3_B구간_정답복구.png", dpi=110); plt.close(fig)
json.dump(dict(e2={k_: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k_, v in e2sum.items()}, A=len(A), B=len(B), surv=surv, tab=tab, time=e3time),
          open(OUT / "요약.json", "w"), ensure_ascii=False, indent=1, default=float)
print("E2", hit, fhit, dict(cnt)); print("A", len(A), "B", len(B)); print(surv); print(e3time)
