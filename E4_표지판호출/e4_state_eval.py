#!/usr/bin/env python3
"""E4(새 계획서 6장) 위치 모호성·센서 상태 판단 평가(2026-10-12, PILOT). 카메라·멈춤은 다루지 않는다(E5).
입력: E3 연결판 3초 후보(65개 출발점 × 3~60초), 위치추정 오돔. 정답(채점용 정답 36_)은 채점에만 쓴다.

상태 5가지(우선순위 순):
  센서 이상   : OdomGuard(sensor_check.py: 0.1 s 오돔 걸음 3 m/s·150°/s, 오돔 끊김 1 s, 자이로와 회전 맞춤 9°) 사건이 직전 3초 안
  관측 부족   : 애매하면서 최근 3초 벽 방향 다양성 중앙 < 0.1(긴 직선 복도)
  층 간 모호  : 1위와 merit Δ 안에 다른 층 가설이 있음, 또는 최근 K초 안에 1위의 층이 바뀜
  층 내 모호  : Δ 안의 다른 가설이 모두 같은 층 5 m 밖, 또는 최근 K초 안에 같은 층 5 m 넘게 1위가 옮김
  위치 확정   : 그 밖
판단 방식 비교(같은 후보·같은 시점):
  다층(제안)   : 위 정의(Δ + 1위 이동 뒤 K초)
  다층(점수만) : 이동 뒤 K초 없이 Δ 만
  단일 층(C식) : 1위와 같은 층의 5 m 밖 가설만 경쟁 후보로 봄(다른 층 무시) — 기존 층 아는 판 판단을 층 모름 후보에 그대로 쓴 것
채점: '모호해야 하는 순간' = 그 시점 1위가 틀림(정답 층·3 m·30° 를 못 맞춤). 애매 판정을 이 순간의 탐지기로 보고
  정밀도 = 애매일 때 실제로 틀린 비율, 재현율 = 틀린 순간 중 애매로 잡은 비율, 잘못된 확신 = 확정인데 틀림(전체 대비),
  불필요 이벤트 = 1위가 맞는데 애매(맞은 순간 대비). 지연 = 맞던 1위가 틀어진 계산부터 첫 애매 판정까지(초).
  정답 층 포함 = 층 간 모호일 때 Δ 안 가설 집합에 정답 층이 있는 비율(의미정보로 가를 수 있는 상태인가).
센서 이상 탐지: 출발점 구간 오돔 중간(30초)에 순간 이동(0.25~3 m)·순간 회전(10~90°)을 넣어 그 초에 센서 이상으로 잡는지.
    python3 scripts/e4_state_eval.py
"""
import csv, json, math, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e4_unknown_gate as UG  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); OUT = R / "70_층모름E4" / f"E4_상태판단{UG.SFX}"
W = Path("/data2/e4_unknown_20261011"); SEEDED = UG.SEEDED; G = Path("reports/_01_E2_초기위치/_작업자료")
GC = R / "59_E4_호출정책" / "paper_C" / "csv"
wrap = lambda a: (a + 180) % 360 - 180


def load():
    segs = {s["task_id"]: s for s in csv.DictReader(open(W / "segments.csv"))}
    GTt = {r["task_id"]: r for r in csv.DictReader(open(G / "tasks/tasks_gt_EVAL_ONLY.csv"))}
    e2 = {}
    for r in csv.DictReader(open(UG.E2TOP)):
        e2.setdefault(r["trial_id"], []).append(r)
    import sensor_check as SCK
    SCJ = R / "59_E4_호출정책" / "paper" / "sensor_check"
    recs, gcache, ocache, gevents = [], {}, {}, {}
    for k, sg in segs.items():
        F = sg["floor"]; ts = float(sg["start_sec"]); off = float(sg["off"]); m_true = int(GTt[k]["true_map"])
        S = json.load(open(R / "36_채점용정답" / F / "summary.json")); ppm = S["ppm"]; hd_gt = S["theta_deg"] + S.get("beta_deg", 228.0)
        g = list(csv.DictReader(open(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")))
        gt = np.array([float(x["relative_sec"]) for x in g]); gx = np.array([[float(x["x_px"]), float(x["y_px"])] for x in g])
        if F not in gcache:
            gc = list(csv.DictReader(open(GC / f"geom_cells_{F}.csv")))
            gcache[F] = (np.array([float(r["relative_sec"]) for r in gc]), np.array([float(r["normal_diversity"]) if r["normal_diversity"] else np.nan for r in gc]))
            rows_o = list(csv.DictReader(open(UG.data_dir(F) / "odom_link.csv")))
            OT = np.array([float(r["t_sec"]) for r in rows_o])
            ocache[F] = (OT - OT[0], np.array([float(r["fastlio_x"]) for r in rows_o]), np.array([float(r["fastlio_y"]) for r in rows_o]), np.array([float(r["fastlio_yaw_rad"]) for r in rows_o]))
            t_, x_, y_, w_ = SCK.load_odom(str(UG.data_dir(F) / "odom_link.csv")); bag = json.load(open(SCJ / f"{F}.json")).get("bag")
            ev, _ = SCK.check(t_, x_, y_, w_, SCK.load_gyro(bag) if bag else None)
            gevents[F] = [(e[0] - t_[0], e[1]) for e in ev]               # OdomGuard 사건(오돔 첫 샘플 기준 초, 이유)
        gt_t, gdv = gcache[F]
        J = [json.load(open(p)) for p in (SEEDED / k).glob("*.json")]
        r1 = min(e2[k], key=lambda r: int(r["rank"]))
        prev = dict(m=int(r1["floor_id"]), x=float(r1["x_px"]), y=float(r1["y_px"]), hd=float(r1["heading_deg"]), s=float(r1["ppm"]))
        for T in range(3, 61, 3):
            P = UG.hypotheses(J, T)
            if not P:
                continue
            top = P[0]; pg = gx[int(np.argmin(abs(gt - (ts + T))))]
            ok = lambda c: c["m"] == m_true and math.hypot(c["x"] - pg[0], c["y"] - pg[1]) / ppm <= 3.0 and abs(wrap(c["hd"] - hd_gt)) <= 30
            gaps_other = [c["merit"] - top["merit"] for c in P[1:] if c["m"] != top["m"]]
            gaps_same = [c["merit"] - top["merit"] for c in P[1:] if c["m"] == top["m"] and UG.distinct(top, c)]
            m_tw = (np.datetime64 if False else None)
            sw_floor = prev["m"] != top["m"]; sw_place = (not sw_floor) and UG.distinct(prev, top)
            t = ts + T
            msk = (gt_t > t - 3.0) & (gt_t <= t + 1e-6) & np.isfinite(gdv); div = float(np.median(gdv[msk])) if msk.any() else None
            near_floors = lambda d: sorted({c["m"] for c in P if c["merit"] <= top["merit"] + d})
            sens = [why for (te, why) in gevents[F] if t - 3.0 < te <= t + 1e-6]
            recs.append(dict(k=k, F=F, b=sg["building"], T=T, t=t, ok=ok(top), floor_ok=top["m"] == m_true, sensor=sens,
                             go=min(gaps_other) if gaps_other else None, gs=min(gaps_same) if gaps_same else None,
                             sw_floor=sw_floor, sw_place=sw_place, div=div, m_true=m_true, P=P, top=top))
            prev = dict(m=top["m"], x=top["x"], y=top["y"], hd=top["hd"], s=top["s"])
    return segs, recs, ocache


def classify(recs, delta, K, mode):
    """mode: 'multi'(제안), 'multi_score', 'single'. 출발점별로 이동 뒤 K초를 이어서 셈."""
    out = []; last_sw = {}
    for r in recs:
        if r["T"] == 3:
            last_sw[r["k"]] = (None, None)
        if mode == "multi" and (r["sw_floor"] or r["sw_place"]):
            last_sw[r["k"]] = (r["T"], "inter" if r["sw_floor"] else "intra")
        inter = mode != "single" and r["go"] is not None and r["go"] <= delta
        intra = r["gs"] is not None and r["gs"] <= delta
        swT, swkind = last_sw.get(r["k"], (None, None))
        by_sw = mode == "multi" and swT is not None and r["T"] - swT < K
        if inter or (by_sw and swkind == "inter"):
            st = "층 간 모호"
        elif intra or (by_sw and swkind == "intra"):
            st = "층 내 모호"
        else:
            st = "위치 확정"
        if st != "위치 확정" and r["div"] is not None and r["div"] < 0.10:
            st = "관측 부족"
        if r.get("sensor"):
            st = "센서 이상"
        out.append(st)
    return out


def metrics(recs, states):
    amb = np.array([s not in ("위치 확정", "센서 이상") for s in states]); wrong = np.array([not r["ok"] for r in recs]); n = len(recs)
    tp = int((amb & wrong).sum()); fp = int((amb & ~wrong).sum()); fn = int((~amb & wrong).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    conf = ~amb
    # 지연: 출발점마다 맞던 1위가 틀어진 계산부터 첫 애매까지
    lat = []; byk = collections.defaultdict(list)
    for i, r in enumerate(recs):
        byk[r["k"]].append(i)
    for k, idx in byk.items():
        for a, b in zip(idx[:-1], idx[1:]):
            if recs[a]["ok"] and not recs[b]["ok"]:
                d = None
                for j in idx[idx.index(b):]:
                    if amb[j]:
                        d = recs[j]["T"] - recs[b]["T"]; break
                    if recs[j]["ok"]:
                        break
                lat.append(d)
    lat_found = [x for x in lat if x is not None]
    return dict(n=n, amb_rate=amb.mean(), precision=prec, recall=rec, f1=f1, missed=1 - rec,
                false_event=fp / max((~wrong).sum(), 1), wrong_confident=int((conf & wrong).sum()) / n,
                confident_acc=(conf & ~wrong).sum() / max(conf.sum(), 1), conf_rate=conf.mean(),
                onsets=len(lat), onset_caught=len(lat_found), lat_median=float(np.median(lat_found)) if lat_found else float("nan"))


def sensor_injection(segs, ocache):
    """출발점 구간 30초에 오돔 순간 이동·회전을 넣고 make_gate_json 과 같은 규칙(1초 전 대비 2 m/s, 120°/s)으로 그 초에 잡는지."""
    res = []
    for kind, sizes in (("이동(m)", [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), ("회전(°)", [10, 30, 60, 90, 120, 150, 180])):
        for sz in sizes:
            hit = 0; n = 0; base_fp = 0
            for k, sg in segs.items():
                OT, OX, OY, OYAW = ocache[sg["floor"]]; t_inj = float(sg["start_sec"]) + 30.0
                i0 = int(np.argmin(abs(OT - (t_inj - 1.0)))); i1 = int(np.argmin(abs(OT - t_inj)))
                dt = OT[i1] - OT[i0]
                if dt <= 0:
                    continue
                dx, dy = OX[i1] - OX[i0], OY[i1] - OY[i0]; dyaw = OYAW[i1] - OYAW[i0]
                sp0 = math.hypot(dx, dy) / dt; yr0 = abs(math.degrees(math.atan2(math.sin(dyaw), math.cos(dyaw)))) / dt
                base_fp += (sp0 > 2.0 or yr0 > 120.0)
                if kind.startswith("이동"):
                    sp = math.hypot(dx + sz, dy) / dt; yr = yr0                       # 진행과 무관한 방향(x)으로 순간 이동
                else:
                    dd = dyaw + math.radians(sz); sp = sp0; yr = abs(math.degrees(math.atan2(math.sin(dd), math.cos(dd)))) / dt
                hit += (sp > 2.0 or yr > 120.0); n += 1
            res.append(dict(kind=kind, size=sz, n=n, detected=hit, rate=hit / max(n, 1), base_false=base_fp))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segs, recs, ocache = load()
    D, K = 0.2, 10.0
    lines = [f"# E4 위치 상태 판단 평가 (65개 출발점, 3초 계산 {len(recs)}번, 파일럿)\n",
             "'모호해야 하는 순간' = 그 시점 연결판 1위가 틀림(정답 층·3 m·30°). 정답은 채점에만 씀.\n"]
    # 1) 상태 5가지 분포(제안)
    st = classify(recs, D, K, "multi")
    cnt = collections.Counter(st); n = len(st)
    lines += ["## 1. 상태 분포 (제안 방식, Δ 0.2·이동 뒤 10초)\n", "| 상태 | 비율 | 그 상태에서 1위 맞음 | 그 상태에서 정답 층이 Δ 안 가설에 있음 |", "|---|---|---|---|"]
    for s in ("위치 확정", "층 간 모호", "층 내 모호", "관측 부족", "센서 이상"):
        idx = [i for i, x in enumerate(st) if x == s]
        okr = np.mean([recs[i]["ok"] for i in idx]) if idx else float("nan")
        inc = np.mean([recs[i]["m_true"] in {c["m"] for c in recs[i]["P"] if c["merit"] <= recs[i]["top"]["merit"] + D} for i in idx]) if idx else float("nan")
        lines.append(f"| {s} | {cnt[s]} ({100 * cnt[s] / n:.1f}%) | {100 * okr:.0f}% | {100 * inc:.0f}% |")
    lines.append("\n센서 이상은 OdomGuard(sensor_check.py) 사건이 그 계산 직전 3초 안에 있으면 붙인다(우선순위 가장 높음). 65곳에서 잡힌 사건은 모두 오돔 끊김이다.\n")
    # 2) 판단 방식 비교
    lines += ["## 2. 판단 방식 비교 (같은 후보·같은 시점)\n",
              "| 방식 | 애매 비율 | 정밀도 | 재현율 | F1 | 놓친 모호성 | 불필요 이벤트(맞을 때 애매) | 잘못된 확신(전체 대비) | 확정일 때 맞음 | 틀어진 순간 잡음 / 지연 중앙 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    rows_cmp = []
    for name, mode in (("단일 층(C식: 같은 층 5 m 밖만)", "single"), ("다층(점수만)", "multi_score"), ("다층 + 이동 뒤 10초(제안)", "multi")):
        mt = metrics(recs, classify(recs, D, K, mode)); rows_cmp.append((name, mt))
        lines.append(f"| {name} | {100 * mt['amb_rate']:.0f}% | {100 * mt['precision']:.0f}% | {100 * mt['recall']:.0f}% | {mt['f1']:.2f} | {100 * mt['missed']:.0f}% | "
                     f"{100 * mt['false_event']:.0f}% | {100 * mt['wrong_confident']:.1f}% | {100 * mt['confident_acc']:.0f}% | {mt['onset_caught']}/{mt['onsets']} · {mt['lat_median']:.0f}초 |")
    # 3) Δ 곡선
    lines += ["\n## 3. 기준값 Δ 를 바꾸면 (제안 방식 / 단일 층)\n", "| Δ | 제안: 정밀도·재현율·잘못된 확신 | 단일 층: 정밀도·재현율·잘못된 확신 |", "|---|---|---|"]
    sweep = []
    for d in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6):
        a = metrics(recs, classify(recs, d, K, "multi")); b = metrics(recs, classify(recs, d, K, "single")); sweep.append((d, a, b))
        lines.append(f"| {d} | {100 * a['precision']:.0f}% · {100 * a['recall']:.0f}% · {100 * a['wrong_confident']:.1f}% | {100 * b['precision']:.0f}% · {100 * b['recall']:.0f}% · {100 * b['wrong_confident']:.1f}% |")
    # 4) 건물별
    lines += ["\n## 4. 건물별 (제안 방식)\n", "| 건물 | 계산 수 | 1위 틀림 비율 | 재현율 | 잘못된 확신 | 층 간 모호 비율 |", "|---|---|---|---|---|---|"]
    for bld, nm in (("kci", "정보관"), ("gong", "공학관"), ("sw", "소융대")):
        idx = [i for i, r in enumerate(recs) if r["b"] == bld]; rr = [recs[i] for i in idx]; ss = [st[i] for i in idx]
        mt = metrics(rr, ss)
        lines.append(f"| {nm} | {len(rr)} | {100 * np.mean([not r['ok'] for r in rr]):.0f}% | {100 * mt['recall']:.0f}% | {100 * mt['wrong_confident']:.1f}% | {100 * np.mean([s == '층 간 모호' for s in ss]):.0f}% |")
    # 5) 센서 이상 주입
    inj = sensor_injection(segs, ocache)
    lines += ["\n## 5. (참고) 예전 1초 규칙만으로 본 센서 이상 탐지 — 6절 OdomGuard 와 비교용\n", "| 종류 | 크기 | 탐지 | 주입 안 했을 때 그 초 오탐 |", "|---|---|---|---|"]
    for x in inj:
        lines.append(f"| {x['kind']} | {x['size']} | {x['detected']}/{x['n']} ({100 * x['rate']:.0f}%) | {x['base_false']} |")
    # 층 간 모호 판정 따로: 1위 층이 틀린 순간을 잡나
    fw = np.array([not r["floor_ok"] for r in recs]); inter = np.array([x == "층 간 모호" for x in st]); ambx = np.array([x in ("층 간 모호", "층 내 모호", "관측 부족") for x in st])
    infl = np.array([r["m_true"] in {c["m"] for c in r["P"] if c["merit"] <= r["top"]["merit"] + D} for r in recs])
    lines += ["\n## 4-1. 층 판정 쪽으로 따로 본 결과\n",
              f"- 1위 층이 틀린 계산 {int(fw.sum())}번 중 '층 간 모호'로 잡은 것 {int((fw & inter).sum())}번({100 * (fw & inter).sum() / max(fw.sum(), 1):.0f}%), 애매(층 간·층 내·관측 부족) 전체로는 {int((fw & ambx).sum())}번({100 * (fw & ambx).sum() / max(fw.sum(), 1):.0f}%)",
              f"- 1위 층이 틀렸는데 '위치 확정'이라고 한 것 {int((fw & (np.array(st) == '위치 확정')).sum())}번",
              f"- '층 간 모호'일 때 1위 층이 실제로 틀린 비율 {100 * (fw & inter).sum() / max(inter.sum(), 1):.0f}%, 정답 층이 Δ 안 가설 집합에 들어 있는 비율 {100 * (inter & infl).sum() / max(inter.sum(), 1):.0f}%",
              "  (정답 층이 집합 안에 있어야 다음 단계에서 번호판 하나로 층을 가를 수 있다. 집합 밖이면 의미정보를 읽어도 후보 재탐색이 필요하다)"]
    # E5 로 넘길 관측 필요 이벤트 목록
    ev_rows = []
    for r, x in zip(recs, st):
        if x not in ("층 간 모호", "층 내 모호", "관측 부족"):
            continue
        nr = [c for c in r["P"] if c["merit"] <= r["top"]["merit"] + D]; fl = sorted({c["m"] for c in nr})
        ev_rows.append(dict(출발점=r["k"], 층=r["F"], 시점초=r["T"], 상태=x, 일위층=r["top"]["m"], 애매가설수=len(nr), 가설층=" ".join(map(str, fl)),
                            번호판이_가를수있음=int(len(nr) >= 2 or bool(r["sw_floor"] or r["sw_place"])), 한층안애매_계단도_도움가능=int(len(fl) == 1)))
    with open(OUT / "E5전달_관측필요이벤트.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ev_rows[0])); w.writeheader(); w.writerows(ev_rows)
    lines += [f"\n## 7. E5 로 넘기는 관측 필요 이벤트\n", f"`E5전달_관측필요이벤트.csv`: 애매(층 간·층 내·관측 부족)이고 센서 정상인 계산 {len(ev_rows)}번. "
              f"번호판이 가를 수 있는 경우 {sum(e['번호판이_가를수있음'] for e in ev_rows)}번, 애매가 한 층 안이라 계단·시설 위치도 도움이 될 수 있는 경우 {sum(e['한층안애매_계단도_도움가능'] for e in ev_rows)}번. "
              "E4 는 정지를 결정하지 않는다. 실제 정지는 E5 가 이 목록과 영상 검출을 합쳐 정한다(새 계획서 6.6)."]
    txt = "\n".join(lines); (OUT / "E4_상태판단_평가.md").write_text(txt + "\n"); print(txt)
    with open(OUT / "계산별_상태.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["출발점", "층", "시점초", "상태", "1위맞음", "1위층맞음", "다른층최소차", "같은층최소차", "1위층바뀜", "같은층이동", "벽방향다양성"])
        for r, s in zip(recs, st):
            w.writerow([r["k"], r["F"], r["T"], s, int(r["ok"]), int(r["floor_ok"]), "" if r["go"] is None else round(r["go"], 4), "" if r["gs"] is None else round(r["gs"], 4),
                        int(r["sw_floor"]), int(r["sw_place"]), "" if r["div"] is None else round(r["div"], 4)])
    json.dump(dict(sweep=[(d, a, b) for d, a, b in sweep], cmp=rows_cmp, inj=inj), open(OUT / "평가값.json", "w"), ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
