#!/usr/bin/env python3
"""E4 오돔 튐 시험, 고친 판단 비교(2026-10-08, PILOT). 같은 4층 튐 8개에 대해 세 조건을 비교한다.
  원래        튐 없는 오돔, 기존 게이트
  고치기 전    튐 넣은 오돔, 1초 속도 기준(2 m/s, 120°/s)만, 위치추정 그대로
  창 끊기      튐 넣은 오돔, sensor_check.py 감지 + 튐에서 위치추정 창 끊기(loc_topk_series --cut-events) — 시험했으나 더 나빠져 안 씀
  고친 뒤      튐 넣은 오돔, sensor_check.py 감지(0.1 s 오돔 + 자이로 맞춤) + 튄 걸음을 예측 이동으로 바꿈(sensor_check --repair-to)
지표(튐마다 그 튐부터 다음 튐 전까지): 튐 감지, 센서 오류 때문에 새로 '헷갈림'이 되어 카메라 멈춤이 허용된 초(착각), 1위 3 m 안 비율.
  tables/E4_fault_fix.md, csv/E4_fault_fix.csv, visual_ko/V7_오돔튐_고친뒤.png
    python3 scripts/e4_fault_eval2.py
"""
import csv, json, subprocess, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from e4_fault_eval import top1_err  # noqa: E402

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트"); OUT = R / "59_E4_호출정책/paper"; FI = OUT / "fault_inject"
INJ = json.load(open("/data/kci_work/e4_fault_inject_20261008/4F/INJECTION.json"))["plan"]
SC = OUT / "sensor_check/INJ_4F_repair.json"


def main():
    for d in ("4F_cut", "4F_repaired"):
        g = FI / d / "모호함_v3.json"
        if not g.exists():
            subprocess.run([sys.executable, "scripts/make_gate_json.py", "--loc", str(FI / d / "loc"), "--sensor-events", str(SC), "--out", str(g)], check=True)
    C = {"원래": (R / "35_위치추정_축척추정/4F/모호함_v2.json", R / "35_위치추정_축척추정/4F/loc"),
         "고치기 전": (FI / "4F/모호함_v2.json", FI / "4F/loc"), "창 끊기": (FI / "4F_cut/모호함_v3.json", FI / "4F_cut/loc"),
         "고친 뒤": (FI / "4F_repaired/모호함_v3.json", FI / "4F_repaired/loc")}
    G = {k: {round(x["t"], 3): x for x in json.load(open(v[0]))} for k, v in C.items()}
    E = {k: top1_err(v[1]) for k, v in C.items()}
    det_new = [e["rel_sec"] for e in json.load(open(SC))["events"]]
    rows = []
    for k, j in enumerate(INJ):
        t0 = float(j["rel_sec"]); t1 = float(INJ[k + 1]["rel_sec"]) if k + 1 < len(INJ) else 400.0
        ts = [t for t in G["원래"] if t0 <= t < t1 and all(t in G[c] for c in G)]
        r = dict(rel_sec=t0, jump_m=j["jump_m"], jump_deg=j["jump_deg"], window_s=len(ts),
                 detected_before=int(any(G["고치기 전"].get(round(t0 + d, 3), {}).get("sensor_bad") for d in (0, 1, 2))),
                 detected_after=int(any(t0 - 0.2 <= x <= t0 + 1.0 for x in det_new)))
        for c in ("고치기 전", "창 끊기", "고친 뒤"):
            # 착각 = 원래는 헷갈림이 아니었는데, 센서 이상으로 안 막히고 헷갈림이 되어 멈춤이 허용된 초
            r[f"false_amb_{c}"] = sum(bool(G[c][t]["amb"]) and not G[c][t].get("sensor_bad") and not G["원래"][t]["amb"] for t in ts)
        for c in G:
            v = [E[c][t] <= 3 for t in ts if t in E[c]]; r[f"top1_3m_{c}"] = round(float(np.mean(v)), 3) if v else ""
        rows.append(r)
    with open(OUT / "csv/E4_fault_fix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["4층 오돔 사본 튐 8개. 구간 = 그 튐부터 다음 튐 전까지. 착각 = 원래는 헷갈림이 아니던 초가 센서 오류 때문에 헷갈림이 되어 카메라 멈춤이 허용된 초. PILOT.", "",
         "| 튐 | 감지 (전 → 후) | 착각 초 (전 → 후) | 1위 3 m 안: 원래 / 전 / 후 | 참고: 창 끊기 착각·3 m 안 |", "|---|---|---|---|---|"]
    for r in rows:
        nm = " + ".join(x for x in ((f"{r['jump_m']:g} m" if r["jump_m"] else ""), (f"{r['jump_deg']:g}°" if r["jump_deg"] else "")) if x)
        L.append(f"| {r['rel_sec']:.0f}s {nm} | {'O' if r['detected_before'] else 'X'} → {'O' if r['detected_after'] else 'X'} | "
                 f"{r['false_amb_고치기 전']} → {r['false_amb_고친 뒤']} | {100 * r['top1_3m_원래']:.0f}% / {100 * r['top1_3m_고치기 전']:.0f}% / {100 * r['top1_3m_고친 뒤']:.0f}% | "
                 f"{r['false_amb_창 끊기']}초 · {100 * r['top1_3m_창 끊기']:.0f}% |")
    S = sum(r["window_s"] for r in rows)
    L += ["", f"합계: 감지 {sum(r['detected_before'] for r in rows)}/8 → {sum(r['detected_after'] for r in rows)}/8, "
              f"착각 {sum(r['false_amb_고치기 전'] for r in rows)}초 → {sum(r['false_amb_고친 뒤'] for r in rows)}초 (구간 {S}초), "
              f"1위 3 m 안 원래 {100 * np.mean([E['원래'][t] <= 3 for t in E['원래'] if t >= 60]):.0f}% / 전 {100 * np.mean([E['고치기 전'][t] <= 3 for t in E['고치기 전'] if t >= 60]):.0f}% / "
              f"후 {100 * np.mean([E['고친 뒤'][t] <= 3 for t in E['고친 뒤'] if t >= 60]):.0f}% (60초 이후). "
              f"창 끊기: 착각 {sum(r['false_amb_창 끊기'] for r in rows)}초, 3 m 안 {100 * np.mean([E['창 끊기'][t] <= 3 for t in E['창 끊기'] if t >= 60]):.0f}% (더 나빠져 안 씀)",
          "정상 녹화본 13개 층 오경보: 순간 이동·회전·자이로 불일치 0건(자이로는 9개 층 중 5개 층에서 켜짐, 나머지는 시각 맞춤 상관 < 0.8 이라 자동으로 끔). 잡힌 것은 실제 오돔 끊김(> 1 s)뿐."]
    (OUT / "tables/E4_fault_fix.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    return G, E


if __name__ == "__main__":
    main()
