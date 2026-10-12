#!/usr/bin/env python3
"""E3 본 결과 폴더(1_노드그대로, 2_연결판_E2후보이어받기)의 시점별 README: 건물 → 층별 표, 그림 링크(2026-10-11)."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e2_visual_ko as EV
import e3_main_figures as M

M._setup(); P = M.PKG; W = M.W
N = {(r["trial_id"], int(r["T"])): r for r in csv.DictReader(open(W / "노드" / "노드_채점.csv"))}
S = {(r["trial_id"], int(r["T"])): r for r in csv.DictReader(open(W / "연결판" / "연결판_채점.csv"))}
MG = {(r["trial_id"], int(r["T"])): r for r in csv.DictReader(open(W / "연결판_재탐색" / "연결판_채점.csv"))}   # 2026-10-12 합치기
E2 = {r["trial_id"]: r["true_rank"] for r in csv.DictReader(open(EV.G / "E2_dense_T0_nospace_정밀검증" / "E2_top100_per_trial.csv"))}
mark = lambda b: "✓" if str(b) == "1" else "✗"
for mode, D, title in (("node", N, "노드 그대로(로봇 실험 모드와 같은 계산)"), ("seeded", S, "연결판(E2 정지 Top-100 이어받기)"),
                       ("merged", MG, "합치기·최종(E2 정지 Top-100 + 15초마다 노드가 새로 찾은 후보를 함께 이어받기)")):
    for T in M.TS:
        tasks = [k for k in EV.TASK if (k, T) in D]; rows = [D[(k, T)] for k in tasks]
        if mode == "node":
            head = f"층 1위 {sum(r['floor_rank'] == '1' for r in rows)} · 1위 위치 맞음 {sum(int(r['top1_ok']) for r in rows)} · 층을 안다면 위치 맞음 {sum(int(r['truefloor_ok']) for r in rows)}"
            cols = "| 출발점 | 종류 | 판정 층 | 정답 층 순위 | 1위 위치 | 층을 안다면 위치(오차) | 그림 |\n|---|---|---|---|---|---|---|"
        else:
            head = f"1위 위치 맞음 {sum(int(r['top1_ok']) for r in rows)} · Top-10 {sum(int(r['top10_ok']) for r in rows)} · Top-100 {sum(int(r['top100_ok']) for r in rows)}"
            cols = "| 출발점 | 종류 | 정답 순위 정지(E2) → 지금 | 남은 후보 | 1위 위치 | 그림 |\n|---|---|---|---|---|---|"
        L = [f"# E3 {T}초 · {title}", "", f"{len(tasks)}곳 중 {head}. 정답 = 정답 층 + 지금 실제 자리 3 m + 방향 30°. 그림은 건물 → 층 폴더에 출발점마다 한 장.", ""]
        for bn in dict.fromkeys(EV.BNAME[EV.TASK[k]["building"]] for k in tasks):
            kb = [k for k in tasks if EV.BNAME[EV.TASK[k]["building"]] == bn]
            for fl in dict.fromkeys(M.subpath(k)[0].name for k in kb):
                kf = [k for k in kb if M.subpath(k)[0].name == fl]
                L += [f"## {bn} {fl} — [폴더]({bn}/{fl}/)", "", cols]
                for k in kf:
                    r = D[(k, T)]; sd, name = M.subpath(k); typ = EV.TYPE[EV.TASK[k]["start_type"]]; link = f"[보기]({sd.as_posix()}/{name})"
                    if mode == "node":
                        L.append(f"| 출발점{k.split('_T')[1]} | {typ} | {r['verdict']}층 | {r['floor_rank']}위 | {mark(r['top1_ok'])} | {mark(r['truefloor_ok'])} ({r['truefloor_pos_err_m']} m) | {link} |")
                    else:
                        e2 = E2[k] if str(E2[k]).isdigit() and int(E2[k]) <= 100 else "밖"
                        L.append(f"| 출발점{k.split('_T')[1]} | {typ} | {e2} → {r['true_rank'] or '밖'} | {r['n_left']} | {mark(r['top1_ok'])} | {link} |")
                L.append("")
        od = P / M.FOLDER[mode] / f"{T}초"; od.mkdir(parents=True, exist_ok=True); (od / "README.md").write_text("\n".join(L) + "\n")
print("README 끝")
