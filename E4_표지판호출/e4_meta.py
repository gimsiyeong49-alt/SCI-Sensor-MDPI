#!/usr/bin/env python3
"""E4 처리 시간 표와 판(버전) 기록(2026-10-07, 실험 지침서 14.6·18절, PILOT). 59_E4_호출정책/paper/ 아래에 만든다.
  tables/E4_latency.md, csv/E4_latency.csv  층별 장면 처리 시간(run 의 timing.json 중앙·p95) + Qwen 계단 질문 시간(facility_asks.json 원값)
  E4_versions.yaml                           모델 파일·코드·설정·입력의 sha256, 문구, 기준값, 기기
주의: timing.json 은 층별 중앙·p95 만 남아 있어 13개 층을 합친 p95 는 못 낸다(층별 값의 범위를 적는다). 개발 노트북 기준이다.
    python3 scripts/e4_meta.py
"""
import csv, hashlib, json, os, platform, subprocess
from pathlib import Path
import numpy as np

R = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트")
P = R / "59_E4_호출정책"; OUT = P / os.environ.get("E4_PAPER", "paper")
MAIN = os.environ.get("E4_MAIN", "P_AMBIGUITY")      # 본 정책(2026-10-09: 배포판 C = P_R3B)
MAIN_NAME = os.environ.get("E4_MAIN_NAME", MAIN)

FL = "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split()
D = Path("/data/kci_work/maps_perfloor_20261001")


def sha(p, cache={}):
    p = Path(p)
    if p not in cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[p] = h.hexdigest()
    return cache[p]


def model_hashes(d):
    d = Path(d)
    return {f.name: sha(f) for f in sorted(d.iterdir()) if f.is_file() and f.suffix in (".safetensors", ".json", ".bin", ".model", ".txt")}


def main():
    rows = []; vlm = []
    for F in FL:
        run = P / MAIN / F / "run"; T = json.load(open(run / "timing.json"))
        r = dict(trial_id=f"PILOT_{F}", scenes_computed=T["scenes"], wall_sec=T["wall_sec"])
        for k, v in T["timing"].items():
            r[f"{k}_med_ms"] = v["med_ms"]; r[f"{k}_p95_ms"] = v["p95_ms"]
        rows.append(r)
        vlm += [x["vlm_sec"] for x in json.load(open(run / "facility_asks.json")) if x.get("vlm_sec")]
    with open(OUT / "csv" / "E4_latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stages = list(json.load(open(P / MAIN / "4F" / "run" / "timing.json"))["timing"])
    name = {"후보+누적": "candidate proposals + accumulation", "벽 구간": "wall-segment grouping", "SigLIP(작업자)": "SigLIP scoring (worker)",
            "SigLIP 왕복": "SigLIP round trip", "기억+판단": "memory + decision", "합계": "total per scene"}
    L = ["개발 노트북(RTX 5070 Laptop, Core Ultra 9 275HX), 녹화본 재생, " + MAIN_NAME + ". 층별 중앙·p95 의 13개 층 범위.", "",
         "| 단계 | 중앙(ms), 층별 범위 | p95(ms), 층별 범위 |", "|---|---|---|"]
    for s in stages:
        md = [r[f"{s}_med_ms"] for r in rows]; p9 = [r[f"{s}_p95_ms"] for r in rows]
        L.append(f"| {name.get(s, s)} | {min(md):.0f}–{max(md):.0f} (층 중앙의 중앙 {np.median(md):.0f}) | {min(p9):.0f}–{max(p9):.0f} |")
    v = np.array(vlm)
    L += ["", f"Qwen3-VL-2B 계단 질문(13개 층 {len(v)}번, 원값): 중앙 {np.median(v):.2f} s, p95 {np.percentile(v, 95):.2f} s. 걷는 중 비동기라 멈춤을 늦추지 않는다.",
          "멈춤에서 조준 명령까지: 설계상 1초(멈춘 뒤 1초 기다렸다가 지금 자세로 다시 찾아 겨눈다). 로봇(Orin) 시간은 아직 재지 않았다."]
    (OUT / "tables" / "E4_latency.md").write_text("\n".join(L) + "\n"); print("\n".join(L))

    scripts = ["live_event_runner.py", "make_gate_json.py", "make_search_config.py", "region_siglip_rerank.py", "live_vlm_worker.py", "obs_feedback.py",
               "score_loc_pseudo_gt.py", "e4_informative.py", "e4_eval.py", "e4_detection.py", "e4_geom_cells.py", "e4_paper.py", "e4_figures.py", "e4_meta.py"]
    pyv = subprocess.run(["/data2/vlm_env/bin/python", "-c", "import torch,transformers;print(torch.__version__, transformers.__version__)"],
                         capture_output=True, text=True).stdout.split()
    Y = {"experiment_id": "E4", "split": "PILOT (개발 녹화본 13개 층, 최종 평가 아님)",
         "device": {"gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "cpu": "Intel Core Ultra 9 275HX", "os": platform.platform(),
                    "python_main": platform.python_version(), "vlm_env_torch": pyv[0] if pyv else "", "vlm_env_transformers": pyv[1] if len(pyv) > 1 else ""},
         "models": {"siglip": {"path": "/data2/kci_models/siglip-base-patch16-224", "files": model_hashes("/data2/kci_models/siglip-base-patch16-224")},
                    "qwen_vl": {"path": "/data3/kci_models/Qwen3-VL-2B-Instruct", "files": model_hashes("/data3/kci_models/Qwen3-VL-2B-Instruct")}},
         "config": {"sign_prompts": ["a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall"],
                    "sign_score": "max over prompts", "sim_floor": 0.078, "relative_threshold": "top 10% of segments seen so far",
                    "floor_height_band_m": [0.6, 2.8], "max_width_m": 0.85, "max_depth_m": 7.0, "gate_mode": "every3", "stop_rule": "amb_sensor_ok",
                    "ambiguity": "margin(top-1 vs best candidate >5 m away) <= 1.5", "sensor_bad": "odom speed > 2 m/s or yaw rate > 120 deg/s",
                    "insufficient": "ambiguous and median wall-normal diversity (last 3 s) < 0.1 (pilot threshold)",
                    "informative": "entropy of near-candidate answers, near = score >= best - 1.5", "stairs": "Qwen3-VL-2B, observe only (no stop)",
                    "oracle_feedback": "obs_feedback --oracle-numbers --tol-per-m 0.08 --max-contra 1 (upper bound, uses pseudo-GT room numbers)"},
         "code_sha256": {s: sha(Path("scripts") / s) for s in scripts},
         "inputs_sha256": {F: {"gate": sha(R / "35_위치추정_축척추정" / F / "모호함_v2.json"),
                               "loc_topk": sha(R / "35_위치추정_축척추정" / F / "loc" / "topk_series.csv"),
                               "semantic_map": sha(D / F / f"{F}_semantic_map.json"),
                               "pseudo_gt": sha(R / "36_채점용정답" / F / "warp_clean" / "gt_warp.csv")} for F in FL},
         "judgments_sha256": {"list": sha(R / "25_판정" / "판정목록.json"), "result": sha(R / "25_판정" / "판정결과.json")},
         "git": "저장소 커밋 없음(작업 트리 미추적) — 위 code_sha256 로 판을 고정한다"}
    import yaml
    (OUT / "E4_versions.yaml").write_text(yaml.safe_dump(Y, allow_unicode=True, sort_keys=False, width=200))
    print("판 기록:", OUT / "E4_versions.yaml")


if __name__ == "__main__":
    main()
