#!/usr/bin/env python3
"""E4 정리본(배포판 C) 한글 꾸러미(2026-10-10). 작업 폴더 59_E4_호출정책/paper_C 를 복사해 한글 파일·열 이름으로 바꾼다(원본은 그대로).
결과: reports/_02_E4_표지판호출/{1_시각자료, 2_이해용그림, 3_논문용그림_영문, 4_표, 5_데이터, 참고_*}. README.md 는 따로 쓴다.
    python3 scripts/e4_package_ko.py
"""
import csv, re, shutil
from pathlib import Path

SRC = Path("reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/59_E4_호출정책"); P = SRC / "paper_C"; DST = Path("reports/_02_E4_표지판호출/3_보조검증_층고정_배포판C")   # 2026-10-12 폴더 정리: 배포판 C 보조 검증은 이 안으로
FL = {"G1F": "공학관 1층", "G2F": "공학관 2층", "S1F": "소융대 1층", "S6F": "소융대 6층"}
fl = lambda v: FL.get(v, re.sub(r"^(\d)F$", r"\1층", v))
VIS = {"V1_흐름_4F243": "1_한순간흐름_4층243초", "V2_확실vs헷갈림": "2_확실vs헷갈림", "V3_전층지도": "3_전층지도", "V4_규칙비교_4F": "4_규칙비교_4층",
       "V5_층별효과": "5_층별효과", "V6_오돔튐": "6_오돔튐"}
KO = {"K1_규칙비교": "1_규칙비교", "K2_상태별": "2_상태별", "K3_4층흐름": "3_4층흐름", "K4_틀린종류": "4_틀린종류", "K5_예시사진": "5_예시사진"}
FIG = {"E4_fig_detection_examples": "1_검출예시", "E4_fig_pr_room_sign": "2_PR곡선_호실표지판", "E4_fig_policy": "3_멈춤규칙비교", "E4_fig_uncertainty_classes": "4_매초상태4종",
       "E4_fig_fp_types": "5_틀린멈춤종류", "E4_fig_confusion": "6_착각표", "E4_fig_fault_injection": "7_오돔튐주입", "E4_fig_timeline_4F": "8_시간흐름_4층",
       "E4_fig_timeline_S6F": "9_시간흐름_소융대6층"}
TAB = {"E4_uncertainty_classes": "01_매초상태4종", "E4_rates": "02_멈춤빈도", "E4_confusion": "03_착각표", "E4_fp_types": "04_틀린멈춤종류",
       "E4_recall_P_ALL_P_AMBIGUITY_P_R3B": "05_호실단위_재현율", "E4_recall_why": "06_놓친이유", "E4_restop": "07_재멈춤막기", "E4_prompts": "08_문구비교",
       "E4_fault_injection": "09_오돔튐_주입시험", "E4_fault_fix": "10_오돔튐_고친결과", "E4_sensor_check_normal": "11_센서판단_정상13층", "E4_latency": "12_처리시간"}
CLS = {"GEOMETRIC_CONFIDENT": "확실", "GEOMETRIC_AMBIGUITY": "헷갈림", "GEOMETRIC_INSUFFICIENT": "헷갈림+긴 직선 복도", "SENSOR_UNRELIABLE": "센서 이상"}
POL = {"P_ALL": "보이면 다 멈춤", "P_AMBIGUITY": "A 헷갈릴 때만(10-08)", "P_R3B": "C 배포판(10-09)", "P_INFORMATIVE_0.5bit": "정보 될 때만 0.5 bit"}
TAB_REPL = {  # 표마다 바꿀 영어(문구 표는 모델 입력 문장이라 손대지 않음)
    "01_매초상태4종": list(CLS.items()),
    "02_멈춤빈도": list(POL.items()),
    "03_착각표": [("| room sign | other sign / lettering | not a sign | stair | unjudged | floor not judged |", "| 호실 표지판 | 다른 표지·글씨 | 표지판 아님 | 계단 | 미판정 | 판정 안 한 층 |"),
              ("| room_sign |", "| 호실 표지판으로 멈춤 |"), ("| stair |", "| 계단 관측 |")],
    "12_처리시간": [("candidate proposals + accumulation", "후보 뽑기 + 누적"), ("wall-segment grouping", "벽 구간 묶기"), ("SigLIP scoring (worker", "SigLIP 점수(작업자"),
                ("SigLIP round trip", "SigLIP 왕복"), ("memory + decision", "기억 + 판단"), ("total per scene", "장면당 합계")],
}
COL = {"trial_id": "층", "run_id": "실행", "event_id": "멈춤번호", "relative_sec": "시각_초", "bbox_x": "상자_x", "bbox_y": "상자_y", "bbox_w": "상자_폭", "bbox_h": "상자_높이",
       "predicted_class": "예측종류", "siglip_score": "SigLIP점수", "depth_m": "깊이_m", "uncertainty_class": "그순간_상태", "info_bits": "정보량_bit", "trigger_main": "배포판에서_멈춤",
       "human_label": "사람판정", "source": "출처", "class_label": "정답종류", "object_id": "대상번호", "rel_sec": "튐시각_초", "jump_m": "튐_m", "jump_deg": "튐_도",
       "window_s": "구간_초", "detected_before": "고치기전_감지", "detected_after": "고친뒤_감지", "sensor_flag_at_jump": "튐순간_센서이상감지", "sensor_bad_seconds": "센서이상_초",
       "amb_seconds_orig": "헷갈림초_원래", "amb_seconds_injected": "헷갈림초_주입", "new_amb_seconds": "새로헷갈린초", "top1_3m_orig": "1위3m안_원래",
       "top1_3m_injected": "1위3m안_주입", "no": "번호", "floor": "층", "fp_type": "틀린종류_코드", "fp_type_ko": "틀린종류", "fp_type_en": "틀린종류_영문",
       "scenes_computed": "처리장면수", "wall_sec": "걸린시간_초", "prompt": "문구", "kind": "구분", "n": "사진수", "n_pos": "표지판수", "pr_auc": "PR_AUC", "roc_auc": "ROC_AUC",
       "condition": "규칙", "sign_stops": "표지판멈춤", "minutes": "분", "distance_m": "이동거리_m", "stops_per_min": "1분당_멈춤", "stops_per_10m": "10m당_멈춤",
       "judged_floor_minutes": "판정층_분", "false_stops_per_min_judged": "판정층_틀린멈춤_분당", "true_stops_per_min_judged": "판정층_맞은멈춤_분당",
       "repeat_stop_rate": "재멈춤비율", "split": "구분", "judged": "사람판정층", "rooms_passed": "지나간호실", "rooms_passed_amb": "헷갈릴때_지나간호실",
       "rooms_seen_amb_main": "헷갈릴때_본호실_C", "rooms_seen_P_ALL": "본호실_다멈춤", "rooms_seen_P_AMBIGUITY": "본호실_A", "rooms_seen_P_R3B": "본호실_C",
       "recall_P_ALL": "재현율_다멈춤", "recall_P_AMBIGUITY": "재현율_A", "recall_P_R3B": "재현율_C", "room_id": "호실id", "room_number": "호실번호",
       "closest_pass_sec": "가장가까이_지난시각_초", "passed_while_ambiguous": "헷갈릴때_지남", "seen_P_ALL": "봄_다멈춤", "seen_P_AMBIGUITY": "봄_A", "seen_P_R3B": "봄_C",
       "margin": "점수차", "spread_m": "후보흩어짐_m", "n_near": "근처후보수", "normal_diversity": "벽방향_다양성", "sensor_bad": "센서이상", "allow_P_ALL": "멈춤허용_다멈춤",
       "allow_P_AMBIGUITY": "멈춤허용_헷갈릴때", "top1_within_3m_gt": "1위_3m안", "sample_id": "표본id", "cells_all": "칸_전체", "cells_used": "칸_사용"}
VAL = {"uncertainty_class": CLS, "predicted_class": {"room_sign": "호실표지판", "stair": "계단"}, "human_label": {"TP": "표지판맞음", "FP": "표지판아님", "UNJUDGED": "미판정"},
       "condition": POL}
CSV = {"E4_trigger_decisions": "01_매초_상태와_멈춤허용", "E4_detections": "02_멈춤마다_상자점수판정", "E4_event_annotations": "03_4층_프레임정답과_사람판정",
       "E4_rates": "04_멈춤빈도", "E4_fp_types": "05_틀린멈춤_종류", "E4_recall_P_ALL_P_AMBIGUITY_P_R3B": "06_층별_호실재현율", "E4_recall_rooms_P_ALL_P_AMBIGUITY_P_R3B": "07_호실마다_봤나",
       "E4_prompts": "08_문구비교", "E4_fault_injection": "09_오돔튐_주입시험", "E4_fault_fix": "10_오돔튐_고친결과", "E4_latency": "11_처리시간"}


def col_ko(c):
    for pre, ko in (("false_amb_", "착각초_"), ("top1_3m_", "1위3m안_")):
        if c.startswith(pre) and c not in COL:
            return ko + c[len(pre):]
    return COL.get(c, c)


def val_ko(c, v):
    if c in ("trial_id", "floor"):
        return fl(v.replace("PILOT_", ""))
    if c == "run_id":
        return POL.get(v, v)
    if c == "event_id":
        for k, ko in POL.items():
            v = v.replace(f"_{k}_", f"_{ko}_")
        return v
    return VAL.get(c, {}).get(v, v)


def main():
    if DST.exists() and any((DST / d).exists() for d in ("1_시각자료", "4_표")):
        print("이미 있음 — 덮어쓰기 전 확인:", DST)
    for sub, src, names, exts in (("1_시각자료", P / "visual_ko", VIS, (".png",)), ("2_이해용그림", P / "figures_ko", KO, (".png",)),
                                  ("3_논문용그림_영문", P / "figures", FIG, (".png", ".pdf"))):
        (DST / sub).mkdir(parents=True, exist_ok=True)
        for a, b in names.items():
            for e in exts:
                if (src / f"{a}{e}").exists():
                    shutil.copy2(src / f"{a}{e}", DST / sub / f"{b}{e}")
    (DST / "4_표").mkdir(exist_ok=True)
    for a, b in TAB.items():
        s = (P / "tables" / f"{a}.md").read_text()
        for x, y in TAB_REPL.get(b, []):
            s = s.replace(x, y)
        (DST / "4_표" / f"{b}.md").write_text(s)
    (DST / "5_데이터").mkdir(exist_ok=True)
    for a, b in CSV.items():
        rows = list(csv.reader(open(P / "csv" / f"{a}.csv")))
        head = rows[0]
        with open(DST / "5_데이터" / f"{b}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow([col_ko(c) for c in head])
            for r in rows[1:]:
                w.writerow([val_ko(c, v) for c, v in zip(head, r)])
    (DST / "5_데이터/층별_벽방향칸").mkdir(exist_ok=True)
    for f in sorted((P / "csv").glob("geom_cells_*.csv")):
        F = f.stem.split("_")[-1]; rows = list(csv.reader(open(f)))
        with open(DST / "5_데이터/층별_벽방향칸" / f"{fl(F).replace(' ', '')}.csv", "w", newline="", encoding="utf-8-sig") as g:
            w = csv.writer(g); w.writerow([col_ko(c) for c in rows[0]]); w.writerows(rows[1:])
    shutil.copy2(SRC / "버전비교_20261008.md", DST / "참고_버전비교_20261008.md")
    shutil.copy2(P / "E4_versions.yaml", DST / "참고_모델코드버전.yaml")
    left = [str(p) for p in DST.rglob("*") if p.is_file() and re.search(r"[A-Za-z]{4,}", p.stem) and "SigLIP" not in p.stem and "PR" not in p.stem]
    print("영어 이름 남은 파일:", left)


if __name__ == "__main__":
    main()
