#!/usr/bin/env python3
"""층별 '찾을 대상' 설정 파일을 의미지도에서 자동으로 만든다(2026-10-05).

모든 건물에 같은 변환표 하나(COMMON)를 쓰고, 층마다 달라지는 것은 '의미지도에 어떤 종류가 있나'뿐이다.
사람이 결과를 고치지 않는다. 지도가 실제와 다르면(엘베·호실 누락 등) 지도를 고친다.
엘베는 모든 지도에서 쓰지 않는다(사용자 결정 2026-10-05, 걸으면서 판정 31%).

    python3 scripts/make_search_config.py <의미지도.json> <층 이름> --out configs/search_targets/<층>.json
"""
import argparse
import json
from pathlib import Path

# 표준 목록(2026-10-05 사용자 결정). 이것 외에는 찾지 않는다. 의미지도에 그 종류가 있는 층에서만 켜진다.
# small_sign = 벽의 작은 후보를 SigLIP 문구로 점수(문구 여러 개면 가장 높은 값), scene_object = 화면 전체를 VLM 에 물음.
# 호실 표지판 문구: 판정 멈춤 179개로 비교(AUC) "a small room name plate" 0.67, "...near a door" 0.76, 둘 중 최대 0.74.
#   문 옆이 아닌 건물에 대비해 둘 중 최대를 쓴다. 화장실 표지는 따로 두지 않는다(작은 표지판으로 함께 잡히고 판독에서 가림).
# 2026-10-06: 복도 돌출 표지판 문구를 더함. 판정 230개 AUC 0.72(층별 7F 0.68·G1F 0.70 고름, scripts/phrase_auc_search.py),
#   4F·7F·G1F 주행: 7F+G1F 표지판 비율 50 → 63%, 4층 번호판 5개 그대로(52_문구섞기_3층). 호실 표지판 문구는 반드시 둔다(사용자).
# 비상구·소화전은 표준 문구로 넣어 둠(시험 안 함, 지금 의미지도에 칸이 없어 켜지지 않음). 엘베는 목록에 두되 끔.
COMMON = {
    "room":         dict(detector="small_sign", phrases=["a small room name plate", "a small room name plate near a door",
                                                          "a hallway sign perpendicular to the wall"], need_number=True),
    "exit":         dict(detector="small_sign", phrases=["an emergency exit sign"]),
    "fire_hydrant": dict(detector="small_sign", phrases=["a fire hydrant cabinet"]),
    "stair":        dict(detector="scene_object", name="계단", answer_words=["stair", "staircase"], box_phrase="staircase", need_map=False),
    "escalator":    dict(detector="scene_object", name="에스컬레이터", answer_words=["escalator"], box_phrase="escalator", need_map=False),
    "elevator":     dict(detector="scene_object", name="엘리베이터", answer_words=["elevator"], box_phrase="elevator door", need_map=True,
                         enabled=False),   # 사용자 결정: 모든 지도에서 사용 안 함(걸으면서 판정 31%)
}


def build(sem, floor):
    kinds = {}
    for s in sem["spaces"]:
        t = s.get("type")
        if t not in COMMON or not COMMON[t].get("enabled", True):
            continue
        if COMMON[t].get("need_number") and not str(s.get("room_number") or "").strip():
            continue
        kinds[t] = kinds.get(t, 0) + 1
    cfg = dict(floor=floor, source="의미지도 자동(make_search_config.py 공통 변환표)", map_kinds=kinds,
               small_sign=[], scene_object=[])
    for t in kinds:
        c = COMMON[t]
        if c["detector"] == "small_sign":
            cfg["small_sign"].append(dict(map_kind=t, phrases=c["phrases"]))
        else:
            cfg["scene_object"].append(dict(map_kind=t, name=c["name"], answer_words=c["answer_words"],
                                            box_phrase=c["box_phrase"], need_map=c["need_map"]))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sem"); ap.add_argument("floor"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = build(json.load(open(a.sem)), a.floor)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(a.floor, cfg["map_kinds"], "| 작은 표지:", [p for x in cfg["small_sign"] for p in x["phrases"]],
          "| 장면:", [x["name"] for x in cfg["scene_object"]])


if __name__ == "__main__":
    main()
