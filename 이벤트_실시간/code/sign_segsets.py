#!/usr/bin/env python3
"""표지 후보 실험 구간 설정. 층마다 후보 파일, 정답, 프레임 폴더, 깊이 캐시, 오돔을 묶는다.

프레임 경로는 프레임 폴더 기준 상대경로(proposals.jsonl의 "frame")이고, 스크립트는 root / frame 으로 읽는다.
시간 t 는 각 구간 프레임 index.csv 의 상대 초(4층 header_relative_sec, 3층 bag_relative_sec)다.
t0 는 그 상대 초 0에 해당하는 절대시각(오돔 t_sec 과 맞추는 용도).
오돔 규약은 두 층 모두 월드 = R(yaw + 228°)·p + t (이동 방향과 yaw 차이로 정답 없이 확인).
3층 odom_link 는 좌우 미러가 안 풀려 있지만 이 두 구간은 직진이라 같은 자리 모으기에는 영향이 없다.
"""

from pathlib import Path

R = Path("reports/_00_SIGN_PIPELINE_20260920")

SETS = {
    "4F": [
        dict(name="4F_148_178", proposals=R / "6_기하점수/proposals.jsonl",
             gt=R / "1_정답/gt_marks_148_178_1s_objid.jsonl",
             root=Path("/data2/kci_datasets/4F_RGB_FULL_V0_1"),
             depth="/data3/kci_datasets/depth_cache_sign_eval/4F_148.npz",
             odom="/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv", t0=1783340324.059948032),
        dict(name="4F_228_258", proposals=R / "17_평가구간_228_258/prior/proposals.jsonl",
             gt=R / "17_평가구간_228_258/gt_marks_228_258_1s_objid.jsonl",
             root=Path("/data2/kci_datasets/4F_RGB_FULL_V0_1"),
             depth="/data3/kci_datasets/depth_cache_sign_eval/4F_228.npz",
             odom="/data2/a131_rebuild/a131_4F_clean_v7tr2s/odom_link.csv", t0=1783340324.059948032),
    ],
    "3F": [
        dict(name="3F_450_480", proposals=R / "19_3층평가_450_480/prior/proposals.jsonl",
             gt=R / "19_3층평가_450_480/gt_marks_3F_450_480_1s_objid.jsonl",
             root=Path("/data3/kci_datasets/3F_RGB_448_481_V0_1"),
             depth="/data3/kci_datasets/depth_cache_sign_eval/3F_450.npz",
             odom="/data2/a131_rebuild/a131_3F_clean_v7tm5/odom_link_lc.csv",
             t0=1788170891.309801930 - 448.016452712),
        dict(name="3F_650_680", proposals=R / "21_3층평가2_650_680/prior/proposals.jsonl",
             gt=R / "21_3층평가2_650_680/gt_marks_3F_650_680_1s_objid.jsonl",
             root=Path("/data3/kci_datasets/3F_RGB_648_681_V0_1"),
             depth="/data3/kci_datasets/depth_cache_sign_eval/3F_650.npz",
             odom="/data2/a131_rebuild/a131_3F_clean_v7tm5/odom_link_lc.csv",
             t0=None),                                  # 아래에서 index.csv 로 채운다
    ],
    # 고정 평가 구간(2026-09-22). 방법 코드를 고정한 뒤 한 번만 채점한다.
    "3F_eval": [
        dict(name="3F_330_360", proposals=R / "31_벽구간이벤트/10_고정평가_3층_330_360/prior/proposals.jsonl",
             gt=R / "31_벽구간이벤트/10_고정평가_3층_330_360/gt_marks_3F_330_360_1s_objid.jsonl",
             root=Path("/data3/kci_datasets/3F_RGB_328_361_V0_1"),
             depth="/data3/kci_datasets/depth_cache_sign_eval/3F_330.npz",
             odom="/data2/a131_rebuild/a131_3F_clean_v7tm5/odom_link_lc.csv", t0=1788170443.293349),
    ],
}


def _fill_t0():
    import csv
    for s in SETS["3F"]:
        if s["t0"] is None:
            r = next(csv.DictReader(open(s["root"] / "index.csv")))
            s["t0"] = int(r["bag_timestamp_ns"]) / 1e9 - float(r["bag_relative_sec"])


_fill_t0()


def get(name):
    return SETS[name]
