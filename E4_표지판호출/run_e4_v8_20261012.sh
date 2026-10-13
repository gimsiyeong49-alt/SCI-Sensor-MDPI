#!/bin/bash
# 2026-10-12 E4 층 모름을 지도 8차로 다시: 3초 간격 합치기 후보(노드 v8 재사용) → 애매 기록 → 애매할 때 재생(65곳 반씩 동시) → 갈릴 때 거르기 → 채점 → 상태 판단 → 그림.
# 기존(_merged) 결과는 그대로. 새 결과 이름은 _v8. P_ALL 은 위치추정을 안 써서 다시 안 돌림.
cd /data/kci_work/KCI_L0_P0_UNIFIED
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
W=/data2/e4_unknown_20261011; R=reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/70_층모름E4; PY=/data2/vlm_env/bin/python
echo "3초 후보 시작 $(date +%H:%M)"
KCI_FLOORID_PKG=/data2/floor_id_pkg_v8_20261012/floor_id E3_ADD_NODE=1 E3_NODE_DIR=/data2/e3_node_v8_20261012 \
  E3_SEED=reports/_01_E2_초기위치/_작업자료/E2_dense_T0_nospace_지도v8_정밀검증/E2_top100_candidates.csv \
  nice python3 scripts/e3_seeded.py --times 3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60 --workers 8 --out /data2/e3_merged_3s_v8_20261012 > /data2/e3_merged_3s_v8_20261012.log 2>&1
echo "3초 후보 끝 $(date +%H:%M)"
export E4_VAR=v8
for d in 0.1 0.2 0.4; do [ -d $W/gate_v8_d$d ] || $PY scripts/e4_unknown_gate.py --seg $W --out $W/gate_v8_d$d --delta $d --k 10; done
echo "애매 기록 끝 $(date +%H:%M)"
POL=P_AMBIGUITY GATE_DIR=$W/gate_v8_d0.2 TASKS=1F_T1,1F_T3,1F_T5,2F_T2,2F_T4,3F_T1,3F_T3,3F_T5,4F_T2,4F_T4,5F_T1,5F_T3,5F_T5,6F_T2,6F_T4,7F_T1,7F_T3,7F_T5,8F_T2,8F_T4,9F_T1,9F_T3,9F_T5,G1F_T2,G1F_T4,G2F_T1,G2F_T3,G2F_T5,S1F_T2,S1F_T4,S6F_T1,S6F_T3,S6F_T5 nice scripts/run_e4_unknown_20261011.sh > $W/본실행_P_AMBIGUITY_v8_a.log 2>&1 &
POL=P_AMBIGUITY GATE_DIR=$W/gate_v8_d0.2 TASKS=1F_T2,1F_T4,2F_T1,2F_T3,2F_T5,3F_T2,3F_T4,4F_T1,4F_T3,4F_T5,5F_T2,5F_T4,6F_T1,6F_T3,6F_T5,7F_T2,7F_T4,8F_T1,8F_T3,8F_T5,9F_T2,9F_T4,G1F_T1,G1F_T3,G1F_T5,G2F_T2,G2F_T4,S1F_T1,S1F_T3,S1F_T5,S6F_T2,S6F_T4 nice scripts/run_e4_unknown_20261011.sh > $W/본실행_P_AMBIGUITY_v8_b.log 2>&1 &
wait
echo "재생 끝 $(date +%H:%M)"
$PY scripts/e4_unknown_informative.py --gate-dir $W/gate_v8_d0.2 --runs $R/P_AMBIGUITY_gate_v8_d0.2 --out $R/P_INFORMATIVE_v8_d0.2
$PY scripts/e4_unknown_score.py
$PY scripts/e4_state_eval.py > /dev/null
$PY scripts/e4_state_figures.py
$PY scripts/e4_unknown_visual_ko.py
echo "전부 끝 $(date +%H:%M)"
