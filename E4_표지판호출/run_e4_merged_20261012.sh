#!/bin/bash
# E4 층 모름을 E3 합치기 후보(E2 후보 + 노드 재탐색, 3초)로 다시(2026-10-12). 기존(_merged 없는) 결과는 그대로 둔다.
# P_ALL 은 애매 기록을 쓰지 않아 후보와 무관하므로 기존 70_층모름E4/P_ALL 을 그대로 쓴다.
cd /data/kci_work/KCI_L0_P0_UNIFIED
export E4_VAR=merged OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
PY=/data2/vlm_env/bin/python; W=/data2/e4_unknown_20261011; R=reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/70_층모름E4
echo "시작 $(date +%H:%M)"
for d in 0.1 0.2 0.4; do [ -d $W/gate_merged_d$d ] || $PY scripts/e4_unknown_gate.py --seg $W --out $W/gate_merged_d$d --delta $d --k 10; done
POL=P_AMBIGUITY GATE_DIR=$W/gate_merged_d0.2 scripts/run_e4_unknown_20261011.sh > $W/본실행_P_AMBIGUITY_merged_d0.2.log 2>&1
$PY scripts/e4_unknown_informative.py --gate-dir $W/gate_merged_d0.2 --runs $R/P_AMBIGUITY_gate_merged_d0.2 --out $R/P_INFORMATIVE_merged_d0.2
$PY scripts/e4_unknown_score.py
$PY scripts/e4_state_eval.py > /dev/null
$PY scripts/e4_state_figures.py
echo "전부 끝 $(date +%H:%M)"
