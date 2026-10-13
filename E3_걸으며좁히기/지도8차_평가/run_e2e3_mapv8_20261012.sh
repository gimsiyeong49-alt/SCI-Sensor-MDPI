#!/bin/bash
# 2026-10-12 지도 8차(로봇과 같은 10-05 최신 지도 + 계단 앞·9층 복도 셔터를 문 마스크로)로 E2 정밀 검증 → E3 노드 → E3 합치기 다시. 기존 결과는 그대로, 새 폴더에.
cd /data/kci_work/KCI_L0_P0_UNIFIED
export KCI_FLOORID_PKG=/data2/floor_id_pkg_v8_20261012/floor_id OMP_NUM_THREADS=2 PYTHONNOUSERSITE=1
N=T0_nospace_후보_지도v8; L=reports/_01_E2_초기위치/_작업자료/미세정합시험/실행_지도v8; mkdir -p $L
echo "E2 정밀 검증 시작 $(date +%H:%M)"
T=$(python3 -c "import csv;print(','.join(t['task_id'] for t in csv.DictReader(open('reports/_01_E2_초기위치/_작업자료/tasks/E2_tasks.csv'))))")
IFS=, read -ra A <<< "$T"; n=${#A[@]}
for w in 0 1 2; do S=""; for ((i=w;i<n;i+=3)); do S="$S,${A[$i]}"; done
  nice python3 scripts/e2_refine_test.py --src T0_nospace --top 1000 --save-top --out-name $N --tasks ${S#,} > $L/w$w.log 2>&1 & done; wait
python3 scripts/e2_method_candidates.py --name $N --suffix _nospace_지도v8
echo "E2 끝 $(date +%H:%M)"
export E3_NODE_OUT=/data2/e3_node_v8_20261012 E3_NODE_RES=reports/_03_E3_걸으며좁히기/_작업자료/노드_지도v8
nice /data2/vlm_env/bin/python scripts/e3_node.py --times 15,30,45,60 --workers 4 > /data2/e3_node_v8_20261012.log 2>&1
/data2/vlm_env/bin/python scripts/e3_node.py --score >> /data2/e3_node_v8_20261012.log 2>&1
echo "E3 노드 끝 $(date +%H:%M)"
export E3_ADD_NODE=1 E3_NODE_DIR=/data2/e3_node_v8_20261012 E3_SEED=reports/_01_E2_초기위치/_작업자료/E2_dense_T0_nospace_지도v8_정밀검증/E2_top100_candidates.csv
export E3_OUT=/data2/e3_merged_v8_20261012 E3_RES=reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색_지도v8
nice python3 scripts/e3_seeded.py --workers 8 > /data2/e3_merged_v8_20261012.log 2>&1
python3 scripts/e3_seeded.py --score >> /data2/e3_merged_v8_20261012.log 2>&1
echo "전부 끝 $(date +%H:%M)"
