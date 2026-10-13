#!/bin/bash
# 2026-10-12 지도 8차 E3 노드를 GPU(정보관 1~5층)·CPU 작업자 6개(정보관 6~9층·소융대)로 나눠 동시에, 끝나면 채점·합치기·채점. 공학관 노드는 재사용(복사해 둠).
cd /data/kci_work/KCI_L0_P0_UNIFIED
export KCI_FLOORID_PKG=/data2/floor_id_pkg_v8_20261012/floor_id PYTHONNOUSERSITE=1
export E3_NODE_OUT=/data2/e3_node_v8_20261012 E3_NODE_RES=reports/_03_E3_걸으며좁히기/_작업자료/노드_지도v8
echo "E3 노드(나눠서) 시작 $(date +%H:%M)"
OMP_NUM_THREADS=2 nice /data2/vlm_env/bin/python scripts/e3_node.py --times 15,30,45,60 --workers 4 --tasks 1F_T1,1F_T2,1F_T3,1F_T4,1F_T5,2F_T1,2F_T2,2F_T3,2F_T4,2F_T5,3F_T1,3F_T2,3F_T3,3F_T4,3F_T5,4F_T1,4F_T2,4F_T3,4F_T4,4F_T5,5F_T1,5F_T2,5F_T3,5F_T4,5F_T5 > /data2/e3_node_v8_gpu.log 2>&1 &
OMP_NUM_THREADS=1 nice python3 scripts/e3_node.py --times 15,30,45,60 --workers 6 --tasks 6F_T1,6F_T2,6F_T3,6F_T4,6F_T5,7F_T1,7F_T2,7F_T3,7F_T4,7F_T5,8F_T1,8F_T2,8F_T3,8F_T4,8F_T5,9F_T1,9F_T2,9F_T3,9F_T4,9F_T5,S1F_T1,S1F_T2,S1F_T3,S1F_T4,S1F_T5,S6F_T1,S6F_T2,S6F_T3,S6F_T4,S6F_T5 > /data2/e3_node_v8_cpu.log 2>&1 &
wait
python3 scripts/e3_node.py --score >> /data2/e3_node_v8_20261012.log 2>&1
echo "E3 노드 끝 $(date +%H:%M)"
export OMP_NUM_THREADS=1 E3_ADD_NODE=1 E3_NODE_DIR=/data2/e3_node_v8_20261012 E3_SEED=reports/_01_E2_초기위치/_작업자료/E2_dense_T0_nospace_지도v8_정밀검증/E2_top100_candidates.csv
export E3_OUT=/data2/e3_merged_v8_20261012 E3_RES=reports/_03_E3_걸으며좁히기/_작업자료/연결판_재탐색_지도v8
nice python3 scripts/e3_seeded.py --workers 8 > /data2/e3_merged_v8_20261012.log 2>&1
python3 scripts/e3_seeded.py --score >> /data2/e3_merged_v8_20261012.log 2>&1
echo "전부 끝 $(date +%H:%M)"
