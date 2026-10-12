#!/bin/bash
# 층 모름 E4(지침서판, 2026-10-11): 65개 출발점 [t_s, t_s+60] 구간만 이벤트 실행기(배포판 C 기본값)로 돌린다.
# 층 아는 판(64_·69_)과 다른 점: ① 애매 기록 = E3 연결판 후보(건물 전체 층)로 만든 것(scripts/e4_unknown_gate.py)
#   ② --fac-map-topk(정답 층 위치추정으로 시설 예측) 안 씀 → 계단 질문은 번호판 후보가 없을 때 묻는 기본 방식 ③ 출발점마다 기억을 비우고 시작
# POL=P_AMBIGUITY(GATE_DIR 의 <출발점>.json) | P_ALL(애매 기록 없이). TASKS=쉼표 목록(비우면 65곳 전부)
# 구간·장면 목록은 scripts/e4_unknown_segments.py 결과($W/segments.csv, $W/frames/)를 쓴다(시각 off 맞춤).
#   예: POL=P_AMBIGUITY GATE_DIR=/data2/e4_unknown_20261011/gate_d0.2 TASKS=4F_T1,G1F_T2 scripts/run_e4_unknown_20261011.sh
# 결과: 31_벽구간이벤트/70_층모름E4/<POL 이름>/<출발점>/run
cd /data/kci_work/KCI_L0_P0_UNIFIED
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
G=reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트
C=/data3/kci_datasets/depth_cache_sign_eval
W=/data2/e4_unknown_20261011
case "$POL" in
  P_ALL) NAME=${POL_NAME:-P_ALL} ;;
  P_AMBIGUITY) [ -d "$GATE_DIR" ] || { echo "GATE_DIR 없음: $GATE_DIR"; exit 1; }; NAME=${POL_NAME:-P_AMBIGUITY_$(basename $GATE_DIR)} ;;
  *) echo "POL 은 P_ALL 또는 P_AMBIGUITY"; exit 1 ;;
esac
[ -f $W/segments.csv ] || { echo "$W/segments.csv 없음(e4_unknown_segments.py 먼저)"; exit 1; }
OUT=$G/70_층모름E4/$NAME; mkdir -p $OUT
t0(){ python3 -c "import csv;r=next(csv.DictReader(open('/data3/kci_datasets/$1_RGB_FULL_5HZ_V0_1/index.csv')));print(int(r['bag_timestamp_ns'])/1e9-float(r['bag_relative_sec']))"; }
declare -A ODOM=([1F]=/data2/a131_rebuild/a131_1F_clean_v7/odom_link_lc.csv [2F]=/data2/a131_rebuild/a131_2F_clean_v7t/odom_link_icp_winfar06.csv
  [3F]=/data2/a131_rebuild/a131_3F_clean_v7tm5/odom_link_lc_winfar06.csv [5F]=/data2/a131_rebuild/a131_5F_clean_v7t/odom_link_lc_winfar06.csv
  [6F]=/data2/a131_rebuild/a131_6F_clean_v5/odom_link.csv [7F]=/data2/a131_rebuild/a131_7F_clean_v5m2/odom_link_icp.csv
  [8F]=/data2/a131_rebuild/a131_8F_clean_v5/odom_link.csv [9F]=/data2/a131_rebuild/a131_9F_clean_v7/odom_link_icp.csv
  [G1F]=/data2/a131_rebuild/a131_G1_clean_v7m2/odom_link_fuse_pos1s_win06.csv [G2F]=/data2/a131_rebuild/a131_G2_clean_v7/odom_link_fuse_pos1s_win06.csv
  [S1F]=/data2/a131_rebuild/a131_S1_clean_v7t/odom_link_fuse_pos1s_winfar06.csv [S6F]=/data2/a131_rebuild/a131_S6_clean_v7t3/odom_link_fuse_pos1s_winfar06.csv)
echo "$NAME 시작 $(date +%H:%M)"
tail -n +2 $W/segments.csv | while IFS=, read K F B TS OFF OT0 LO HI NSC FS LS; do
  [ -n "$TASKS" ] && [[ ",$TASKS," != *",$K,"* ]] && continue
  TE=$HI
  [ "$POL" = P_AMBIGUITY ] && { [ -f $GATE_DIR/$K.json ] || { echo "$K 애매 기록 없음"; continue; }; }
  if [ $F = 4F ]; then SRC=(); else
    SRC=(--root /data3/kci_datasets/${F}_RGB_FULL_5HZ_V0_1 --depth $C/${F}_full.npz --odom ${ODOM[$F]} --t0 $(t0 $F)); fi
  mkdir -p $OUT/$K
  [ -f $OUT/$K/run/events_live_replay.json ] || python3 -u scripts/live_event_runner.py replay --workers 8 --reference "" --name $K "${SRC[@]}" \
    --frames-csv $W/frames/$K.csv --max-t $TE $([ "$POL" = P_AMBIGUITY ] && echo "--gate $GATE_DIR/$K.json") \
    --out $OUT/$K/run < /dev/null 2>&1 | grep -v "^\\[" > $OUT/$K/run.log
  echo "$K 끝 $(date +%H:%M:%S) | $(grep -h '멈춤 [0-9]*번,' $OUT/$K/run.log | tail -1) | $(grep -h '시설확인:' $OUT/$K/run.log)"
done
echo "$NAME 전부 끝 $(date +%H:%M)"
