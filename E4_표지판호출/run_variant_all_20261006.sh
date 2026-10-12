#!/bin/bash
# 2026-10-06 지금 코드에 인자(EXTRA)만 더해 13개 층 전체 + 비교(42_ 와 같은지, 점검, 판정 채점). run_current_all_20261005.sh 에서 만듦.
# GATE_NAME=모호함_v2.json(게이트 파일 이름), NOGATE=1(게이트 없이 = 무조건 멈춤 P_ALL)
# 예: OUTD=48_5m제한_지금문구 VN=5m지금 EXTRA="--ask-dmax 5" scripts/run_variant_all_20261006.sh
# 결과: 31_벽구간이벤트/<OUTD>/<층>/run, 비교 <OUTD>/점검.jsonl·판정채점.txt·같은지.txt(42_ 와 같은지). SC_DIR 을 주면 층마다 --search-config $SC_DIR/<층>.json
cd /data/kci_work/KCI_L0_P0_UNIFIED
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
G=reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트
O=$G/35_위치추정_축척추정; N=$G/42_새구조_최종문구; C46=$G/${OUTD:-46_${VN:-현재코드}_전층}; mkdir -p $C46
B=/data2/loc_inputs_20261005
D=/data/kci_work/maps_perfloor_20261001
C=/data3/kci_datasets/depth_cache_sign_eval
t0(){ python3 -c "import csv;r=next(csv.DictReader(open('/data3/kci_datasets/$1_RGB_FULL_5HZ_V0_1/index.csv')));print(int(r['bag_timestamp_ns'])/1e9-float(r['bag_relative_sec']))"; }
TABLE="4F|/data/kci_work/a131_full_v0_4/odom_link.csv
1F|/data2/a131_rebuild/a131_1F_clean_v7/odom_link_lc.csv
2F|/data2/a131_rebuild/a131_2F_clean_v7t/odom_link_icp_winfar06.csv
3F|/data2/a131_rebuild/a131_3F_clean_v7tm5/odom_link_lc_winfar06.csv
5F|/data2/a131_rebuild/a131_5F_clean_v7t/odom_link_lc_winfar06.csv
6F|/data2/a131_rebuild/a131_6F_clean_v5/odom_link.csv
7F|/data2/a131_rebuild/a131_7F_clean_v5m2/odom_link_icp.csv
8F|/data2/a131_rebuild/a131_8F_clean_v5/odom_link.csv
9F|/data2/a131_rebuild/a131_9F_clean_v7/odom_link_icp.csv
G1F|/data2/a131_rebuild/a131_G1_clean_v7m2/odom_link_fuse_pos1s_win06.csv
G2F|/data2/a131_rebuild/a131_G2_clean_v7/odom_link_fuse_pos1s_win06.csv
S1F|/data2/a131_rebuild/a131_S1_clean_v7t/odom_link_fuse_pos1s_winfar06.csv
S6F|/data2/a131_rebuild/a131_S6_clean_v7t3/odom_link_fuse_pos1s_winfar06.csv"
echo "지금 코드 13개 층 시작 $(date +%H:%M)"
while IFS='|' read F ODOM; do
  PPM=$(python3 -c "import csv;print(round(float(list(csv.DictReader(open('$O/$F/loc/timeline.csv')))[-1]['ppm_est']),3))")
  if [ $F = 4F ]; then SRC=(); else
    SRC=(--root /data3/kci_datasets/${F}_RGB_FULL_5HZ_V0_1 --frames-csv $C/${F}_full_frames.csv --depth $C/${F}_full.npz --odom $ODOM --t0 $(t0 $F)); fi
  mkdir -p $C46/$F
  [ -f $C46/$F/run/events_live_replay.json ] || python3 -u scripts/live_event_runner.py replay --workers 8 --reference "" --name ${F}_full "${SRC[@]}" \
    $([ -z "$NOGATE" ] && echo "--gate $O/$F/${GATE_NAME:-모호함.json}") --fac-map-topk $O/$F/loc/topk_series.csv --fac-map-sem $D/$F/${F}_semantic_map.json --fac-map-ppm $PPM \
    --fac-map-base $D/$F/base_map.png ${EXTRA} ${SC_DIR:+--search-config $SC_DIR/$F.json} --out $C46/$F/run 2>&1 | grep -v "^\[" > $C46/$F/run.log
  echo "$F 끝 $(date +%H:%M) | $(grep -h '멈춤 [0-9]*번,' $C46/$F/run.log | tail -1) | $(grep -h '시설확인:' $C46/$F/run.log)"
done <<< "$TABLE"

echo "비교 $(date +%H:%M)"
python3 - $N $C46 <<'PY' | tee $C46/같은지.txt
import json, sys
n, c = sys.argv[1:3]
for F in "1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F".split():
    r = []
    for name in ("events_live_replay.json", "facility_events.json"):
        a = json.load(open(f"{n}/{F}/run/{name}")); b = json.load(open(f"{c}/{F}/run/{name}"))
        strip = lambda L: [{k: v for k, v in e.items() if k != "vlm_sec"} for e in L]
        r.append(f"{len(a)}→{len(b)} {'같음' if strip(a) == strip(b) else '다름'}")
    print(F, "번호판 멈춤", r[0], "| 계단 멈춤", r[1])
PY
rm -f $C46/점검.jsonl
for F in 1F 2F 3F 4F 5F 6F 7F 8F 9F G1F G2F S1F S6F; do
  R=$C46/view/$F; mkdir -p $R
  ln -sfn $PWD/$O/$F/loc $R/loc; ln -sfn $PWD/$O/$F/모호함.json $R/모호함.json; ln -sfn $PWD/$C46/$F/run $R/run
  ln -sfn $PWD/$C46/$F/run $C46/view/${F}_${VN:-현재코드}
  if [ $F = 4F ]; then OD=/data/kci_work/a131_full_v0_4/odom_link.csv; else OD=$B/$F/odom_link.csv; fi
  python3 scripts/audit_events_vs_gt.py $F $R $G/36_채점용정답/$F $D/$F/${F}_semantic_map.json $D/$F/base_map.png $OD | sed "s/^{/{\"V\": \"${VN:-현재코드}\", /" >> $C46/점검.jsonl
done
python3 scripts/score_labeled_stops.py $C46/view ${VN:-현재코드} > $C46/판정채점.txt 2>&1
echo 전부 끝 $(date +%H:%M)
