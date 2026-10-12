#!/usr/bin/env bash
# 연구 저장소의 층 판별 지도를 이 패키지 maps/ 로 **실물 복사**한다(링크를 따라가서 복사).
# 지도를 고친 뒤(정리 단계 → regen_all.sh) 로봇에 넣기 전에 한 번 돌린다.
#   tools/sync_maps.sh            # 세 건물 전부
#   tools/sync_maps.sh sw         # 한 건물만
# 원본 위치는 각 실행기(run_v2*.py)의 SEMLOC_MAPS_ROOT·SEMLOC_SEMMAPS_ROOT 기본값과 같다.
set -euo pipefail
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 정보관 의미지도만 이름이 다르다(a131_semmaps, rt_localizer.py 기본값).
src_of() { if [ "$1_$2" = kci_semmaps ]; then echo /home/user/a131_semmaps; else echo "/home/user/$1_$2"; fi; }
for b in "${@:-kci gong sw}"; do
  for b1 in $b; do
    for kind in maps_perfloor semmaps; do
      src="$(src_of "$b1" "$kind")"
      dst="$PKG/maps/${b1}_${kind}"
      [ -d "$src" ] || { echo "없음: $src" >&2; exit 1; }
      rm -rf "$dst.new"
      cp -rL "$src" "$dst.new"
      # 지도 해상도(yaml)는 링크된 실제 지도 옆에 있어 cp -L 로는 안 따라온다. 소융대처럼 축척을 지도별로
      # 잡는 건물(per_map)은 이것이 없으면 floor_id.py 가 멈춘다. base_map.png 옆에 같이 넣는다.
      if [ "$kind" = maps_perfloor ]; then
        for png in "$src"/*/maps/base_map.png; do
          y="$(realpath "$png")"; y="${y%.png}.yaml"
          rel="${png#$src/}"
          [ -f "$y" ] && cp "$y" "$dst.new/${rel%.png}.yaml"
        done
      fi
      # 엘리베이터는 엘리베이터 아이콘이 있을 때만 의미로 쓴다(2026-10-05 결정). 정보관 예전 의미지도(8월 MCL)의
      # elevator.png 는 네모+X 묶음을 엘리베이터로 칠한 것이라 비운다. 층 판별(lam_sem=0)은 이 마스크를 쓰지 않는다.
      if [ "$kind" = semmaps ] && [ "$b1" = kci ]; then
        for e in "$dst.new"/*/mcl/elevator.png; do
          [ -f "$e" ] && python3 -c "import cv2,sys; m=cv2.imread(sys.argv[1],0); m[:]=0; cv2.imwrite(sys.argv[1],m)" "$e"
        done
      fi
      rm -rf "$dst"; mv "$dst.new" "$dst"
      echo "복사 $src -> maps/${b1}_${kind} ($(find "$dst" -type f | wc -l)개 파일)"
    done
  done
done
