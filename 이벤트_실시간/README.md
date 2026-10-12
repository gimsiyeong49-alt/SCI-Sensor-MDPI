# 표지판·계단 이벤트 인수인계 (2026-10-09)

위치추정이 애매할 때 로봇이 화면에서 호실 표지판(문 옆 표지판, 복도에 돌출된 표지판, 화장실 표지 포함)을 찾아 멈추고, 팬틸트 고화질 카메라로 겨누게 하는 코드입니다. 계단은 RealSense 화면을 Qwen VLM 에 물어 걸으면서 기록만 하고 멈추지 않습니다.

10-06 판에서 바뀐 것은 `수정사항_20261008.md`, 10-08 판에서 바뀐 것은 `수정사항_20261009.md` 에 적었습니다.

이 폴더는 인수인계용 사본입니다. 로봇에 이미 있는 `~/pantilt_calib`, `~/go2_ws` 는 건드리지 않았습니다.

## 지금 상태

녹화본 13개 층(정보관 1~9층, 공학관 1·2층, 소융대 1·6층)으로 검증을 끝냈습니다. 로봇 실주행 시험은 아직입니다.

| 항목 | 값 |
|---|---|
| 표지판 보러 멈춤 | 13개 층 합계 321번 |
| 그중 실제 표지판(사람 판정, 판정한 4개 층 74번) | 58% |
| 계단 관측 | 35번 중 30번 맞음 |
| 처리 시간 | 장면당 0.16~0.20초(노트북 GPU), 1초에 2장 |

## 알고리즘

1. 위치추정(협업자 RTLocalizer, 60초 창, 축척·방향은 주행 중 추정)이 애매하고 오돔·센서가 정상일 때만 멈춤을 허용합니다(`--stop-rule amb_sensor_ok`, 10-07 부터 기본). 확실할 때는 3초에 한 장만 처리합니다.
2. 찾을 대상은 실행할 때 의미지도를 읽어 정합니다(`make_search_config.py`, 표준 6종 중 지도에 있는 것만). 지금 13개 층은 모두 호실 표지판 + 계단이고, 엘베는 쓰지 않습니다.
3. RealSense 화면에서 작은 후보를 뽑아 깊이 7 m 안의 벽 구간으로 묶고, SigLIP 문구 세 개("a small room name plate", "a small room name plate near a door", "a hallway sign perpendicular to the wall") 중 가장 높은 점수를 씁니다.
4. 실제 폭 0.85 m 넘음, 바닥 기준 높이 0.6~2.8 m 밖, 게시판처럼 보임은 뺍니다. 점수는 지금까지 본 구간의 상위 10% 선과 바닥값 0.078 을 둘 다 넘어야 합니다.
5. 벽 자리 점수는 그 자리를 최근 3번 본 점수의 평균입니다(`--spot-recent 3`). 다가갈수록 표지판이 커져 점수가 정확해지므로 멀리서 본 낮은 점수를 끌고 다니지 않습니다.
6. 아직 안 본 벽 자리(0.7 m) 중 1위 한 곳에서 멈춥니다. 최근 60초 안에 멈춘 자리에서 2 m 안이면 같은 표지판으로 보고 건너뜁니다(`--ask-same-m 2.0 --ask-same-sec 60`). 1초 뒤 지금 자세로 다시 찾아 팬틸트를 겨눕니다.

자세한 경과와 숫자는 `docs/00_정리_위치추정교체_20261005.md` 맨 위 "현재 알고리즘"에 있습니다.

## 구성 (지금 방식: 노트북이 계산, 로봇은 화면·조준)

```
Go2  aim_console.py  ──화면 /live.mjpg, 깊이 /live/depth.png, 오돔 /live/odom──▶  노트북 live_event_runner.py live
     (팬틸트·IMX477)  ◀──────────────조준 /aim?u=&v=&z=&strict=1────────────────  (SigLIP·Qwen, GPU)
                                         위치추정 1초 기록 ──▶ --loc-url
```

로봇 쪽 `code/aim_console.py`, `code/calib_console.py` 는 지금 로봇의 `~/pantilt_calib/scripts` 판에 실시간 화면·깊이·오돔 내보내기와 `--odom-topic` 을 더한 판입니다. 로봇 판은 아직 바꾸지 않았습니다. 바꿀 때는 기존 파일을 백업하고 덮어씁니다. 추가한 기능은 옵션을 주지 않으면 꺼져 있어 지금 쓰는 방식은 그대로 됩니다. 그리고 Go2 실행 설정에서 RealSense 깊이가 꺼져 있는데, 이벤트가 깊이(거리, 높이, 3D 위치)를 쓰므로 켜야 합니다.

## 실행

모든 명령은 `code/` 안에서 실행합니다(`full_run_events.py` 가 `reports/…/grid.json` 을 상대 경로로 읽습니다).

```bash
# 로봇: 팬틸트 콘솔(화면·깊이·오돔 내보내기)
python3 aim_console.py --odom-topic <오돔 토픽>

# 노트북: 이벤트(4층 예)
python3 live_event_runner.py live --go2 http://<Go2 주소>:8088 --aim \
  --fac-map-sem ../maps/4F/4F_semantic_map.json --fac-map-base ../maps/4F/base_map.png --out <결과 폴더>
```

`--loc-url` 을 주지 않으면 위치추정과 상관없이 표지판이 보일 때마다 멈춥니다. 조준·초점·판독 연결 시험은 이렇게 하면 됩니다. 실시간 위치추정이 생기면 `--loc-url` 로 붙입니다. 위치추정 쪽은 `GET <url>?after=<stamp>` 에 아래 JSON 목록을 돌려주면 됩니다.

```json
[{"stamp": 1788347165.57, "level": "AMBIGUOUS", "margin": 0.61, "ppm": 32.9,
  "odom": [x, y], "cands": [[score, x_px, y_px, yaw_rad], ...]}]
```

`odom_yaw`(rad) 와 `sensor_bad`(bool) 는 넣어도 되고 빼도 됩니다. `sensor_bad` 가 true 면 그 초는 멈추지 않습니다. 위치추정 앞단에서 `code/sensor_check.py` 의 `OdomGuard` 로 FAST-LIO 튐을 잡았을 때 true 로 보내면 됩니다.

`stamp` 는 로봇 시계(초), `level` 은 AMBIGUOUS 또는 CONFIDENT(1위와 5 m 밖 최고 후보의 점수 차 ≤ 1.5 면 AMBIGUOUS), `cands` 는 그 초의 top-K 입니다. `scripts/loc_topk_series.py` 의 timeline·topk 한 초와 같은 내용입니다. 기록이 끊기거나 3초보다 오래되면 그 순간은 애매로 봅니다.

녹화본으로 시험할 때는 `fake_go2_console.py`(녹화 프레임·깊이·오돔·위치추정 기록을 녹화 속도로 내보내는 흉내 콘솔)를 Go2 대신 씁니다.

## 로봇에서 직접 돌리려면 (아직 안 함)

지금은 노트북에서만 돌렸습니다. Orin 에서 돌리려면 모델 두 개(`siglip-base-patch16-224`, `Qwen3-VL-2B-Instruct`)를 옮기고, 코드에 박힌 경로 세 곳을 바꿔야 합니다. `live_event_runner.py` 의 `VLM_PY`(SigLIP·Qwen 작업자를 돌리는 파이썬), `region_siglip_rerank.py` 의 `MODEL`, `live_vlm_worker.py` 의 `MODEL`(환경변수 `KCI_VLM_MODEL` 로도 바꿀 수 있음)입니다. 로봇에는 torch 2.1(CUDA)과 transformers 4.46 이 이미 있습니다. Orin 속도는 재 보지 않았습니다.

## 폴더

| 경로 | 내용 |
|---|---|
| `code/` | 이벤트 코드 26개(노트북 `KCI_L0_P0_UNIFIED/scripts` 사본)와 import 때 읽는 `grid.json` |
| `code/live_event_runner.py` | 본체. `replay`(녹화본), `live`(Go2) |
| `code/make_search_config.py` | 의미지도 → 찾을 대상(표준 6종) |
| `code/sensor_check.py` | 오돔 튐 판단·고치기. `OdomGuard` 가 실시간판 |
| `code/aim_console.py`, `calib_console.py` | 로봇 쪽 팬틸트 콘솔 새 판 |
| `configs/search_targets_v3/` | 13개 층 찾을 대상(실행 때 의미지도에서 다시 만들므로 참고용) |
| `maps/<층>/` | 13개 층 의미지도 JSON 과 base_map.png |
| `docs/` | 정리 문서, 팬틸트 현장 안내 |

## 남은 일

1. 로봇 쪽 팬틸트 콘솔 새 판으로 교체(백업 후), RealSense 깊이 켜기, 오돔 토픽 확인
2. 현장 시험: 멈춤 → 조준 → 초점 → 촬영 → 판독
3. 실시간 위치추정을 `--loc-url` 로 연결. FAST-LIO 자세는 위치추정에 넣기 전에 `OdomGuard` 를 거치게(a1_3_1 라이브판은 `~/floor_id_pkg/a131_live/`)
4. 위치추정 후보가 방 안에 찍히는 경우(4층 1위의 5%)와 같은 자리 재멈춤(27%)은 손대지 않았습니다
