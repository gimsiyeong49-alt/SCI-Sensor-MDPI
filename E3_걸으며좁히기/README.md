# 코드: E3 걸으며좁히기

이 실험에 쓴 스크립트를 실제 import 관계를 따라 모았다. 원래 작업 폴더의 `scripts/` 에서 그대로 복사했고, 공유용이라 사람 이름이 든 이름만 내용 이름으로 바꿨다(예: 위치추정 1초 기록 스크립트는 `loc_topk_series.py`, 협업자 위치추정 패키지 경로는 `floor_id_pkg`).

그대로 돌리려면 세 가지가 더 필요하다. 협업자 위치추정 패키지(`floor_id`: `floor_id.py`, `rt_localizer.py`, `metric_fit_v2.py` 등, 이 묶음에 넣지 않음), 층별 데이터셋(A1.3.1 구조점·오돔·RGB-D 녹화, 채점용 정답)과 안내도 지도, 코드에 적힌 데이터 경로(`/data…`)를 각자 위치로 바꾸는 것이다. 모델(SigLIP, Qwen-VL)은 `/data2/vlm_env` 파이썬 환경에서 불렀다.

| 파일 | 하는 일 |
|---|---|
| `e2_dense.py` | E2 전 후보 점수판(2026-10-09). 출발점(과제) 하나에서 그 건물 모든 층 × 갈 수 있는 모든 격자 칸 × 방향 후보 × 축척 목록의 점수를 낸다. |
| `e2_dense_score.py` | E2 Top-100 채점(평가 전용, 2026-10-09; 2026-10-10 정답 후보·정답 층 최고 후보의 위치·방향·축척 오차, 1위 층·축척, 계산 시간 열 추가). e2_dense.py 가 저장한 전 후보 점수판(모든 층 × 칸 × 방향 × 축척)에서 |
| `e2_scene_labels.py` | E2 출발점 65곳의 장면 라벨(2026-10-10). 정답은 읽지 않는다. |
| `e2_visual_ko.py` | E2 이해용 시각 자료(2026-10-10). 막대 대신 지도·사진 위에서 보이게 한다. 결과: _01_E2_초기위치/시각자료/ |
| `e3_accum_registered.py` | E3 쌓인 라이다를 '그 시점 후보 자세'로 지도에 얹기(2026-10-11, 협업자 위치추정 그림의 스캔 정합과 같은 식). |
| `e3_dense.py` | E3 걸으며 후보 좁히기 점수판(2026-10-10). E2 원후보 전체(모든 층 × 갈 수 있는 칸 × 방향 4 × 축척 7)를 FAST-LIO 이동으로 따라가며 시점마다 점수를 낸다. |
| `e3_main_figures.py` | E3 본 결과 그림(2026-10-11): 노드 그대로 / 연결판(E2 후보 이어받기). 시점마다 폴더, 그 안을 건물 → 층 → 출발점으로 나눈다. |
| `e3_main_readme.py` | E3 본 결과 폴더(1_노드그대로, 2_연결판_E2후보이어받기)의 시점별 README: 건물 → 층별 표, 그림 링크(2026-10-11). |
| `e3_node.py` | E3 최종 노드 그대로(2026-10-11): 로봇 실시간 노드(realtime/rt_live_node → floor_id 와 같은 알고리즘)를 출발점마다 막 켠 상태로 흉내낸다. |
| `e3_online.py` | E3 기존 방식 조건(T3 + 축척·방향 온라인 추정, 2026-10-10). E4 때 협업자 위치추정을 돌린 방식(scripts/loc_topk_series.py --scale-online)과 같은 규칙. |
| `e3_report.py` | E3 채점 모음·표·그림(평가 전용, 2026-10-10). e3_dense.py 점수판을 e2_dense_score.py 로 채점한 결과(_작업자료/score/<조건>_T<T>)를 모은다. |
| `e3_seeded.py` | E3 연결판(2026-10-11): 정지 때 만든 E2 Top-100(A 라이다·축척 탐색·+ 정밀 검증)을 이어받아, 걷는 동안 노드(floor_id)와 같은 후보 처리로 다시 매긴다. |
| `e3_static_vs_time.py` | E3 시점별 '정지 대비' 그림·통계(2026-10-11). 15·30·45·60초를 시점마다 따로, 각 그림은 정지(0초 = E2 제자리 한 번)와 그 시점을 나란히. |
| `e4_figures.py` | E4 논문 그림(2026-10-07, PILOT). e4_paper.py·e4_eval.py·e4_detection.py 결과를 읽어 59_E4_호출정책/paper/figures/ 에 PNG·PDF 로 그린다. |

## 실행 순서 (실험 README 의 '다시 돌리려면')

```bash
python3 scripts/e3_node.py --times 15,30,45,60 --workers 6      # 노드 그대로(/data2/vlm_env/bin/python 이면 GPU, 결과 같음)
python3 scripts/e3_node.py --score
python3 scripts/e3_seeded.py --workers 18                        # 연결판(노드 결과 필요)
python3 scripts/e3_seeded.py --score
E3_ADD_NODE=1 python3 scripts/e3_seeded.py --workers 12          # 합치기(노드 결과 필요, 결과 /data2/e3_merged_20261012)
E3_ADD_NODE=1 python3 scripts/e3_seeded.py --score               # → _작업자료/연결판_재탐색/
E3_ADD_NODE=1 E3_NODE_EVERY=30 python3 scripts/e3_seeded.py --out <폴더>   # 다시 찾기 간격 비교
python3 scripts/e3_main_figures.py --modes node,seeded,merged --workers 8   # 2_비교/1_노드그대로_로봇방식, 2_연결판, 1_최종_합치기 그림
python3 scripts/e3_main_readme.py                                # 시점 폴더 README
python3 scripts/e3_online.py ; python3 scripts/e3_static_vs_time.py --conds T3on   # 3_비교
python3 scripts/e3_dense.py ; python3 scripts/e3_report.py --score                 # 부록(조건 쪼개기)
```
