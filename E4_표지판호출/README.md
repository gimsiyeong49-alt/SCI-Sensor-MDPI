# 코드: E4 표지판호출

이 실험에 쓴 스크립트를 실제 import 관계를 따라 모았다. 원래 작업 폴더의 `scripts/` 에서 그대로 복사했고, 공유용이라 사람 이름이 든 이름만 내용 이름으로 바꿨다(예: 위치추정 1초 기록 스크립트는 `loc_topk_series.py`, 협업자 위치추정 패키지 경로는 `floor_id_pkg`).

그대로 돌리려면 세 가지가 더 필요하다. 협업자 위치추정 패키지(`floor_id`: `floor_id.py`, `rt_localizer.py`, `metric_fit_v2.py` 등, 이 묶음에 넣지 않음), 층별 데이터셋(A1.3.1 구조점·오돔·RGB-D 녹화, 채점용 정답)과 안내도 지도, 코드에 적힌 데이터 경로(`/data…`)를 각자 위치로 바꾸는 것이다. 모델(SigLIP, Qwen-VL)은 `/data2/vlm_env` 파이썬 환경에서 불렀다.

| 파일 | 하는 일 |
|---|---|
| `audit_events_vs_gt.py` | 이벤트 알고리즘 점검(2026-10-05): 채점용 정답으로 게이트·멈춤을 숫자로 본다. |
| `combined_policy.py` | 지금까지 효과가 확인된 것을 묶은 조합. 멈출 때마다 고화질 카메라가 볼 구간을 정한다. |
| `e2_dense.py` | E2 전 후보 점수판(2026-10-09). 출발점(과제) 하나에서 그 건물 모든 층 × 갈 수 있는 모든 격자 칸 × 방향 후보 × 축척 목록의 점수를 낸다. |
| `e2_visual_ko.py` | E2 이해용 시각 자료(2026-10-10). 막대 대신 지도·사진 위에서 보이게 한다. 결과: _01_E2_초기위치/시각자료/ |
| `e3_seeded.py` | E3 연결판(2026-10-11): 정지 때 만든 E2 Top-100(A 라이다·축척 탐색·+ 정밀 검증)을 이어받아, 걷는 동안 노드(floor_id)와 같은 후보 처리로 다시 매긴다. |
| `e4_detection.py` | E4 검출 정확도(2026-10-07, 실험 지침서 14.6, 파일럿). 사람 판정 멈춤 248개(표지판/아님)의 SigLIP 점수(지금 기본 문구 셋 최댓값)로 |
| `e4_eval.py` | E4 호출 정책 비교 채점(2026-10-07, 실험 지침서 14.5~14.6). 파일럿(개발 데이터) 결과다. |
| `e4_extra.py` | E4 추가 지표(2026-10-08, 실험 지침서 14.3·14.6·14.7, PILOT). 59_E4_호출정책/paper/ 아래에 만든다. |
| `e4_fault_eval.py` | E4 센서 이상 주입 결과 채점(2026-10-08, PILOT). e4_fault_inject.py 로 튐을 넣은 오돔으로 협업자 위치추정을 다시 돌린 뒤 |
| `e4_fault_eval2.py` | E4 오돔 튐 시험, 고친 판단 비교(2026-10-08, PILOT). 같은 4층 튐 8개에 대해 세 조건을 비교한다. |
| `e4_fault_inject.py` | E4 센서 이상 주입 시험(2026-10-08, 실험 지침서 14.1·14.6 '센서 오류를 장소 모호성으로 혼동한 빈도', PILOT). |
| `e4_figures.py` | E4 논문 그림(2026-10-07, PILOT). e4_paper.py·e4_eval.py·e4_detection.py 결과를 읽어 59_E4_호출정책/paper/figures/ 에 PNG·PDF 로 그린다. |
| `e4_figures_ko.py` | E4 설명용 한국어 그림(2026-10-08). 논문용(e4_figures.py)보다 단순하게: 그림 하나에 말 하나, 패널 하나. |
| `e4_fp_types.py` | E4 오검출 종류 집계(2026-10-08, PILOT). 25_판정/오검출종류_추정.json(사용자가 짐작을 확인함)으로 |
| `e4_geom_cells.py` | E4 '기하 부족' 판정용: 층마다 a1_3_1 표본(위치추정 입력, /data2/loc_inputs_20261005/<층>)의 쓰는 칸 수(role != 4)와 |
| `e4_informative.py` | E4 호출 정책 P_INFORMATIVE: 멈춤(이벤트)마다 '읽으면 지금 후보들을 가를 수 있나'를 판단한다(2026-10-07, 실험 지침서 14.5). |
| `e4_meta.py` | E4 처리 시간 표와 판(버전) 기록(2026-10-07, 실험 지침서 14.6·18절, PILOT). 59_E4_호출정책/paper/ 아래에 만든다. |
| `e4_package_ko.py` | E4 정리본(배포판 C) 한글 꾸러미(2026-10-10). 작업 폴더 59_E4_호출정책/paper_C 를 복사해 한글 파일·열 이름으로 바꾼다(원본은 그대로). |
| `e4_paper.py` | E4 논문용 자료 정리(2026-10-07, 실험 지침서 14절·19절, PILOT). 59_E4_호출정책/paper/ 아래에 만든다. |
| `e4_read_loop.py` | E4 운용 흉내(2026-10-13): 첫 진짜 번호판에서 층을 알게 되면 그 뒤로는 그 층 후보만으로 판단한다. |
| `e4_recall.py` | E4 놓친 표지판 비율(호실 단위 재현율, 2026-10-08, PILOT). 프레임 단위 정답이 4층 60초뿐이라 의미지도와 채점용 정답 경로로 자동으로 낸다. |
| `e4_recall_why.py` | E4 호실 단위 재현율 분석(2026-10-08, PILOT). e4_recall.py 결과(호실마다)와 멈춤을 놓고 층마다: |
| `e4_sensor_inject.py` | E4 센서 이상 탐지 주입 시험(2026-10-12, PILOT). 65개 출발점마다 구간 30초 지점에 오돔 순간 이동·순간 회전을 넣고 |
| `e4_state_eval.py` | E4(새 계획서 6장) 위치 모호성·센서 상태 판단 평가(2026-10-12, PILOT). 카메라·멈춤은 다루지 않는다(E5). |
| `e4_state_figures.py` | E4 결과물(새 계획서 6.7): 출발점마다 후보 점수·순위 변화와 매 시점 위치 상태 그림(2026-10-12). |
| `e4_unknown_gate.py` | 층 모름 E4(지침서판, 2026-10-11): 출발점 구간의 매 장면(초) 애매 기록을 E3 연결판 후보(건물 전체 층)로 만든다. |
| `e4_unknown_informative.py` | 층 모름 E4 P_INFORMATIVE(2026-10-11): 출발점 구간 실행(P_AMBIGUITY)의 멈춤마다 '읽으면 남은 가설을 가를 수 있나'를 판단한다. |
| `e4_unknown_score.py` | 층 모름 E4 채점(2026-10-11, PILOT). 65개 출발점 60초 구간 실행(70_층모름E4/<정책>/<출발점>/run)을 정책별로 비교한다. |
| `e4_unknown_segments.py` | 층 모름 E4 출발점 구간 정의(2026-10-11). 애매 기록(e4_unknown_gate.py)과 구간 실행(run_e4_unknown_20261011.sh)이 같이 쓴다. |
| `e4_unknown_stop_sheets.py` | E4 층 모름(합치기 후보) 출발점마다 멈춘 장면 사진표(2026-10-12). 사람 판정을 테두리 색으로 보여 준다. |
| `e4_unknown_visual_ko.py` | E4 층 모름(지침서판, 합치기 후보) 이해용 시각 자료(2026-10-12). 배포판 C 판(e4_visual_ko.py V1~V6)과 같은 모양으로 만든다. |
| `e4_visual_ko.py` | E4 이해용 시각 자료(2026-10-08). 사용자가 PPT 그림을 직접 만들기 전에 '무슨 실험을 어떻게 했고 전층에서 어떻게 나왔나'를 |
| `event_criteria.py` | RGB에서 어떤 근거가 있을 때 '관측 시작 이벤트'를 낼지 조건을 재 본다. |
| `event_trigger_sim.py` | 매초 멈추는 대신 '표지 같은 자리가 처음 D m 안에 들어올 때'만 멈추는 이벤트 규칙을 시뮬레이션한다. |
| `full_run_events.py` | 정답 없는 전체 주행에 지금 파이프라인을 1초 간격으로 돌리고, 멈춤 이벤트가 언제 나는지 계산한다. |
| `geometry_filter.py` | 후보에서 바닥·천장 쪽을 걷어낸다 (1단계 보강, 기하 단서). |
| `loc_topk_series.py` | 협업자 최종 위치추정(RTLocalizer, Go2 floor_id_pkg/floor_id/localizer/rt_localizer.py)을 1초마다 불러 |
| `live_event_runner.py` | 표지 멈춤 이벤트를 실시간으로 돌린다. 녹화 데이터를 실시간처럼 흘려 넣어 기존 결과와 대조할 수도 있다. |
| `live_fast.py` | 실시간 실행기용 빠른 버전. 결과는 기존 함수와 비트 단위로 같아야 한다(verify 로 확인). |
| `live_siglip_worker.py` | 실시간 이벤트 실행기용 SigLIP 작업자. vlm_env(torch)에서 돌고, 표준 입출력으로 주고받는다. |
| `live_vlm_worker.py` | 실시간 실행기용 VLM(Qwen3-VL) 작업자. vlm_env 에서 돌고 표준 입출력으로 주고받는다(live_siglip_worker 와 같은 방식). |
| `make_gate_json.py` | 위치추정 1초 기록(export_4f_causal_topk_series.py 출력: timeline.csv, topk_series.csv)으로 게이트 기록(json)을 만든다. |
| `make_gate_switch.py` | 모호함 기록(모호함_v2.json)에 '1위 순위 급변 뒤 K초는 애매'를 더한 새 기록을 만든다(2026-10-11, 모호함 기준 재정립). |
| `make_label_sheet.py` | 미판정 멈춤(score_labeled_stops.py 가 쓴 미판정_멈춤.json)을 사람 판정용 사진표로 만들고 판정목록에 새 번호로 덧붙인다(2026-10-05). |
| `make_search_config.py` | 층별 '찾을 대상' 설정 파일을 의미지도에서 자동으로 만든다(2026-10-05). |
| `obs_feedback.py` | 이벤트 관측(호실 번호판·계단·엘베)을 위치추정 후보 점수에 반영한다(2026-10-05, 작업 순서 ②). |
| `region_memory.py` | 벽 구간을 벽 위 좌표로 기억하고, 다가가는 동안 모인 증거로 순서를 다시 매긴다. |
| `region_siglip_rerank.py` | 벽 구간 상위 8개를 SigLIP으로 한 번 더 보고 순서만 다시 매긴다. |
| `run_e4_merged_20261012.sh` | E4 층 모름을 E3 합치기 후보(E2 후보 + 노드 재탐색, 3초)로 다시(2026-10-12). 기존(_merged 없는) 결과는 그대로 둔다. |
| `run_e4_unknown_20261011.sh` | 층 모름 E4(지침서판, 2026-10-11): 65개 출발점 [t_s, t_s+60] 구간만 이벤트 실행기(배포판 C 기본값)로 돌린다. |
| `run_e4_v8_20261012.sh` | 2026-10-12 E4 층 모름을 지도 8차로 다시: 3초 간격 합치기 후보(노드 v8 재사용) → 애매 기록 → 애매할 때 재생(65곳 반씩 동시) → 갈릴 때 거르기 → 채점 → 상태 판단 → 그림. |
| `run_variant_all_20261006.sh` | 2026-10-06 지금 코드에 인자(EXTRA)만 더해 13개 층 전체 + 비교(42_ 와 같은지, 점검, 판정 채점). run_current_all_20261005.sh 에서 만듦. |
| `score_labeled_stops.py` | 사람 판정표(25_판정)로 실행 결과의 표지판 멈춤을 채점한다(2026-10-04). |
| `score_loc_pseudo_gt.py` | 채점용 정답(scripts/make_pseudo_gt.py 의 gt.csv·summary.json)으로 위치추정 1초 기록을 채점한다(2026-10-05). |
| `sensor_check.py` | 오돔·센서 이상 판단(2026-10-08, 실험 지침서 14.1 ODOMETRY_OR_SENSOR_UNRELIABLE). make_gate_json 의 1초 속도 기준(2 m/s, 120°/s)이 |
| `sign_segsets.py` | 표지 후보 실험 구간 설정. 층마다 후보 파일, 정답, 프레임 폴더, 깊이 캐시, 오돔을 묶는다. |
| `small_proposals.py` | 리얼센스 RGB에서 '벽에 붙은 작은 판'이 될 만한 조각을 빠짐없이 뽑는다 (1단계: 후보 생성). |
| `surface_orient.py` | 깊이로 '후보가 붙은 면이 세로(벽)인가 가로(천장·바닥)인가'를 잰다(2026-10-04). |
| `topo_facility.py` | 의미지도 시설 판정 — 위상 방식(2026-10-04). |
| `wall_region_events.py` | 멈췄을 때 '확인할 벽 구간'을 몇 개 낼지, 그 안에 번호판이 들어오는지 잰다. |

## 실행 순서 (실험 README 의 '다시 돌리려면')

작업 폴더에서 각 스크립트에 `E4_MAIN=P_R3B E4_MAIN_NAME="C 배포판" E4_PAPER=paper_C E4_METRICS=E4_trigger_policy_metrics_dedup.csv E4_FB=feedback_oracle_dedup E4_AUDIT=64_최근3번평균_재멈춤막기` 를 주고 `e4_paper.py → e4_fp_types.py → e4_recall.py P_ALL,P_AMBIGUITY,P_R3B → e4_recall_why.py → e4_extra.py → e4_figures.py → e4_figures_ko.py → e4_visual_ko.py → e4_meta.py` 순서로 돌리면 `paper_C/` 가 나온다. 이 정리본은 `python3 scripts/e4_package_ko.py` 로 `paper_C/` 에서 다시 만든다(README 는 손으로 쓴 것).
