# KCI 위치추정 지도 — 5F

이 묶음은 **5F 단독** 이다. 다른 층은 각자의 묶음을 쓴다.
해상도 [1074, 3694], 마스크 8종 + base_map + (참고)MCL 네비지도.

---

# KCI 위치추정 — 1F~9F 지도 패키지 (2026-08-28)

4층에서 검증된 실시간 위치추정(`rt_localizer.py`)을 다른 층으로 옮기는 데 필요한 지도 일체.
4층 성적: held-out 5지점 중앙 0.74 m · 3 m 이내 5/5 (미래 데이터 0, GT 0).

## 구성

    maps/{1..9}F/
      base_map.png              바탕 안내도 (free>200, wall<50)  ★스케일·격자·거리장의 원천
      corridor_mask.png         복도 영역   -> corridor cue, 장면 사전확률
      room_mask.png             방 영역     -> inside_room cue
      stair_mask.png            계단        -> stairs cue + 후보 배제
      restroom_mask.png         화장실      -> restroom_sign cue + 후보 배제
      door_mask.png             문          -> door_left/right/ahead cue
      building_footprint.png    건물 내부   -> 건물 밖 후보 하드 배제
      no_go.png                 진입금지    -> 소프트 벌점
      elevator.png              엘리베이터  -> elevator cue
      navigation_map_MCL.png    (참고) MCL 쪽 네비지도. base_map 과 다른 층이 있음
    maps/4F/base_map_v2_USED_BY_4F.png   ★4층 검증에 실제로 쓴 판본(아래 주의)
    code/                       rt_localizer.py, rt_node.py, semantic_map.py 등
    MANIFEST.json               91개 파일의 크기·md5·해상도·점유율

**9층 모두 8종이 갖춰져 있고, 각 층에서 마스크와 base_map 의 해상도가 정확히 일치한다(검증됨).**

## 지도 외에 층별로 따로 필요한 것

| 항목 | 내용 |
|---|---|
| `odom_link.csv` | FAST-LIO 자세 (t_sec, fastlio_x/y, fastlio_yaw_rad, sample_id) |
| `samples/*/data_short.npz` | LiDAR 구조점 `a1_3_1_all_candidate_xy` + `_role` (role 4 제외) |
| `far_points.npz` | 원거리 점 보강 (없어도 동작) |
| rosbag `.db3` | 깊이 영상 — 장면 분류용. 없으면 `rt_node._scene_from_scan` 근사 사용 |
| VLM 관측 JSON | 그 층 프레임에 Qwen3-VL 재추론 필요 |

## ★층마다 반드시 다시 정해야 하는 값

    CORRIDOR_M = 94.6      # node_local.py:41, semantic_map.py:17  (4층 값)

`_find_scale()` 이 안내도에서 **가장 긴 가로 자유공간 구간**(픽셀)을 찾아 이 값으로 나눠 px/m 를 만든다.
층마다 최장 복도 실제 길이가 다르므로 반드시 실측 후 교체할 것. 틀리면 스케일이 통째로 어긋난다.
(기존 메모 "3F·4F 스케일 과대, LiDAR팀 실측 재확인 대기" 가 이 문제)

`BETA = 228°` (센서 장착 오프셋) 는 같은 로봇이면 층 무관.

## 주의: 4층만 v2 판본을 썼다

4층 검증은 `base_map_v2_USED_BY_4F.png` 로 했다. 이는 `base_map.png` 대비
**16,753 px (0.242%) 다르고 자유공간이 +235 px** 많은 손질 판본이다. 다른 층에는 대응 v2 가 없다.
따라서 다른 층은 `base_map.png` 로 시작하되, 성능이 낮으면 이 차이를 먼저 의심할 것.

`navigation_map_MCL.png` 는 1F·2F·5F 에서 `base_map.png` 와 내용이 다르다(3F·9F 는 동일).
어느 쪽을 쓸지는 층별로 확인이 필요하다. 마스크와 같은 디렉터리에서 온 `base_map.png` 를 기본 권장.

## 옮기는 순서

1. `semantic_map.py` 의 `SEM`/`MCL`/`LOCMAP`, `node_local.py` 의 `MAPPNG` 를 그 층 경로로
2. `CORRIDOR_M` 을 그 층 실측값으로 교체 → PPM 이 맞는지 확인
3. 그 층 주행 데이터(odom/samples) 준비
4. `Z.fix_phi(T[0], boot_s=15)` 로 φ 확정 — 시동창 점수가 한 mode 에서 뚜렷이 높아야 정상
   (4층 예: 306→+1.36 vs 나머지 -0.36/-2.67/-4.26)
5. VLM 없이 먼저 LiDAR 만으로 검증 → 그 다음 의미항 추가

## 함정 (재현 필수)

방향 히스토그램은 **y 를 뒤집고** 계산할 것: `PPM*(cloud @ A(0).T)`, `A(0)=[[1,0],[0,-1]]`.
누적 점군은 월드좌표(y 위), 지도는 이미지좌표(y 아래). 빠뜨리면 히스토그램이 거울상이 되어
φ 가 통째로 틀리고 증상은 "φ 를 뭘로 고정해도 전부 실패" 로 나타난다.
검증법: 기존 규약 등가값 `-ang` 이 φ modes 안에 있어야 한다.
