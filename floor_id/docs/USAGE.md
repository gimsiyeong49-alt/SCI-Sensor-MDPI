# 온보드 층 판별 — 사용법

안내도로 만든 지도 위에 누적 LiDAR 를 끼워 **지금 몇 층인지** 맞힌다.
**깊이 카메라를 쓰지 않는다**(2026-09-28 결정, 근거는 `docs/DEPTH.md`).

---

## 1. 5분 안에 돌려보기

```bash
cd onboard_robot_20261001/floor_id
python3 floor_id.py --building sw --data /경로/주행데이터
```

출력 예

```
[소융대] 지도 8개 · 시점 6개 · 준비 47.2s · 장면 lidar · 창 60.0s
     70s  판정 7층  (2위 6층)  위치 px 2841,612  ppm 58.7  φ 314  벽비용 6.3cm  장면 corridor
    130s  판정 6층  (2위 2층)  ...
저장 out/sw/result.json
```

`--building` 은 셋 중 하나다.

| 키 | 건물 | 층 |
|---|---|---|
| `kci` | 정보관 | 1F~9F |
| `gong` | 공대 | G1F~G7F |
| `sw` | 소융대 | S1F~S8F |

자주 쓰는 인자

```bash
--at 70 130 190      # 이 시점(초)만 판별. 생략하면 70초부터 60초 간격 전부
--window 60          # 정합 창 길이(초). 기본 60. 바꾸면 성적을 다시 재야 한다
--out 결과폴더        # 기본 out/<건물>
```

---

## 2. 필요한 것

### 실행 환경
- Python 3.10, `numpy` `scipy` `opencv-python`(`cv2.ximgproc` 포함) `pyyaml` `pillow`
- GPU 불필요. CPU 로 돈다(구간·지도 조합 하나에 8~25초).
- ROS 불필요. 아래 데이터 형식만 맞으면 된다.

### 주행 데이터 폴더 (`--data`)

```
주행폴더/
  odom_link.csv      t_sec, fastlio_x, fastlio_y, fastlio_yaw_rad, sample_id
  samples/<sample_id>/data_short.npz
                     a1_3_1_all_candidate_xy   (N,2) 로봇 몸체 기준 구조점 [m] (x 앞, y 왼쪽)
                     a1_3_1_all_candidate_role (N,)  4 = 제외할 점
  far_points.npz     sample_id 별 (M,2) 원거리 점(9~18m). 없어도 돌지만 성적이 떨어진다
```

- 좌표계: **x 앞 · y 왼쪽**, 단위 m. `fastlio_yaw_rad` 는 odom 기준 라디안.
- 이 형식은 기존 a1_3_1 추출 파이프라인 출력 그대로다. 새로 만들 필요 없다.
- `scene_cache_v1.npy`(깊이 장면 캐시)가 폴더에 있어도 **쓰지 않는다**.

---

## 3. 결과 읽기

`out/<건물>/result.json`

```jsonc
{
  "building": "sw", "n_maps": 8, "window_s": 60.0,
  "scene_source": "lidar",
  "rows": [
    { "ts": 130.0,
      "verdict": 6,                 // 판정된 층
      "ranking": [6, 2, 4, ...],    // twoway 점수 내림차순
      "cands": [ { "map": 6, "twoway": -0.83, "ppm": 58.7, "phi": 314,
                   "pos": [2841, 612], "wall_cost_m": 0.063, "scene": "corridor" }, ... ] }
  ]
}
```

- `verdict` — 그 시점의 판정 층.
- `pos` — 그 층 지도의 **픽셀** 좌표. m 로 바꾸려면 `pos / ppm`.
- `wall_cost_m` — 누적 LiDAR 점이 지도 벽에서 떨어진 평균 거리. **0.1 m 아래면 자세가 맞는 것**, 0.3 m 이상이면 그 층이 아니거나 자세가 틀렸다고 보면 된다.
- `scene` — 그 시점 LiDAR 가 본 장면(`corridor` / `junction` / `dead-end`).

### 성적 (참고용, 오프라인 실측)

세 건물 100구간에서 **정답 1위 21%(21/100) · 상위 3위 45%** (2026-10-01 지도, 창 60초 · 장면 lidar · 소융대 per_map).
무작위 기준은 1위 약 12%, 상위3 약 40% 다. **한 구간만 보고 믿으면 안 된다** — 여러 구간을 모아 다수결로 쓸 것.

| 건물 | 구간 | 1위 | 상위3 |
|---|---|---|---|
| 정보관 | 69 | 14 | 30 |
| 공대 | 21 | 5 | 10 |
| 소융대 | 10 | 2 | 5 (09-29 지도 6) |

---

## 4. 폴더 구성

```
floor_id/
  floor_id.py          ← 진입점. 이것만 실행하면 된다
  config/buildings.yaml  건물 3개 설정(지도 경로·층 수·축척 방식·상수)
  localizer/           알고리즘 18개 모듈 (192KB)
  maps/                세 건물 24개 층 지도 (9.1MB)
  tools/               검증·비교 도구 (온보드 실행에는 불필요)
  docs/                이 문서 · DEPTH.md · TUNING.md
```

지도는 패키지 안에 들어 있다. **경로를 고칠 필요가 없다.**

---

## 5. 다른 건물 추가하기

1. 그 건물 지도를 `maps/<이름>_maps_perfloor/<n>F/maps/` 와 `maps/<이름>_semmaps/<n>F/{sem,mcl}/` 에 넣는다
   (안내도에서 지도를 만드는 절차는 `HANDOFF_FLOORPLAN_ALL_20260921.md`).
2. `config/buildings.yaml` 에 항목을 추가한다.

```yaml
  myblg:
    name: 새건물
    maps_root: maps/myblg_maps_perfloor
    semmaps_root: maps/myblg_semmaps
    n_maps: 6
    scale: fixed        # 층별 안내도 해상도가 크게 다르면 per_map
```

**코드는 건드리지 않는다.** 층별 분기나 하드코딩을 넣지 말 것 — 이 프로젝트의 최우선 제약이다.

`scale` 고르는 법: 층마다 안내도 해상도(가로 픽셀)가 1.5배 이상 차이나면 `per_map`.
소융대가 3700~6400px 라 `per_map` 이고, 정보관·공대는 비슷해서 `fixed` 다.
`per_map` 은 각 층 `maps/<n>F/maps/base_map.yaml` 의 `resolution`(m/px)을 읽는다. **이 파일이 없으면 실행을 멈춘다**
(예전에는 조용히 `fixed` 로 떨어졌다 — 2026-10-01 까지 소융대 패키지가 그 상태였다). 지도는 `tools/sync_maps.sh` 로 넣으면 yaml 도 같이 들어간다.

---

## 6. 막히면

| 증상 | 원인 | 조치 |
|---|---|---|
| `주행 데이터에 ... 가 없다` | 폴더 형식 불일치 | §2 의 세 파일을 확인 |
| `cv2.ximgproc` 없음 | opencv-contrib 미설치 | `pip install opencv-contrib-python` |
| 준비 단계가 아주 느림 | 지도 수 × 부트스트랩 | 정상이다. 층당 5~7초, 9층이면 1분 안팎 |
| 모든 구간이 같은 층으로 몰림 | 복도만 지나가는 구간 | 복도는 층 구분 정보가 거의 없다. 교차로·특징 구간을 포함해 재실행 |
| 벽비용이 늘 0.3m 이상 | 축척 탐색 범위 밖 | `config` 의 `scale` 을 `per_map` 으로 바꿔 본다 |
| `base_map.yaml 이 없는 층` | per_map 건물에 해상도 파일 누락 | `tools/sync_maps.sh <건물>` 로 지도를 다시 넣는다 |

---

## 7. 바꾸면 안 되는 것

- `config/buildings.yaml` 의 `consts` — 창 길이·장면 각도 등. 바꾸면 위 성적표가 무효가 된다.
- `scene_side_near_m: 0`(제한 없음) — 옆 거리를 제한하면 성능이 떨어진다(`docs/DEPTH.md` 참고).
- 층별 조건문·하드코딩 — 다른 안내도로의 일반화가 이 프로젝트의 최우선 목표다.
