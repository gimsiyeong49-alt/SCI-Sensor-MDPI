#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localizer_node.py  —  GT-free 안내도 초기 위치추정 (배포 노드형)
================================================================
지금까지 개발한 알고리즘을 하나로 합친 실행 가능한 노드.

무엇을 하나:
  시각 t 를 주면 그 순간 로봇의 (LiDAR 10초창 + odom + 깊이)로
  안내도(2D 평면도) 위 top-K 위치 후보 + 확신도를 낸다. GT 미사용.

알고리즘 (한 개의 가중 log-likelihood 합, 케이스별 하드코딩 아님):
  post = LiDAR벽정합(soft)              # 스캔이 벽에 맞나
       + log(scene_prior)               # 깊이가 본 장면종류(분기/끝/복도) 지도칸에 가중
       + 0.8 * traversability           # 로봇 실제 움직임이 그 자리서 벽 안뚫나
       + free-space                     # 관측한 열린정도보다 좁은 칸 벌점(좁은구석 방지)
  여러 프레임 causal 누적. 방향은 odom축→지도복도 정렬 8방향 자동탐색.

확신도:
  top-1 과 top-2 점수 마진으로 CONFIDENT / AMBIGUOUS 판정.
  AMBIGUOUS 면 "이 종류의 장소들 중 하나"로 top-K 만 내고 우기지 않음.

실시간 노드로 쓰는 법 (맨 아래 ROS2 스켈레톤):
  LiDAR/odom/depth 토픽 구독 -> Localizer.localize_live(...) -> top-K 를 MarkerArray publish.
  본 파일 단독 실행은 녹화 데이터(로즈백 추출본)로 돌려 검증/시각화.

실행:
  python3 localizer_node.py            # 5개 랜드마크 시각 다 돌려서 표+그림
  python3 localizer_node.py 214        # 특정 시각(초, 데이터 시작 기준)만
================================================================
"""
import sys, csv, sqlite3, struct, json, subprocess, os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, map_coordinates

# ---------- 경로/상수 ----------
DATA   = '/home/user/HANDOFF_4F_CONTINUE/data_a131'
MAPPNG = '/home/user/HANDOFF_4F_CONTINUE/maps_4F/4F_map_v2.png'
from rosbag_paths import usb_bag
BAG    = usb_bag(4)
CORRIDOR_M = 94.6          # 지도 최장 복도 실측 길이(m) -> px/m 스케일
GRID_STEP  = 22            # 후보격자 간격(px)
SEQ_HORIZON = 18.0         # 순차 belief 전파 관측시간(초). 실시간에선 계속 누적
TRAV_W, FREE_W, PRIOR_BOOST = 0.8, 2.0, 3.0
# --- 장면 분류원 (2026-09-28) ---------------------------------------------------
# 'depth'(기본) = 깊이 카메라, 'lidar' = LiDAR 만으로 같은 라벨 생성(깊이 카메라 불필요).
SCENE_SRC = os.environ.get('SEMLOC_SCENE', 'depth')
SCENE_FWD_DEG = float(os.environ.get('SCENE_FWD_DEG', 7.0))       # 정면 부채꼴 반각
SCENE_SIDE_LO = float(os.environ.get('SCENE_SIDE_LO', 17.5))      # 옆 부채꼴 시작
SCENE_SIDE_HI = float(os.environ.get('SCENE_SIDE_HI', 35.0))      # 옆 부채꼴 끝
# 옆 거리를 이 안쪽으로 제한한다. 기본 = 제한 없음.
# 9m(far 배제)가 깊이 라벨을 더 잘 흉내내지만(일치율 82.3% vs 79%), 실제 성적은 더 나빴다:
# 4F 실시간 위치추정 9m 1.62m/82% vs 12m 이상 0.62m/88%(깊이 0.62m/91%와 동등).
# 12·13·14·16·20·무제한이 모두 같은 값이라 12m 가 특별한 게 아니라 '제한하지 않는 것'이 옳다.
# 라벨 흉내가 목표가 아니라는 실측 근거다(2026-09-28).
SCENE_SIDE_NEAR_M = float(os.environ.get('SCENE_SIDE_NEAR_M', 1e9))
# --- VLM 소프트 semantic (선택) : 검증된 설정 = 좌우 cue만·λ0.1·단일프레임 ---
LAMBDA_V    = 0.1
VLM_PY      = '/data/kci_work/vlm_env/bin/python'      # Qwen용 별도 env
VLM_ADAPTER = '/data/kci_work/vlm_adapter.py'
FACILITY    = '/home/user/HANDOFF_4F_CONTINUE/maps_4F/map_facility.npz'        # 안내도서 뽑은 계단/엘베 px
ROOMS_NPY   = '/home/user/HANDOFF_4F_CONTINUE/maps_4F/rooms.npy'               # 안내도서 뽑은 방 px
# PHI: odom 주축을 지도 복도방향에 맞추는 yaw offset. ★상수로 주지 않음★ —
#   부트스트랩서 벽정합 yaw 탐색으로 GT-free 복구(_recover_directions). 아래는 폴백 초기값.
PHI_FALLBACK = (228, 48)


class FloorplanLocalizer:
    def __init__(self, map_png=MAPPNG, data_dir=DATA, bag=BAG, verbose=True, use_vlm=False,
                 floor=None, ppm=None, recover_directions=True):
        """floor: 1~9 를 주면 그 층 무축척 안내도를 연다(map_png 무시).
        ppm : 주면 스케일을 그 값으로 고정한다. **CORRIDOR_M(외부 실측값) 을 쓰지 않는다.**
              None 이면 기존 동작(CORRIDOR_M 기반) — 4층 호환용.
        방향(PHI·flip)은 항상 _recover_directions 로 층마다 찾는다. 상수 금지."""
        if floor is not None:
            # 지도 뿌리는 환경변수로 바꿀 수 있다(다른 건물). 기본값은 정보관 그대로.
            map_png = f"{os.environ.get('SEMLOC_MAPS_ROOT', '/home/user/kci_maps_perfloor')}/{floor}F/maps/base_map.png"
        self.floor = floor
        self._ppm_override = ppm
        self.v = verbose; self.use_vlm = use_vlm
        im = np.array(Image.open(map_png).convert('L'))
        self.im = im; self.H, self.W = im.shape
        self.wall = im < 50; self.free = im > 200
        self.DFpx  = distance_transform_edt(~self.wall)   # 벽까지 거리(벽정합)
        self.freeD = distance_transform_edt(self.free)    # 열린정도(free-space)
        self._navigable_region()                          # 건물내부 자동검출(현재 참고용; 탐색 bbox는 아래 상수)
        self._find_scale()
        self._load_odom(data_dir)
        # ★깊이 없는 모드: 전달본(a1_3_1 데이터셋)에는 로스백이 없다. depth 는 depth_scene
        #   (PRIOR 선택) 에만 쓰이고, PRIOR≡1 중립화 실험에서 결과 불변(0/5→0/5)이 확인됐다.
        # A complete dataset scene cache remains usable when its USB bag is
        # disconnected. Do not silently replace cached depth with 'corridor'.
        depth_cache = os.path.join(data_dir, 'scene_cache_v1.npy')
        self.no_depth = (os.environ.get('SEMLOC_NO_DEPTH') == '1' or
                         not ((bag and os.path.exists(bag)) or os.path.exists(depth_cache)))
        if not self.no_depth:
            self._depth_bag = bag
            self._build_scene_cache(data_dir)
        self._build_grid_and_priors()
        self.FAR = np.load(f'{data_dir}/far_points.npz')
        self.dir = data_dir
        if recover_directions:
            self._recover_directions()
        self.ctx = None
        if self.use_vlm: self._build_semantic_ctx()   # VLM 소프트 semantic 준비
        # 채점 전용(옵션): 알고리즘엔 절대 안 씀
        try:
            g = np.load('/home/user/HANDOFF_4F_CONTINUE/maps_4F/GT_building.npz')
            self.gt = dict(axm=g['axm'], perp=g['perp_ax'], ctr=g['ctr'], west=float(g['west']))
        except Exception:
            self.gt = None
        # GT 재정합 (채점 기준만 보정. 알고리즘 무관)
        self._gtr = None
        if os.environ.get('GT_REALIGN', '1') != '0':
            try:
                self._gtr = json.load(open('/home/user/semloc_ws/gt_realign.json'))
                if self.v:
                    print(f"[GT 재정합 적용] dθ={np.degrees(self._gtr['dth']):+.2f}° "
                          f"축={self._gtr['da']:+.2f}m 수직={self._gtr['dp']:+.2f}m "
                          f"(벽위 {self._gtr['before']['onwall']*100:.1f}%→{self._gtr['after']['onwall']*100:.1f}%)")
            except Exception:
                pass

    # ---- ★건물 내부(벽으로 둘러싸인) 자유공간 자동검출 -> 탐색 bbox (하드코딩 제거) ----
    #   이미지 테두리에 연결된 자유공간 = 건물 바깥/여백. 그걸 뺀 게 로봇이 있을 수 있는 내부.
    #   후보를 여기로 제한하는 건 '지도에서 아는 것'(건물 안)이라 정당. GT 아님.
    def _navigable_region(self):
        from scipy.ndimage import label
        lab, _ = label(self.free)
        edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        border = set(np.unique(edge)); border.discard(0)
        self.inside = self.free & ~np.isin(lab, list(border))
        ys, xs = np.where(self.inside)
        # bbox + 소폭 여유, GRID_STEP 정렬
        self.gx0, self.gx1 = int(xs.min()) - 5, int(xs.max()) + 5
        self.gy0, self.gy1 = int(ys.min()) - 5, int(ys.max()) + 5

    # ---- 지도 스케일: 최장 가로 복도폭으로 px/m ----
    def _find_scale(self):
        free, wall, H, W = self.free, self.wall, self.H, self.W
        best = (0, 0, 0, 0)
        for r in range(5, H - 5):
            idx = np.where(free[r])[0]
            if len(idx) < 50: continue
            for s in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1):
                a, b = s[0], s[-1]
                if a > 2 and b < W - 3 and wall[r, a - 1] and wall[r, b + 1] and (b - a) > best[0]:
                    best = (b - a, a, b, r)
        _, self.x0, self.x1, self.rc = best
        if getattr(self, '_ppm_override', None) is not None:
            self.PPM = float(self._ppm_override)      # 외부 실측값(CORRIDOR_M) 미사용
        else:
            self.PPM = (self.x1 - self.x0) / CORRIDOR_M

    def _load_odom(self, d):
        rows = list(csv.DictReader(open(f'{d}/odom_link.csv')))
        self.T   = np.array([float(r['t_sec']) for r in rows])
        self.X   = np.array([float(r['fastlio_x']) for r in rows])
        self.Y   = np.array([float(r['fastlio_y']) for r in rows])
        self.YAW = np.array([float(r['fastlio_yaw_rad']) for r in rows])
        self.SID = [r['sample_id'] for r in rows]
        self.ctr = np.c_[self.X, self.Y].mean(0)
        P = np.c_[self.X, self.Y] - self.ctr
        _, ev = np.linalg.eigh(P.T @ P)
        self.AX0 = ev[:, -1]   # odom 주축(복도축, GT-free). 실시간 배포시 최근창 PCA로 대체.

    def _open_depth(self, bag):
        self.con = sqlite3.connect(f"file:{bag}?mode=ro&immutable=1", uri=True)
        tp = {n: (i, t) for i, n, t in self.con.execute("SELECT id,name,type FROM topics")}
        from rosidl_runtime_py.utilities import get_message
        did, dt = tp['/camera/aligned_depth_to_color/image_raw']; self.DI = get_message(dt)
        ts, rid = [], []
        for r, hdr in self.con.execute(f"SELECT rowid,substr(data,1,16) FROM messages WHERE topic_id={did}"):
            b = bytes(hdr); e = '<' if b[1] == 1 else '>'
            ts.append(struct.unpack_from(e + 'i', b, 4)[0] + struct.unpack_from(e + 'I', b, 8)[0] * 1e-9); rid.append(r)
        self.dts = np.array(ts); self.drid = np.array(rid); self.dt0 = self.dts.min()

    def _build_scene_cache(self, data_dir):
        """장면 라벨을 1초 격자로 한 번 계산해 데이터셋 폴더에 캐시한다."""
        import numpy as _np
        p = os.path.join(data_dir, 'scene_cache_v1.npy')
        n = int(self.T[-1] - self.T[0]) + 2
        if os.path.exists(p):
            c = list(_np.load(p, allow_pickle=True))
            if len(c) >= n: self._scn_cache = c; return
        self._scn_cache = None
        self._open_depth(self._depth_bag)
        c = [self.depth_scene(float(k)) for k in range(n)]
        try: _np.save(p, _np.array(c, dtype=object), allow_pickle=True)
        except Exception: pass
        self._scn_cache = c

    # ---- 깊이 한 프레임의 앞/좌/우 거리 ----
    def _depth_dist(self, tsec):
        if not hasattr(self, 'con'):
            self._open_depth(self._depth_bag)
        from rclpy.serialization import deserialize_message
        k = np.argmin(np.abs(self.dts - (self.dt0 + tsec)))
        blob = self.con.execute(f"SELECT data FROM messages WHERE rowid={int(self.drid[k])}").fetchone()[0]
        mm = deserialize_message(bytes(blob), self.DI)
        dep = np.frombuffer(mm.data, np.uint16).reshape(mm.height, mm.width).astype(float) / 1000.0
        dep[dep == 0] = np.nan; Hh, Wd = dep.shape; mid = dep[Hh // 3:2 * Hh // 3]
        fwd = np.nanmedian(mid[:, 2 * Wd // 5:3 * Wd // 5])
        L = np.nanmedian(mid[:, :Wd // 4]); Rr = np.nanmedian(mid[:, 3 * Wd // 4:])
        return fwd, L, Rr

    # ---- LiDAR 장면분류 (깊이 카메라 대체) ----
    def _lidar_dist(self, t):
        """LiDAR 점군에서 깊이 카메라와 **같은 각도 구간**의 거리를 잰다.

        _depth_dist 는 영상 가운데 1/5 를 정면, 좌우 1/4 을 옆으로 쓴다.
        HFOV 를 70도로 보면 정면 ±7도, 옆은 ±17.5~35도다. 같은 구간을 재면
        임계값(4.5 / 7.5 / 12 m)을 그대로 쓸 수 있다.
        옆은 **근거리(≤9m)만** 쓴다. far(9~18m) 점을 섞으면 옆이 늘 넓게 나와
        복도가 교차로로 뒤집힌다(정보관 3F 일치율 74.5% → 85.4%, 2026-09-28 실측).
        """
        pts, _ = self._scan(t)
        if len(pts) < 20:
            return np.nan, np.nan, np.nan
        r = np.hypot(pts[:, 0], pts[:, 1])
        ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        near = r <= SCENE_SIDE_NEAR_M

        def med(m):
            return float(np.median(r[m])) if m.sum() >= 5 else np.nan
        f = med(np.abs(ang) <= SCENE_FWD_DEG)
        l = med((ang >= SCENE_SIDE_LO) & (ang <= SCENE_SIDE_HI) & near)
        rr = med((ang <= -SCENE_SIDE_LO) & (ang >= -SCENE_SIDE_HI) & near)
        return f, l, rr

    def lidar_scene(self, tsec):
        """깊이 카메라 없이 LiDAR 만으로 같은 장면 라벨을 만든다.

        깊이 라벨 재현율은 옆 거리를 9m 로 제한할 때 가장 높지만(평균 82.3%), **성적은 제한 없을 때가 낫다**
        (4F 실시간 위치추정 9m 1.62m/82% vs 무제한 0.62m/88%, 깊이 0.62m/91%). 라벨 흉내가 목표가 아니다.
        어긋나는 쪽은 대부분 '깊이는 복도인데 LiDAR 는 교차로' 다 — 360도라 좁은 화각이 못 보는 옆 개구부를 본다.
        """
        samp = [self._lidar_dist(self.T[0] + max(0.0, tsec - dts)) for dts in (0.0, 0.5, 1.0, 1.5)]
        fwd = float(np.nanmedian([f for f, _, _ in samp])) if any(np.isfinite(f) for f, _, _ in samp) else np.nan
        if not np.isfinite(fwd):
            return 'corridor'
        def nm(v):
            v = [x for x in v if np.isfinite(x)]
            return float(np.median(v)) if v else 0.0
        L = nm([l for _, l, _ in samp]); Rr = nm([r for _, _, r in samp])
        wide = max(L, Rr)
        if fwd < 4.5: return 'dead-end'
        if (fwd > 12 and wide < 7.5) or wide < 4.5: return 'corridor'
        return 'junction'

    # ---- 깊이 자율 장면분류 (최근 여러 프레임 중앙값 = 순간 튐 무시) ----
    def depth_scene(self, tsec):
        # SEMLOC_SCENE=lidar → 깊이 카메라를 쓰지 않고 LiDAR 로 장면을 만든다(2026-09-28 사용자 결정).
        if SCENE_SRC == 'lidar':
            c = getattr(self, '_lscn_cache', None)
            k = int(round(max(0.0, tsec)))
            if c is None:
                c = self._lscn_cache = {}
            if k not in c:
                c[k] = self.lidar_scene(float(k))
            return c[k]
        if getattr(self, 'no_depth', False): return 'corridor'   # 깊이 없음 → 고정 장면
        # ★장면 캐시: USB 백에서 깊이를 매 호출 읽으면 시점당 84회 디스크 접근이 된다.
        #   1초 격자로 한 번만 계산해 데이터셋 옆에 저장하고 이후엔 조회만 한다.
        c = getattr(self, '_scn_cache', None)
        if c is not None:
            k = int(round(max(0.0, tsec)))
            if 0 <= k < len(c): return c[k]
        # 과거 ~1.5초(0.5초 간격) 프레임들의 거리 중앙값으로 분류 -> 한 프레임 오검 방지
        # (2초로 늘리면 인접 전이구간이 섞여 LINC 오분류 -> 1.5초가 스윗스팟)
        samp = [self._depth_dist(max(0.0, tsec - dts)) for dts in (0.0, 0.5, 1.0, 1.5)]
        fwd = float(np.nanmedian([f for f, _, _ in samp]))
        L = float(np.nanmedian([l for _, l, _ in samp])); Rr = float(np.nanmedian([r for _, _, r in samp]))
        wide = max(L, Rr)
        if fwd < 4.5: return 'dead-end'
        if (fwd > 12 and wide < 7.5) or wide < 4.5: return 'corridor'
        return 'junction'

    # ---- 후보격자 + 지도 종류 prior ----
    def _build_grid_and_priors(self):
        H, W, free = self.H, self.W, self.free
        # ★후보격자 범위: 4층 전용 하드코딩(950~3300, 420~1460) 이었다. 다른 층은 해상도가
        #   달라 그대로 쓰면 지도 밖을 가리킨다. _navigable_region 이 구한 건물내부 bbox 를 쓴다.
        gx0,gx1,gy0,gy1 = self.gx0, self.gx1, self.gy0, self.gy1
        if os.environ.get('SEMLOC_OLDGRID') == '1':   # 진단용: 옛 4층 전용 격자로 복귀
            self.GX = np.arange(950, 3300, GRID_STEP); self.GY = np.arange(420, 1460, GRID_STEP)
        else:
            self.GX = np.arange(max(0,gx0), min(W,gx1)+1, GRID_STEP)
            self.GY = np.arange(max(0,gy0), min(H,gy1)+1, GRID_STEP)
        self.GXX, self.GYY = np.meshgrid(self.GX, self.GY, indexing='ij')
        self.Pf = np.c_[self.GXX.ravel(), self.GYY.ravel()]; self.shp = self.GXX.shape
        rng = np.arange(0, int(20 * self.PPM), 6)
        AN = np.linspace(0, 2 * np.pi, 24, endpoint=False); CO, SI = np.cos(AN), np.sin(AN)
        def fdir(px, py):
            fd = []
            for c, s in zip(CO, SI):
                xs = (px + rng * c).astype(int); ys = (py + rng * s).astype(int)
                ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                fr = free[np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)] & ok
                d = np.argmax(~fr) if (~fr).any() else len(rng); fd.append(rng[min(d, len(rng) - 1)] / self.PPM)
            return np.array(fd)
        JU = np.ones(self.shp); DE = np.ones(self.shp); CObp = np.ones(self.shp)
        for ii in range(self.shp[0]):
            for jj in range(self.shp[1]):
                px, py = self.Pf[ii * self.shp[1] + jj]
                if not free[int(py), int(px)]: continue
                fd = fdir(px, py); nfar = (fd > 8).sum(); d0 = int(np.argmax(fd)); opp = fd[(d0 + 12) % 24]
                if nfar >= 6: JU[ii, jj] = PRIOR_BOOST
                if fd[d0] > 10 and opp < 4 and nfar <= 3: DE[ii, jj] = PRIOR_BOOST
                if nfar <= 4 and opp > 7: CObp[ii, jj] = PRIOR_BOOST
        self.PRIOR = {'junction': JU, 'dead-end': DE, 'corridor': CObp}
        # ★층 간 비교 가능하게 정규화. PRIOR 는 '그 층 안에서 어디가 그럴듯한가'를 나타내는
        #   위치 사전확률이지 층의 우열이 아니다. 그런데 부스트(3.0) 칸 비율이 층마다
        #   2.4%(2F)~10.6%(9F) 로 달라, 정규화 없이 log 를 더하면 21프레임에 최대 +23 의
        #   층별 오프셋이 생긴다. 실측: 1F 벽정합 -11.03 이 9F -19.63 에 지는데, 장면사전이
        #   9.53 대 23.07 이라 뒤집혔다(4개 사례 전부 벽정합 좋은 쪽이 패배).
        #   자유공간 평균으로 나누면 층별 오프셋이 사라지고 층 내 상대적 선호만 남는다.
        if os.environ.get('SEMLOC_RAW_PRIOR') != '1':
            fg = free[np.clip(self.GYY.astype(int), 0, H - 1), np.clip(self.GXX.astype(int), 0, W - 1)]
            for k, v in self.PRIOR.items():
                m = float(v[fg].mean()) if fg.any() else 1.0
                self.PRIOR[k] = v / max(m, 1e-9)
        self.freeD_grid = self.freeD[np.clip(self.GYY.astype(int), 0, H - 1),
                                     np.clip(self.GXX.astype(int), 0, W - 1)] / self.PPM

    # ---- ★ 방향(PHI/yaw offset) GT-free 부트스트랩 복구 ----
    #   원리(GPT 감사): 지도 위 로봇 yaw는 '주어지는 상수'가 아니라 찾아야 하는 상태.
    #   여러 시각의 rich LiDAR를 지도에 여러 yaw로 돌려보며(0~360° 탐색) 벽에 가장 잘 맞는
    #   yaw를 고른다. 이게 곧 candidate=(x,y,yaw)에서 yaw를 관측으로 정하는 것.  GT 미사용.
    def _recover_directions(self, n_boot=3, coarse_step=45, phi_step=12, dur=10.0, ns=6):
        # 부트스트랩 앵커: 경로 전체서 고르게. 각 앵커는 10초 odom-일관 윈도우로 평가.
        idx = np.linspace(int(len(self.T) * 0.15), int(len(self.T) * 0.8), n_boot).astype(int)
        bts = [self.T[i] for i in idx]
        GXc = np.arange(max(0,self.gx0), min(self.W,self.gx1)+1, coarse_step)
        GYc = np.arange(max(0,self.gy0), min(self.H,self.gy1)+1, coarse_step)
        Pc = np.c_[np.meshgrid(GXc, GYc, indexing='ij')[0].ravel(),
                   np.meshgrid(GXc, GYc, indexing='ij')[1].ravel()]
        def fit(t0, a, pp, PHIdeg):
            # ★odom-일관 윈도우 벽정합: 스캔을 후보위치에 얹되, 윈도우 내 프레임을 odom대로
            #   이동시키며 계속 벽에 맞는지 누적. 틀린 PHI는 odom 이동 후 벽서 벗어남 -> 저점.
            AX, PERP, ang = self._mk(a, pp)
            b0 = self._b(np.c_[self.X, self.Y][np.argmin(np.abs(self.T - t0))], AX, PERP)
            logb = np.zeros(len(Pc))
            for dt in np.linspace(0, dur, ns):
                pts, yaw = self._scan(t0 + dt)
                if len(pts) > 150: pts = pts[np.random.RandomState(int(t0 + dt) % 97).choice(len(pts), 150, replace=False)]
                db = self._b(np.c_[self.X, self.Y][np.argmin(np.abs(self.T - (t0 + dt)))], AX, PERP) - b0
                robx = Pc[:, 0] + db[0] * self.PPM; roby = Pc[:, 1] - db[1] * self.PPM
                hd = (yaw - ang) + np.deg2rad(PHIdeg); pyl = pts[:, 1]; c, s = np.cos(hd), np.sin(hd)
                ex = robx[:, None] + ((pts[:, 0] * c - pyl * s) * self.PPM)[None]
                ey = roby[:, None] + (-(pts[:, 0] * s + pyl * c) * self.PPM)[None]
                ci = np.clip(np.round(ex).astype(int), 0, self.W - 1); ri = np.clip(np.round(ey).astype(int), 0, self.H - 1)
                dd = np.clip(self.DFpx[ri, ci], 0, 10)
                logb += -(np.sort(dd, 1)[:, :int(ex.shape[1] * 0.7)].mean(1) / 5) ** 2
            return float(logb.max())
        # 1) coarse: flip x PHI(0~360) 전체탐색, 부트스트랩 앵커 합산 벽정합
        best = (-9e9, 1, 1, 0.0)
        for a in (1, -1):
            for pp in (1, -1):
                for PHI in np.arange(0, 360, phi_step):
                    sc = sum(fit(t, a, pp, PHI) for t in bts)
                    if sc > best[0]: best = (sc, a, pp, float(PHI))
        _, a0, pp0, phi0 = best
        # 2) fine: 복구값 ±phi_step, 2° 스텝
        bp = (-9e9, phi0)
        for PHI in phi0 + np.arange(-phi_step, phi_step + 1, 2):
            sc = sum(fit(t, a0, pp0, PHI % 360) for t in bts)
            if sc > bp[0]: bp = (sc, PHI % 360)
        phi0 = bp[1]
        # 3) 미러(pp) 1비트: 벽정합 near-tie -> 궤적 free-space 점유율로 tiebreak (GT-free)
        def occ(a, pp, PHI):
            AX, PERP, ang = self._mk(a, pp)
            i0 = np.argmin(np.abs(self.T - bts[0])); b_a = self._b(np.c_[self.X, self.Y][i0], AX, PERP)
            # 앵커=최고 벽정합 위치
            AXp, PERPp, angp = AX, PERP, ang; pts, yaw = self._scan(bts[0])
            if len(pts) > 150: pts = pts[np.random.RandomState(int(bts[0]) % 97).choice(len(pts), 150, replace=False)]
            hd = (yaw - angp) + np.deg2rad(PHI); pyl = pts[:, 1]; cc, ss = np.cos(hd), np.sin(hd)
            ex = Pc[:, 0][:, None] + ((pts[:, 0] * cc - pyl * ss) * self.PPM)[None]
            ey = Pc[:, 1][:, None] + (-(pts[:, 0] * ss + pyl * cc) * self.PPM)[None]
            ci = np.clip(np.round(ex).astype(int), 0, self.W - 1); ri = np.clip(np.round(ey).astype(int), 0, self.H - 1)
            dd = np.clip(self.DFpx[ri, ci], 0, 10)
            lock = Pc[np.argmax(-(np.sort(dd, 1)[:, :int(ex.shape[1] * 0.7)].mean(1) / 5) ** 2)]
            infr = []
            for t in self.T[::max(1, len(self.T) // 120)]:
                i = np.argmin(np.abs(self.T - t)); db = self._b(np.c_[self.X, self.Y][i], AX, PERP) - b_a
                px = int(np.clip(lock[0] + db[0] * self.PPM, 0, self.W - 1)); py = int(np.clip(lock[1] - db[1] * self.PPM, 0, self.H - 1))
                infr.append(0 if self.wall[py, px] else 1)
            return np.mean(infr)
        o_pos, o_neg = occ(a0, pp0, phi0), occ(a0, -pp0, phi0)
        if o_neg > o_pos: pp0 = -pp0                     # 궤적이 더 free-space에 있는 미러 채택
        # 결과: yaw offset 2개(정/역 180°), 복구 flip(a0,pp0)
        self._a0, self._pp0 = a0, pp0
        self.PHI_SET = (round(phi0), round((phi0 + 180) % 360))
        self._boot_mirror_margin = float(abs(o_pos - o_neg))
        if self.v:
            print(f"[방향 GT-free 복구] flip 축{a0:+d} perp{pp0:+d}, PHI={self.PHI_SET} "
                  f"(부트스트랩 {n_boot}프레임 벽정합, 미러 free-space 마진 {self._boot_mirror_margin:.2f})")

    # ---- ★ VLM 소프트 semantic (검증설정: 좌우 cue·λ0.1·단일프레임) ----
    def _build_semantic_ctx(self):
        import semantic_likelihood as SL
        self.SL = SL
        # 검증설정: 좌우(문/방 인접)만. forward=노이즈, object=inert, space=depth중복 → 끔.
        for f in ('forward', 'object', 'space'): SL.CUE[f]['on'] = False
        SL.CUE['left']['on'] = True; SL.CUE['right']['on'] = True
        fac = np.load(FACILITY); rooms = np.load(ROOMS_NPY)
        self.ctx = SL.MapSemanticContext(self.wall, self.free, self.PPM, self.GX, self.GY,
                                         stairs_px=fac['stairs'], elev_px=fac['elev'], rooms_px=rooms)
        if self.v: print(f"[VLM] semantic context 준비 (좌우 cue, λ={LAMBDA_V}, rooms={len(rooms)})")

    def _vlm_observe(self, ts_rel):
        """라이브: rosbag RGB(ts_rel)에 Qwen yes/no 질의 -> VisualObservation. vlm_env 서브프로세스."""
        outf = f'/tmp/vlm_obs_{int(ts_rel * 1000)}.json'
        try:
            subprocess.run([VLM_PY, VLM_ADAPTER, '--ts', str(float(ts_rel)), '--out', outf],
                           check=True, capture_output=True, timeout=400)
            d = json.load(open(outf))
            return d.get(str(float(ts_rel))) or next(iter(d.values()))
        except Exception as ex:
            if self.v: print(f"[VLM] 관측 실패 -> 기하만 사용 ({str(ex)[:80]})")
            return None

    def _semantic_ll(self, t0, ORI, obs):
        """관측 obs -> 방향가설별 semantic log-lik 격자. heading=검증규약 -((yaw-ang)+PHI)."""
        if obs is None or self.ctx is None: return None
        yaw0 = self.YAW[np.argmin(np.abs(self.T - t0))]; out = {}
        for o in ORI:
            a, pp, PHI = o; _, _, ang = self._mk(a, pp)
            heading = -((yaw0 - ang) + np.deg2rad(PHI))
            out[o] = self.SL.score_frame(self.ctx, heading, obs).reshape(self.shp)
        return out

    # ---- LiDAR 로드(밀집점) ----
    def _scan(self, t, include_far=True):
        i = np.argmin(np.abs(self.T - t)); sid = self.SID[i]
        d = np.load(f'{self.dir}/samples/{sid}/data_short.npz')
        a = d['a1_3_1_all_candidate_xy'].astype(float); role = d['a1_3_1_all_candidate_role']; p = a[role != 4]
        # ★far(9~18m) 합류. 목적은 '열린장면에서 near(≤9m) 만으로 제약이 없을 때 원거리 벽 제공'.
        #   다만 실측상 far 가 near 의 2~3배(비중 45~72%)이고 프레임마다 0~3000 으로 출렁여
        #   정합 성격이 창마다 달라진다. SEMLOC_NO_FAR=1 / SEMLOC_FAR_CAP=<n> 으로 조절.
        if include_far and os.environ.get('SEMLOC_NO_FAR') != '1' and sid in self.FAR.files and len(self.FAR[sid]):
            fp = self.FAR[sid].astype(float)
            cap = int(os.environ.get('SEMLOC_FAR_CAP', '0'))
            if cap and len(fp) > cap:
                fp = fp[np.random.RandomState(i % 97).choice(len(fp), cap, replace=False)]
            p = np.vstack([p, fp])
        return p, self.YAW[i]

    def _mk(self, a, pp):
        AX = self.AX0 * a; return AX, np.array([-AX[1], AX[0]]) * pp, np.arctan2(AX[1], AX[0])
    def _b(self, xy, AX, PERP): return np.array([(xy - self.ctr) @ AX, (xy - self.ctr) @ PERP])

    def _smap(self, t, a, pp, PHI):
        AX, PERP, ang = self._mk(a, pp); pts, yaw = self._scan(t)
        if len(pts) > 400: pts = pts[np.random.RandomState(int(t) % 97).choice(len(pts), 400, replace=False)]
        hd = (yaw - ang) + np.deg2rad(PHI); pyl = pts[:, 1]; c, s = np.cos(hd), np.sin(hd)
        ex = self.Pf[:, 0][:, None] + ((pts[:, 0] * c - pyl * s) * self.PPM)[None]
        ey = self.Pf[:, 1][:, None] + (-(pts[:, 0] * s + pyl * c) * self.PPM)[None]
        ci = np.clip(np.round(ex).astype(int), 0, self.W - 1); ri = np.clip(np.round(ey).astype(int), 0, self.H - 1)
        dd = np.clip(self.DFpx[ri, ci], 0, 10)
        return (-(np.sort(dd, 1)[:, :int(ex.shape[1] * 0.7)].mean(1) / 5) ** 2).reshape(self.shp)

    def spatial_submap(self, t0, dur=20.0, ds=0.7, mode='persist', vox=0.10):
        """★보조(옵션): t0~t0+dur 의 a1_3_1 구조점을 FAST-LIO odom(pose)으로 정렬해
        t0 로봇프레임에 누적한 공간기억 submap. 시각화/맵확장용 (belief 랭킹과 별개 —
        E110서 GT rank는 belief가 나아 랭킹엔 belief 유지). mode=raw/vox/persist(>=3프레임 셀).
        반환: Nx2 점(m, t0 로봇프레임). (E110 consolidation_test.py 방식을 노드에 이식)"""
        T, X, Y, YAW = self.T, self.X, self.Y, self.YAW
        win = [i for i in range(len(T)) if t0 <= T[i] <= t0 + dur]
        sel = []; last = -9.0
        for i in win:
            if T[i] - last >= ds: sel.append(i); last = T[i]
        if not sel: return np.empty((0, 2))
        i0 = sel[0]; x0, y0, yw0 = X[i0], Y[i0], YAW[i0]
        c0, s0 = np.cos(-yw0), np.sin(-yw0)
        acc = []; seen = {}
        for k, f in enumerate(sel):
            p, yw = self._scan(T[f])                          # a1_3_1 점 (로봇프레임)
            cf, sf = np.cos(yw), np.sin(yw)
            wx = X[f] + (p[:, 0] * cf - p[:, 1] * sf); wy = Y[f] + (p[:, 0] * sf + p[:, 1] * cf)  # world
            dx, dy = wx - x0, wy - y0
            rx = dx * c0 - dy * s0; ry = dx * s0 + dy * c0    # t0 로봇프레임
            pf = np.c_[rx, ry]; acc.append(pf)
            if mode == 'persist':
                for g in map(tuple, np.floor(pf / vox).astype(int)): seen.setdefault(g, set()).add(k)
        A = np.vstack(acc)
        if mode == 'vox':
            _, idx = np.unique(np.floor(A / vox).astype(int), axis=0, return_index=True); return A[idx]
        if mode == 'persist':
            keep = {g for g, fr in seen.items() if len(fr) >= 3}
            m = np.array([tuple(g) in keep for g in np.floor(A / vox).astype(int)]); return A[m]
        return A  # raw

    def _obs_open(self, t0):
        pts, _ = self._scan(t0); r = np.hypot(pts[:, 0], pts[:, 1])
        return np.median(np.sort(r)[:max(5, len(r) // 5)])

    def _shift(self, field, off):
        """절대프레임 field를 base프레임으로 (odom offset off만큼 당김)."""
        cx = (self.GXX + off[0] - self.GX[0]) / GRID_STEP; cy = (self.GYY + off[1] - self.GY[0]) / GRID_STEP
        return map_coordinates(field, [cx.ravel(), cy.ravel()], order=1, mode='nearest', cval=-5).reshape(self.shp)

    def scan_px(self, t0, P0, ori):
        """추정 pose(P0 px, 방위 ori)에 LiDAR 스캔을 얹어 지도픽셀 좌표로 반환.
        '이 위치면 스캔이 벽에 이렇게 맞는다'를 시각화하는 용도."""
        a, pp, PHI = ori; pts, yaw = self._scan(t0); AX, PERP, ang = self._mk(a, pp)
        hd = (yaw - ang) + np.deg2rad(PHI); pyl = pts[:, 1]; c, s = np.cos(hd), np.sin(hd)
        ex = P0[0] + (pts[:, 0] * c - pyl * s) * self.PPM
        ey = P0[1] - (pts[:, 0] * s + pyl * c) * self.PPM
        # 벽정합 오차(m): 얹은 점들이 지도 벽까지 평균거리 (작을수록 잘맞음)
        dd = np.clip(self.DFpx[np.clip(np.round(ey).astype(int), 0, self.H - 1),
                               np.clip(np.round(ex).astype(int), 0, self.W - 1)], 0, 10)
        fit = float(np.sort(dd)[:int(len(dd) * 0.7)].mean() / self.PPM)
        return ex, ey, fit

    # ================= 핵심 API: 순차 belief 전파 (최종 알고리즘, E57) =================
    def localize(self, t0, K=6, horizon=None, dt=3.0):
        """순차 belief 전파. t0부터 horizon초까지 매 프레임 belief 누적 -> 수렴 top-K.
        실시간 노드: 매 콜백마다 이 누적을 이어가면 됨(horizon=지금까지 관측한 시간).
        반환: (scene, top-K [(score,xy_px,orient)], confidence, trace[(elapsed,top-K)])."""
        T, X, Y, PPM, shp = self.T, self.X, self.Y, self.PPM, self.shp
        GXX, GYY, Pf, free = self.GXX, self.GYY, self.Pf, self.free
        if horizon is None: horizon = SEQ_HORIZON
        scn0 = self.depth_scene(t0 - T[0]); oo = self._obs_open(t0)
        # 장면-적응 free-space: dead-end(막힌 공간)면 끔 -> 참이 좁다고 자기벌점 받는 것 방지
        fw = 0.0 if scn0 == 'dead-end' else FREE_W
        freepen = -fw * np.maximum(0, oo * 0.6 - self.freeD_grid)
        # 방향 가설 = flip(축·미러) × {PHI, PHI+180}.  PHI 값은 부트스트랩 벽정합으로
        #   GT-free 복구한 것(상수 아님).  yaw를 탐색상태로 들고 후보별 증거로 정함
        #   = candidate=(x,y,yaw).  미러/정역 모호성은 가설로 유지, 증거가 붕괴시킴.
        ORI = [(a, pp, PHI) for a in (1, -1) for pp in (1, -1) for PHI in self.PHI_SET]
        # VLM 소프트 semantic: 단일(t0) 프레임 관측 -> 방향가설별 격자 점수 (검증설정)
        sem = self._semantic_ll(t0, ORI, self._vlm_observe(t0 - T[0])) if self.use_vlm else None
        belief = {o: np.zeros(shp) for o in ORI}; b0 = {}
        for o in ORI:
            a, pp, PHI = o; AX, PERP, _ = self._mk(a, pp)
            b0[o] = self._b(np.c_[X, Y][np.argmin(np.abs(T - t0))], AX, PERP)
        frames = t0 + np.arange(0, horizon + 1e-6, dt); frames = frames[frames <= T[-1]]
        trace = []
        for tq in frames:
            scn = self.depth_scene(tq - T[0]); logpri = np.log(self.PRIOR[scn] + 1e-6)
            for o in ORI:                                     # belief 누적 (LiDAR+prior, odom정렬)
                a, pp, PHI = o; AX, PERP, _ = self._mk(a, pp)
                db = self._b(np.c_[X, Y][np.argmin(np.abs(T - tq))], AX, PERP) - b0[o]
                off = np.array([db[0] * PPM, -db[1] * PPM])
                belief[o] += self._shift(self._smap(tq, a, pp, PHI), off) + self._shift(logpri, off)
            top, conf, _ = self._collapse(belief, b0, ORI, t0, tq, freepen, K, sem)  # 현재까지 top-K
            trace.append((float(tq - t0), top))
        top, conf, allc = self._collapse(belief, b0, ORI, t0, frames[-1], freepen, K, sem)
        return scn0, top, conf, trace

    def _collapse(self, belief, b0, ORI, t0, tq_now, freepen, K, sem=None):
        """현재 belief + 누적 궤적 traversability + free-space -> top-K 봉우리."""
        T, X, Y, PPM, shp = self.T, self.X, self.Y, self.PPM, self.shp
        GXX, GYY, Pf, free = self.GXX, self.GYY, self.Pf, self.free
        TR = T[0] + np.arange((t0 - T[0]) - 3, (tq_now - T[0]) + 0.1, 1.5)
        allc = []
        for o in ORI:
            a, pp, PHI = o; AX, PERP, _ = self._mk(a, pp); trav = np.zeros(shp)
            for ttr in TR:
                db = self._b(np.c_[X, Y][np.argmin(np.abs(T - ttr))], AX, PERP) - b0[o]
                off = np.array([db[0] * PPM, -db[1] * PPM])
                pxg = np.clip(np.round(GXX + off[0]).astype(int), 0, self.W - 1)
                pyg = np.clip(np.round(GYY + off[1]).astype(int), 0, self.H - 1)
                trav += np.where(free[pyg, pxg], 0.0, -1.0)
            post = belief[o] + TRAV_W * trav + freepen
            if sem is not None: post = post + LAMBDA_V * sem[o]   # ★ VLM 소프트 융합 (한 줄)
            mx = maximum_filter(post, size=7); pk = (post == mx); pi = np.where(pk.ravel())[0]
            for k in pi[np.argsort(-post.ravel()[pi])[:8]]:
                allc.append((float(post.ravel()[k]), Pf[k], o))
        allc.sort(key=lambda z: -z[0])
        return allc[:K], self._confidence(allc), allc

    def _confidence(self, allc):
        """top-1 vs top-2(다른 위치) 점수 마진 -> CONFIDENT/AMBIGUOUS."""
        if not allc: return dict(level='NONE', margin=0.0)
        s1, p1, _ = allc[0]; s2 = None
        for s, p, _ in allc[1:]:
            if np.hypot(*(p - p1)) / self.PPM > 5:  # 5m 이상 떨어진 다른 봉우리
                s2 = s; break
        if s2 is None: return dict(level='CONFIDENT', margin=99.0)
        margin = (s1 - s2)                          # log점수 차 (0에 가까우면 애매)
        level = 'CONFIDENT' if margin > 1.5 else 'AMBIGUOUS'
        return dict(level=level, margin=float(margin))

    # ---- 채점 전용 좌표(옵션, 알고리즘 미사용) ----
    def gt_px(self, t):
        """채점 전용. 환경변수 GT_REALIGN=0 이면 원본 정합, 1(기본)이면 보정 정합.
        보정값은 gt_realign.json (odom 궤적 + 안내도 자유공간만으로 추정, 알고리즘 무관)."""
        if not self.gt: return None
        p = np.c_[self.X, self.Y][np.argmin(np.abs(self.T - t))]
        axm, perp = self.gt['axm'], self.gt['perp']
        da = dp = 0.0
        if getattr(self, '_gtr', None):
            r = self._gtr; dth = r['dth']; da, dp = r['da'], r['dp']
            c, s_ = np.cos(dth), np.sin(dth)
            axm = np.array([axm[0]*c - axm[1]*s_, axm[0]*s_ + axm[1]*c])
            perp = np.array([-axm[1], axm[0]])
        ba = (p - self.gt['ctr']) @ axm - self.gt['west'] + da
        bp = (p - self.gt['ctr']) @ perp + dp
        return np.array([self.x0 + ba * self.PPM, self.rc - bp * self.PPM])

    def close(self):
        try: self.con.close()
        except Exception: pass


# ===================== 단독 실행: 검증 + 시각화 =====================
def _run_cli():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for fp in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        try: fm.fontManager.addfont(fp); plt.rcParams['font.family'] = fm.FontProperties(fname=fp).get_name(); break
        except Exception: pass
    plt.rcParams['axes.unicode_minus'] = False
    argv = [a for a in sys.argv[1:] if a != '--vlm']
    use_vlm = '--vlm' in sys.argv                          # 이미지 semantic 라이브 융합
    L = FloorplanLocalizer(use_vlm=use_vlm)
    O = '/home/user/semloc_ws/out'
    LM = {'자판기': 37, 'LINC': 54, '게시판': 145, '화분끝': 214, '사물함': 303}
    if argv:
        LM = {f't={argv[0]}s': float(argv[0])}
    def stability(trace):
        """마지막 3프레임 top-1 위치가 서로 <2m로 모여있으면 STABLE (GT 미사용)."""
        if len(trace) < 3: return False
        p1 = [tk[0][1] for _, tk in trace[-3:]]
        return all(np.hypot(*(p1[i] - p1[0])) / L.PPM < 2 for i in range(1, 3))
    n = len(LM); fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 6)); axs = np.atleast_1d(axs)
    print(f"{'지점':>8}{'장면':>10}  경과별 참순위(스트림)                  스트림최고     안정성")
    for ax, (nm, ts) in zip(axs, LM.items()):
        t0 = L.T[0] + ts; scn, top, conf, trace = L.localize(t0); gp = L.gt_px(t0)
        stab = stability(trace); stab_txt = 'STABLE(신뢰)' if stab else 'FLICKER(불안정)'
        best = None  # 스트림서 참이 도달한 최고순위 + 그 순간 (GT는 채점만)
        if gp is not None:
            ranks = [(el, next((j + 1 for j, (s, p, o) in enumerate(tk) if np.hypot(*(p - gp)) / L.PPM < 5), 99), tk)
                     for el, tk in trace]
            bel, brk, btk = min(ranks, key=lambda z: (z[1], z[0]))
            best = (bel, brk, btk)
            tline = " ".join(f"{el:.0f}s:{(rk if rk < 99 else 'X')}" for el, rk, _ in ranks)
            bstr = f"{brk}위@{bel:.0f}s" if brk < 99 else "top6밖"
        else:
            tline = "(GT없음)"; bstr = "-"; best = (trace[-1][0], None, trace[-1][1])
        print(f"{nm:>8}{scn:>10}  {tline}   {bstr:>10}   {stab_txt}")
        # 시각화: 추정 top-1 pose에 LiDAR 스캔을 얹어 "벽에 맞는다"를 보여줌
        drawtop = best[2]; P0 = drawtop[0][1]; ori0 = drawtop[0][2]
        ex, ey, fit = L.scan_px(t0, P0, ori0)
        ax.imshow(L.im, cmap='gray')
        ax.scatter(ex, ey, s=4, c='red', alpha=.5, zorder=3, label='LiDAR 스캔')            # 얹은 스캔
        ax.scatter(*P0, s=230, marker='o', facecolor='none', edgecolor='lime', lw=3, zorder=5, label='추정 위치')
        for i, (s, p, o) in enumerate(drawtop[1:], 2):                                       # 나머지 후보(작게)
            ax.scatter(*p, s=45, c='orange', edgecolor='k', lw=.4, zorder=4)
        if gp is not None:
            ax.scatter(*gp, s=260, marker='*', c='cyan', edgecolor='blue', lw=1.4, zorder=6, label='참(GT)')
        ax.set_xlim(P0[0] - 470, P0[0] + 470); ax.set_ylim(P0[1] + 320, P0[1] - 320)
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc='upper right', fontsize=8, framealpha=.85)
        ax.set_title(f'{nm} [{scn}] {stab_txt}\n추정 스트림최고 {bstr}, 벽정합 {fit:.2f}m', fontsize=11)
    plt.suptitle('localizer_node: 추정 위치(초록)에 LiDAR 스캔(빨강)을 얹으면 지도 벽에 맞음 = 위치추정. ★=GT(채점용)', fontsize=12)
    plt.tight_layout(); out = f'{O}/E62_node.png'; plt.savefig(out, dpi=90)
    print(f'\n그림 저장: {out}'); L.close()


# ===================== 로봇 배포용 ROS2 노드 스켈레톤 =====================
# 실제 로봇에선 아래처럼 감싼다. (본 파일 단독 실행엔 안 쓰임)
#
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2, Image as Img
# from nav_msgs.msg import Odometry
# from visualization_msgs.msg import MarkerArray, Marker
#
# class LocalizerNode(Node):
#     def __init__(self):
#         super().__init__('floorplan_localizer')
#         self.loc = FloorplanLocalizer()          # 지도/prior 1회 로드
#         self.buf = []                            # 최근 (t, scan, odom, depth) 링버퍼 ~20초
#         self.create_subscription(PointCloud2, '/lidar/points', self.on_scan, 10)
#         self.create_subscription(Odometry,    '/odom',        self.on_odom, 10)
#         self.create_subscription(Img,          '/depth',       self.on_depth, 10)
#         self.pub = self.create_publisher(MarkerArray, '/localization/topk', 1)
#         self.create_timer(1.0, self.tick)        # 1Hz 로 위치추정
#     def tick(self):
#         # 최근 버퍼로 scene/scan/odom-track 구성 -> loc.localize_live(...) 호출
#         scn, top, conf = self.loc.localize_live(self.buf)
#         if conf['level'] == 'CONFIDENT':
#             self.publish_markers(top)            # 확신설때만 확정 발표
#         else:
#             self.publish_markers(top, ghost=True) # 애매하면 후보만(반투명)
#     # on_scan/on_odom/on_depth: self.buf 에 최근값 append, 20초 넘으면 pop
#     # localize_live: localize()의 로직을 '버퍼 데이터' 입력으로 바꾼 래퍼
#     #   (파일 로드 대신 콜백 메시지 사용 — 수학은 동일)


if __name__ == '__main__':
    _run_cli()
