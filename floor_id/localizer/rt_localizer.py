#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rt_localizer.py — 실시간(인과) 위치추정. 미래·전역 정보를 일절 쓰지 않는다.

기존 격자판이 쓰던 비인과 요소와 그 대체:
  AX0 전체궤적 PCA          -> 사용 안 함. 이웃 스캔으로 센서/odom 회전 보정
  _recover_directions       -> PHI-02 causal bootstrap (시동창 누적점군 방향 히스토그램)
  forward_gate 전체궤적      -> 시동창 내부 전진 일관성만
  loo_cal 미래관측 포함       -> 창 시작 이전 관측만
  창 t0..t0+W (미래)         -> t_now-W..t_now (과거) 로 재앵커

유지한 것(정확도 우위 실측분):
  프레임별 벽정합 누적(공간누적 1회 평가보다 dev 8/8 우세) + VLM 의미항 + 후보 허용마스크
공간누적은 '방향 부트스트랩의 입력'으로 쓴다 — 그 자리에서는 대체 불가.
"""
import os, json, sys, numpy as np
import node_local as NL
from semantic_map import SemanticMap
from semloc import CueCalibration, frame_ll
from scipy.ndimage import binary_opening
from scipy.spatial import cKDTree
from scan_timing import window_indices

try:                              # 벽정합 GPU 가속(있으면 사용, 없으면 CPU 폴백)
    import wallfit_gpu as _GPU
except Exception:
    _GPU = None

BETA = np.deg2rad(228.0)          # Legacy A/B scripts only; never a runtime default.
NBIN_SEM = 36


def _A(th):
    c, s = np.cos(th), np.sin(th); return np.array([[c, -s], [-s, -c]])


def _R(th):
    c, s = np.cos(th), np.sin(th); return np.array([[c, -s], [s, c]])


def ori_hist(pts, k=15, NB=180, cap=4000, rs=None):
    if len(pts) > cap: pts = pts[np.linspace(0, len(pts)-1, cap).astype(int)]
    if len(pts) < k+1: return np.zeros(NB)
    tree = cKDTree(pts); h = np.zeros(NB)
    for p in pts:
        Q = pts[tree.query(p, k=k)[1]]; Q = Q-Q.mean(0)
        ev, evec = np.linalg.eigh(Q.T@Q); v = evec[:, -1]
        lin = 1-ev[0]/(ev[1]+1e-9)
        if lin < 0.5: continue
        h[int(round(np.degrees(np.arctan2(v[1], v[0])) % 180)) % NB] += lin
    g = np.exp(-0.5*(np.arange(-9, 10)/3.0)**2); g /= g.sum()
    return np.convolve(np.r_[h, h, h], g, 'same')[NB:2*NB]


class RTLocalizer:
    def __init__(self, lam_sem=0.5, w_nogo=3.0, verbose=True, floor=None, ppm=None,
                 data_dir=None, bag=None):
        """floor/ppm 을 주면 그 층 무축척 안내도를 그 스케일로 연다.
        (CORRIDOR_M 등 외부 실측값 미사용 — 스케일은 호출측이 스윕으로 정한다)"""
        # ★주행데이터(odom·samples)와 depth bag 도 층별로 바꿀 수 있어야 한다.
        #   지도(floor)와 데이터(data_dir/bag)는 독립 — 4층 데이터를 9개 층 지도와 겨루거나,
        #   2층 데이터를 9개 층 지도와 겨루거나 모두 가능하다.
        kw = {}
        if data_dir is not None: kw['data_dir'] = data_dir
        if bag is not None:      kw['bag'] = bag
        # RT recovers directions from past observations. The legacy loader's
        # full-trajectory flip search is unrelated and must not run here.
        self.L = L = NL.FloorplanLocalizer(verbose=False, floor=floor, ppm=ppm,
                                          recover_directions=False, **kw)
        if floor is not None:
            import semantic_map as _SM
            sem_root = os.environ.get('SEMLOC_SEMMAPS_ROOT', '/home/user/a131_semmaps')
            map_root = os.environ.get('SEMLOC_MAPS_ROOT', '/home/user/kci_maps_perfloor')
            _SM.SEM=f'{sem_root}/{floor}F/sem'
            _SM.MCL=f'{sem_root}/{floor}F/mcl'
            _SM.LOCMAP=f'{map_root}/{floor}F/maps/base_map.png'
        self.sm = sm = SemanticMap(L.GX, L.GY, verbose=verbose)
        sm.PPM = L.PPM
        # VLM 의미 단서 관측(4F 녹화분). 의미 가중치 lam_sem=0(층 판별)이면 점수에 0 을 곱하므로
        # 읽지도 쓰지도 않는다 — 온보드에는 이 파일이 없다(2026-10-01). lam_sem>0 이면 반드시 있어야 한다.
        obs_p = os.environ.get('SEMLOC_VLM_OBS', '/home/user/semloc_ws/vlm_obs_all.json')
        if os.path.exists(obs_p):
            self.obs = json.load(open(obs_p))
        elif lam_sem:
            raise FileNotFoundError(f'lam_sem={lam_sem} 인데 VLM 관측 파일이 없다: {obs_p}')
        else:
            self.obs = {}
        self.okeys = np.array(sorted(float(k) for k in self.obs))
        self.lam_sem, self.w_nogo = lam_sem, w_nogo
        # ★계단 위에는 로봇이 설 수 없다. 기존에는 allow 에서 안 빼고 w_nogo 벌점만 줘서
        #   후보가 계단 위에 섰다(층별 29~173개). 계단은 물리적 배제가 맞다.
        #   화장실/no_go 는 GT 가 그 위에 찍힌 사례(화분끝)가 있어 벌점 유지.
        self.ok_px = sm.free & sm.footprint
        if os.environ.get('SEMLOC_ALLOW_STAIR') != '1':
            self.ok_px = self.ok_px & ~sm.stair_area
        # ★가짜 벽을 벽거리장에서 뺀다. 두 종류 모두 라이다가 보는 벽면이 아니다.
        #   ① 메워진 면 : 해치가 이진화에서 뭉개져 통째로 칠해진 영역(2F 최대두께 156px,
        #                 실제 벽 8~12px). 근처가 전부 '벽 거리 0' 이 되어 점수를 공짜로 준다.
        #   ② 해치 구조 : 계단·에스컬레이터 기호. 선은 얇으나 촘촘해 같은 효과.
        #   임계가 그 지도 중앙두께의 배수라 층·축척과 무관. 제거율 3F 50%·4F 43%
        #   (판정 86% 를 독점하던 두 층) vs 5F 16% — 밀도 이득의 원천이 가짜 벽이었다.
        if os.environ.get('SEMLOC_KEEP_HATCH') != '1':
            from scipy.ndimage import distance_transform_edt as _dt
            from wall_clean import clean_wall
            real_wall, _fake = clean_wall(L.wall)
            L.wall_real = real_wall
            L.DFpx = _dt(~real_wall)
        self.pen_px = (~(sm.free & sm.footprint & ~sm.nogo
                         & ~sm.restroom_area & ~sm.stair_area)) & self.ok_px
        gi = sm.G.astype(int)
        self.allow = self.ok_px[gi[:, 1], gi[:, 0]]
        self.pen_g = self.pen_px[gi[:, 1], gi[:, 0]].reshape(L.shp)
        st = binary_opening(L.wall, iterations=2); ys, xs = np.where(st)
        mp = np.c_[xs, ys].astype(float)
        self.h_map = ori_hist(mp[np.linspace(0, len(mp)-1, 6000).astype(int)])
        self.BETA = None
        self.MIRROR = 1
        self.beta_quality = None
        self.phis = None

    def fit_beta(self, t_start=None, t_end=None, stride=1.5, fine=2.0):
        """Calibrate only from the supplied past interval (default: startup 15s)."""
        from sensor_frame import estimate_beta
        L = self.L
        t_start = L.T[0] if t_start is None else t_start
        t_end = min(t_start + 15.0, L.T[-1]) if t_end is None else t_end
        indices = []
        for t in np.arange(t_start, t_end + 1e-6, stride):
            i = int(np.searchsorted(L.T, t, side='right') - 1)
            if i >= 0 and L.T[i] >= t_start and (not indices or i != indices[-1]):
                indices.append(i)
        frames = []
        for i in indices:
            pts, yaw = L._scan(L.T[i], include_far=False)
            frames.append(dict(t=L.T[i], pts=pts, xy=np.array([L.X[i], L.Y[i]]), yaw=yaw))
        self.BETA, self.beta_quality = estimate_beta(frames, step_deg=fine)
        self.MIRROR = self.beta_quality['mirror']
        return float(np.degrees(self.BETA)), self.beta_quality['error_m']

    def calib_beta(self, verbose=False, **kwargs):
        """Compatibility entry point; calibration is now map independent."""
        beta, error = self.fit_beta(**kwargs)
        self.bootstrap(self.L.T[0])
        if verbose: print(f'[sensor/odom] beta={beta:.1f} deg, pair residual={error:.3f} m')
        return beta, error

    # ---------- 인과 부트스트랩 (시동 1회) ----------
    def bootstrap(self, t_start, boot_s=15.0, stride=0.7):
        L = self.L
        if self.BETA is None:
            self.fit_beta(t_start, min(t_start + boot_s, L.T[-1]))
        sel = []; last = -9.0
        for i in range(len(L.T)):
            if t_start <= L.T[i] <= t_start+boot_s and L.T[i]-last >= stride:
                sel.append(i); last = L.T[i]
        if not sel: sel = [int(np.argmin(np.abs(L.T-t_start)))]
        frames = []
        for f in sel:
            p, _ = L._scan(L.T[f])
            frames.append(dict(pts=p, xy=np.array([L.X[f], L.Y[f]]), yaw=L.YAW[f], t=L.T[f]))
        return self.bootstrap_buf(frames)

    def bootstrap_buf(self, frames):
        """Shared offline/live bootstrap; only supplied observations are read."""
        if self.BETA is None:
            from sensor_frame import estimate_beta
            self.BETA, self.beta_quality = estimate_beta(frames)
            self.MIRROR = self.beta_quality['mirror']
        P = []
        for fr in frames:
            p = np.asarray(fr['pts'])
            if len(p) > 250: p = p[np.linspace(0, len(p)-1, 250).astype(int)]
            d = np.asarray(fr['xy']) - frames[0]['xy']
            P.append(d[None, :] + (p * [1, self.MIRROR]) @ _R(fr['yaw']+self.BETA).T)
        cloud = np.vstack(P)
        # ★A(0)=[[1,0],[0,-1]] 로 y를 뒤집어야 지도(이미지) 좌표계와 같은 방향축이 된다.
        #   이 뒤집기를 빼면 히스토그램이 거울상이 되어 상호상관 φ0 가 통째로 틀린다.
        h0 = ori_hist(self.L.PPM * (cloud @ _A(0.0).T))
        n = lambda v: (v-v.mean())/(np.linalg.norm(v-v.mean())+1e-9)
        a, b = n(h0), n(self.h_map)
        C = np.array([np.dot(a, np.roll(b, p)) for p in range(180)])
        phi0 = int(np.argmax(C))
        self.phis = [(phi0+d) % 360 for d in (0, 90, 180, 270)]
        return self.phis, cloud

    # ---------- 인과 전진 게이트 (시동창 내부만) ----------
    def gate(self, t_start, boot_s=15.0, seg_s=4.0, min_disp=1.0):
        L = self.L; keep = []
        i0 = int(np.argmin(np.abs(L.T-t_start))); i1 = int(np.argmin(np.abs(L.T-(t_start+boot_s))))
        seg = max(4, int(seg_s/((L.T[-1]-L.T[0])/len(L.T))))
        for phi in self.phis:
            d = []
            for i in range(i0, max(i0, i1-seg), max(1, seg//3)):
                j = i+seg
                od = np.array([L.X[j]-L.X[i], L.Y[j]-L.Y[i]])
                mv = L.PPM*(_A(np.deg2rad(phi)) @ od)
                if np.hypot(*mv)/L.PPM < min_disp: continue
                hd = self.heading(phi, L.YAW[i])
                fwd = np.array([np.cos(hd), -np.sin(hd)])
                d.append(np.abs(np.degrees(np.arccos(np.clip(
                    fwd @ (mv/np.linalg.norm(mv)), -1, 1)))))
            if d and np.median(d) < 90: keep.append(phi)
        return keep if keep else list(self.phis)

    # ---------- 과거 관측만으로 캘리브 ----------
    def cal_past(self, t_cut, mode='raw'):
        rel = t_cut - self.L.T[0]
        ts = [f'{k:g}' for k in self.okeys if k < rel]
        if len(ts) < 6: ts = [f'{k:g}' for k in self.okeys[:6]]
        return CueCalibration(self.obs, ts, mode=mode)

    def _ob_near(self, rel, tol=1.6):
        i = int(np.argmin(np.abs(self.okeys-rel)))
        return self.obs[f'{self.okeys[i]:g}'] if abs(self.okeys[i]-rel) <= tol else None

    # ---------- 버퍼 입력 공용 코어 (오프라인/실시간 동일 경로) ----------
    def heading(self, phi, yaw):
        if self.BETA is None:
            raise ValueError("Calibrate sensor/odom rotation before localization")
        return np.deg2rad(phi) + yaw + self.BETA

    def scan_to_map(self, pts, xy, yaw, anchor, pos, phi, ppm=None, beta=None, mirror=None):
        """The same sensor -> odom -> image transform as localize_buf."""
        ppm = self.L.PPM if ppm is None else ppm
        beta = self.BETA if beta is None else beta
        mirror = self.MIRROR if mirror is None else mirror
        if beta is None: raise ValueError('Sensor/odom rotation has not been calibrated')
        off = ppm * (_A(np.deg2rad(phi)) @ (np.asarray(xy)-anchor))
        hd = np.deg2rad(phi) + yaw + beta
        return np.asarray(pos) + off + ppm * ((np.asarray(pts) * [1, mirror]) @ _A(hd).T)

    def localize_scaled(self, t_now, scales, W=60., dt=3.):
        from metric_fit import fit_window
        return fit_window(self, t_now, scales, W=W, dt=dt)

    def accumulate_buf(self, frames, radius_m=.20, min_observations=3):
        """Return raw/confirmed/unconfirmed odom clouds, without map fitting."""
        from spatial_accumulation import accumulate
        return accumulate(frames, self.BETA, self.MIRROR, radius_m, min_observations)

    def localize_buf(self, buf, phis, scene, open_m):
        """buf: [{'pts':Nx2(m,센서프레임), 'xy':(m,odom), 'yaw':rad, 'obs':dict|None}, ...]
        시간순, 마지막 원소가 추정 대상(앵커). 파일 접근 없음 — ROS2 콜백에서 그대로 쓴다.
        오프라인 localize() 도 이 함수를 호출하므로 온·오프라인 수학이 갈라지지 않는다."""
        L = self.L; sm = self.sm
        anchor = np.asarray(buf[-1]['xy'], float)
        fw = 0.0 if scene == 'dead-end' else NL.FREE_W
        freepen = -fw*np.maximum(0, open_m*0.6 - L.freeD_grid)
        cal = buf[-1].get('cal')
        bel = {p: np.zeros(L.shp) for p in phis}
        sem = {p: np.zeros(L.shp) for p in phis}
        trav = {p: np.zeros(L.shp) for p in phis}
        nsem = 0
        for fr in buf:
            od = np.asarray(fr['xy'], float) - anchor
            pts = np.asarray(fr['pts'], float); yaw = float(fr['yaw'])
            ob = fr.get('obs')
            # ★프레임별 장면사전확률(앵커 것으로 고정하면 게시판 0.93->11.86m 로 무너진다)
            lp = np.log(L.PRIOR[fr.get('scene', scene)] + 1e-6)
            # ★부분추출도 원본과 동일해야 한다(linspace 로 바꾸면 사물함 0.16->31.10m)
            if len(pts) > 400:
                rs = np.random.RandomState(int(fr.get('sample_seed_t', fr.get('t', 0))) % 97)
                pts = pts[rs.choice(len(pts), 400, replace=False)]
            pts = pts * [1, self.MIRROR]
            if ob is not None: nsem += 1
            for p in phis:
                off = L.PPM*(_A(np.deg2rad(p)) @ od)
                hd = self.heading(p, yaw)
                c, s = np.cos(hd), np.sin(hd)
                if _GPU is not None and _GPU.available():
                    g = _GPU.wallfit(L, pts, hd, L.PPM).reshape(L.shp)
                else:
                    ex = L.Pf[:, 0][:, None] + ((pts[:, 0]*c-pts[:, 1]*s)*L.PPM)[None]
                    ey = L.Pf[:, 1][:, None] + (-(pts[:, 0]*s+pts[:, 1]*c)*L.PPM)[None]
                    ci = np.clip(np.round(ex).astype(int), 0, L.W-1)
                    ri = np.clip(np.round(ey).astype(int), 0, L.H-1)
                    dd = np.clip(L.DFpx[ri, ci], 0, 10)
                    g = (-(np.sort(dd, 1)[:, :int(ex.shape[1]*0.7)].mean(1)/5)**2).reshape(L.shp)
                bel[p] += L._shift(g, off) + L._shift(lp, off)
                px = np.clip(np.round(L.GXX+off[0]).astype(int), 0, L.W-1)
                py = np.clip(np.round(L.GYY+off[1]).astype(int), 0, L.H-1)
                trav[p] += np.where(L.free[py, px], 0.0, -1.0)
                if ob is not None and cal is not None:
                    sem[p] += L._shift(frame_ll(sm, cal, ob, -hd).reshape(L.shp), off)
        P = {p: bel[p] + NL.TRAV_W*trav[p] + freepen
             + self.lam_sem*sem[p]/max(nsem, 1) - self.w_nogo*self.pen_g for p in phis}
        best = None
        for p in phis:
            v = np.where(self.allow.reshape(L.shp), P[p], -np.inf)
            k = int(np.argmax(v))
            if best is None or v.ravel()[k] > best[0]:
                best = (float(v.ravel()[k]), L.Pf[k], p)
        return dict(pos=best[1], phi=best[2], score=best[0], scene=scene,
                    nsem=nsem, nphi=len(phis), P=P, anchor=anchor)

    def window_scans(self, t_now, W=30., dt=3.):
        """Associate a scan and its linked pose once, at the actual scan time."""
        L = self.L
        buf = []
        for requested, i in window_indices(L.T, t_now, W, dt):
            t = float(L.T[i])
            pts, yaw = L._scan(t)
            buf.append(dict(pts=pts, xy=(L.X[i], L.Y[i]), yaw=yaw, t=t,
                            sample_seed_t=requested, sample_id=L.SID[i]))
        if not buf:
            raise ValueError('No scans inside the requested past window')
        return buf

    # ---------- 인과 위치추정: 최신 관측 pose 를 과거 W초로 ----------
    def localize(self, t_now, W=30.0, dt=3.0, phis=None):
        """파일에서 버퍼를 조립해 localize_buf 로 넘긴다(수학은 공용)."""
        L = self.L
        phis = phis if phis is not None else self.phis
        use_sem = self.lam_sem != 0          # 0 이면 의미 항이 0 이므로 관측을 붙이지 않는다(결과 동일)
        cal = self.cal_past(t_now-W) if use_sem else None
        buf = self.window_scans(t_now, W, dt)
        for fr in buf:
            fr.update(scene=L.depth_scene(fr['t']-L.T[0]),
                      obs=self._ob_near(fr['t']-L.T[0]) if use_sem else None, cal=cal)
        actual = buf[-1]['t']
        r = self.localize_buf(buf, phis, L.depth_scene(actual-L.T[0]), L._obs_open(actual))
        r['frames'] = np.array([fr['t'] for fr in buf])
        r['frame_times'] = r['frames'].copy()
        r['sampling_times'] = np.array([fr['sample_seed_t'] for fr in buf])
        r['t'] = actual
        r['requested_t'] = float(t_now)
        r['age_s'] = float(t_now-actual)
        return r

    def fix_phi(self, t_start, boot_s=15.0):
        """시동창에서 4 mode 를 경쟁시켜 φ 를 1회 확정(인과). 건물 방향은 상수이므로
        지점마다 재경쟁시키면 안 된다 — 실측: held 9.26m/2·5 -> 0.74m/4·5."""
        phis, _ = self.bootstrap(t_start, boot_s=boot_s)
        t_b = t_start + boot_s
        sc = {p: self.localize(t_b, W=boot_s, phis=[p])['score'] for p in phis}
        self.phi = max(sc, key=sc.get)
        return self.phi, sc
