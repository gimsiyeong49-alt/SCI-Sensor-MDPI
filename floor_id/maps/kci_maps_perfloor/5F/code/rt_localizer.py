#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rt_localizer.py — 실시간(인과) 위치추정. 미래·전역 정보를 일절 쓰지 않는다.

기존 격자판이 쓰던 비인과 요소와 그 대체:
  AX0 전체궤적 PCA          -> 사용 안 함. PHI-02 canonical(BETA 상수 + 히스토그램 φ)
  _recover_directions       -> PHI-02 causal bootstrap (시동창 누적점군 방향 히스토그램)
  forward_gate 전체궤적      -> 시동창 내부 전진 일관성만
  loo_cal 미래관측 포함       -> 창 시작 이전 관측만
  창 t0..t0+W (미래)         -> t_now-W..t_now (과거) 로 재앵커

유지한 것(정확도 우위 실측분):
  프레임별 벽정합 누적(공간누적 1회 평가보다 dev 8/8 우세) + VLM 의미항 + 후보 허용마스크
공간누적은 '방향 부트스트랩의 입력'으로 쓴다 — 그 자리에서는 대체 불가.
"""
import json, sys, numpy as np
import node_local as NL
from semantic_map import SemanticMap
from semloc import CueCalibration, frame_ll
from scipy.ndimage import binary_opening
from scipy.spatial import cKDTree

BETA = np.deg2rad(228.0)          # 센서 상수(PHI-02 canonical, 검증됨)
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
    def __init__(self, lam_sem=0.5, w_nogo=3.0, verbose=True):
        self.L = L = NL.FloorplanLocalizer(verbose=False)
        self.sm = sm = SemanticMap(L.GX, L.GY, verbose=verbose)
        self.obs = json.load(open('/home/user/semloc_ws/vlm_obs_all.json'))
        self.okeys = np.array(sorted(float(k) for k in self.obs))
        self.lam_sem, self.w_nogo = lam_sem, w_nogo
        self.ok_px = sm.free & sm.footprint
        self.pen_px = (~(sm.free & sm.footprint & ~sm.nogo
                         & ~sm.restroom_area & ~sm.stair_area)) & self.ok_px
        gi = sm.G.astype(int)
        self.allow = self.ok_px[gi[:, 1], gi[:, 0]]
        self.pen_g = self.pen_px[gi[:, 1], gi[:, 0]].reshape(L.shp)
        st = binary_opening(L.wall, iterations=2); ys, xs = np.where(st)
        mp = np.c_[xs, ys].astype(float)
        self.h_map = ori_hist(mp[np.linspace(0, len(mp)-1, 6000).astype(int)])
        self.phis = None

    # ---------- 인과 부트스트랩 (시동 1회) ----------
    def bootstrap(self, t_start, boot_s=15.0, stride=0.7):
        L = self.L
        sel = []; last = -9.0
        for i in range(len(L.T)):
            if t_start <= L.T[i] <= t_start+boot_s and L.T[i]-last >= stride:
                sel.append(i); last = L.T[i]
        if not sel: sel = [int(np.argmin(np.abs(L.T-t_start)))]
        a0 = sel[0]; P = []
        for f in sel:
            p, _ = L._scan(L.T[f])
            if len(p) > 250: p = p[np.linspace(0, len(p)-1, 250).astype(int)]
            d = np.array([L.X[f]-L.X[a0], L.Y[f]-L.Y[a0]])
            P.append(d[None, :] + p @ _R(L.YAW[f]+BETA).T)
        cloud = np.vstack(P)
        # ★A(0)=[[1,0],[0,-1]] 로 y를 뒤집어야 지도(이미지) 좌표계와 같은 방향축이 된다.
        #   이 뒤집기를 빼면 히스토그램이 거울상이 되어 상호상관 φ0 가 통째로 틀린다.
        h0 = ori_hist(L.PPM * (cloud @ _A(0.0).T))
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
                hd = np.deg2rad(phi)+L.YAW[i]+BETA
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
                rs = np.random.RandomState(int(fr.get('t', 0)) % 97)
                pts = pts[rs.choice(len(pts), 400, replace=False)]
            if ob is not None: nsem += 1
            for p in phis:
                off = L.PPM*(_A(np.deg2rad(p)) @ od)
                hd = np.deg2rad(p)+yaw+BETA
                c, s = np.cos(hd), np.sin(hd)
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

    # ---------- 인과 위치추정: t_now 의 pose 를 과거 W초로 ----------
    def localize(self, t_now, W=30.0, dt=3.0, phis=None):
        """파일에서 버퍼를 조립해 localize_buf 로 넘긴다(수학은 공용)."""
        L = self.L
        phis = phis if phis is not None else self.phis
        cal = self.cal_past(t_now-W)
        frames = t_now - np.arange(0, W+1e-6, dt)[::-1]
        frames = frames[frames >= L.T[0]]
        buf = []
        for tq in frames:
            i = int(np.argmin(np.abs(L.T-tq)))
            pts, yaw = L._scan(tq)
            buf.append(dict(pts=pts, xy=(L.X[i], L.Y[i]), yaw=yaw, t=tq,
                            scene=L.depth_scene(tq-L.T[0]),
                            obs=self._ob_near(tq-L.T[0]), cal=cal))
        r = self.localize_buf(buf, phis, L.depth_scene(t_now-L.T[0]), L._obs_open(t_now))
        r['frames'] = frames
        return r

    def fix_phi(self, t_start, boot_s=15.0):
        """시동창에서 4 mode 를 경쟁시켜 φ 를 1회 확정(인과). 건물 방향은 상수이므로
        지점마다 재경쟁시키면 안 된다 — 실측: held 9.26m/2·5 -> 0.74m/4·5."""
        phis, _ = self.bootstrap(t_start, boot_s=boot_s)
        t_b = t_start + boot_s
        sc = {p: self.localize(t_b, W=boot_s, phis=[p])['score'] for p in phis}
        self.phi = max(sc, key=sc.get)
        return self.phi, sc
