import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""viz_rt.py — 실시간(인과) 파이프라인 결과 대시보드. result_*.png 와 같은 정보 구성.

인과 판이라 시간 의미가 반대다: 창은 [t_now-W, t_now] 이고 **추정 대상은 t_now(창 끝)**.
  창 시작 = 과거, 창 끝 = 지금(추정 시각).
포함: 추정 pose + 방향화살표 + 카메라 시야 / top-k 후보 / 스캔정합 확대 2장 /
      VLM 관측 vs 지도기대 / 입력영상 2장 / 창 길이에 따른 수렴.
"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Wedge
import matplotlib.gridspec as gsp
from PIL import Image
import rt_localizer as RT
from semantic_map import FOV, R_BIG
from eval_semloc import peaks
for fp in ['/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
    try: fm.fontManager.addfont(fp); plt.rcParams['font.family']=fm.FontProperties(fname=fp).get_name(); break
    except Exception: pass
plt.rcParams['axes.unicode_minus']=False

HELD={'자판기':37,'LINC':54,'게시판':145,'화분끝':214,'사물함':303}
FR_DIR='/home/user/semloc_ws/frames'
_FR=sorted(float(f[1:-4]) for f in os.listdir(FR_DIR) if f.startswith('t') and f.endswith('.png'))
def frame_near(rel, tol=3.0):
    """프레임은 불규칙 시각에만 추출돼 있다(평가·캘리브 계획). 최근접을 쓰고 실제 시각을 함께 돌려준다."""
    if not _FR: return None, None
    k=min(_FR, key=lambda v: abs(v-rel))
    return (f'{FR_DIR}/t{k:g}.png', k) if abs(k-rel)<=tol else (None, None)
SHOW=['stairs','elevator','restroom_sign','door_left','door_right','long_view',
      'dead_end','corridor','open_area','desks_chairs']
W=float(sys.argv[1]) if len(sys.argv)>1 else 45.0
NAMES=sys.argv[2:] if len(sys.argv)>2 else list(HELD)
Z=RT.RTLocalizer(lam_sem=0.5,w_nogo=3.0,verbose=False); L=Z.L; sm=Z.sm
phi,boot_sc=Z.fix_phi(L.T[0],boot_s=15.0)
im=np.array(Image.open(RT.NL.MAPPNG).convert('L'))
print(f'φ 확정 {phi}  시동점수 ' + str({k:round(v,2) for k,v in boot_sc.items()}))

def hd_at(t):
    """지도프레임 전방각(이미지 좌표계, y아래). 화살표/시야에 그대로 쓴다."""
    yaw=L.YAW[np.argmin(np.abs(L.T-t))]
    return -(np.deg2rad(phi)+yaw+RT.BETA)

summ=[]
for nm in NAMES:
    ts=HELD[nm]; t_now=L.T[0]+ts
    r=Z.localize(t_now,W=W,phis=[phi])
    t_beg=float(r['frames'][0])          # ★로그 시작에서 잘릴 수 있으므로 실제 첫 프레임
    W_eff=t_now-t_beg
    P_e=r['pos']; gp_e=L.gt_px(t_now); gp_s=L.gt_px(t_beg)
    err_e=float(np.hypot(*(P_e-gp_e))/L.PPM)
    # 창 시작 pose = 추정 pose + A(φ)·(odom(t_beg)-odom(t_now))
    i_b=np.argmin(np.abs(L.T-t_beg)); i_n=np.argmin(np.abs(L.T-t_now))
    od=np.array([L.X[i_b]-L.X[i_n], L.Y[i_b]-L.Y[i_n]])
    P_s=P_e+L.PPM*(RT._A(np.deg2rad(phi))@od)
    err_s=float(np.hypot(*(P_s-gp_s))/L.PPM)
    h_e,h_s=hd_at(t_now),hd_at(t_beg)
    pk=peaks({phi:r['P'][phi]},L.Pf,allowed=Z.allow)[:60]
    traj=np.array([P_e+L.PPM*(RT._A(np.deg2rad(phi))@np.array(
        [L.X[np.argmin(np.abs(L.T-tt))]-L.X[i_n], L.Y[np.argmin(np.abs(L.T-tt))]-L.Y[i_n]]))
        for tt in np.linspace(t_beg,t_now,60)])

    fig=plt.figure(figsize=(17.5,11.2)); g=gsp.GridSpec(3,3,height_ratios=[1.35,1.0,0.92],hspace=.3,wspace=.2)
    ax=fig.add_subplot(g[0,:]); ax.imshow(im,cmap='gray',vmin=0,vmax=255)
    Pp=np.array([p for _,p,_ in pk]); scv=np.array([s_ for s_,_,_ in pk])
    scn=(scv-scv.min())/(np.ptp(scv)+1e-9)
    ax.scatter(Pp[:,0],Pp[:,1],s=14+34*scn,c=scn,cmap='autumn_r',alpha=.7,zorder=3,
               label=f'top-{len(pk)} 후보')
    ax.plot(traj[:,0],traj[:,1],'-',c='deepskyblue',lw=2.6,zorder=4,label=f'추정 경로 {W_eff:.0f}초')
    for Pq,hh,col,lb in ((P_s,h_s,'lime',f'창 시작 t-{W_eff:.0f}s'),(P_e,h_e,'magenta','지금 t (추정 대상)')):
        ax.add_patch(Wedge(Pq,R_BIG*L.PPM,np.rad2deg(hh)-np.rad2deg(FOV),
                           np.rad2deg(hh)+np.rad2deg(FOV),facecolor=col,alpha=.11,zorder=2))
        ax.arrow(Pq[0],Pq[1],110*np.cos(hh),110*np.sin(hh),width=8,color=col,ec='k',lw=.6,
                 zorder=6,length_includes_head=True)
        ax.scatter(*Pq,s=250,marker='o',facecolor='none',edgecolor=col,lw=3,zorder=6,label=f'추정 {lb}')
    ax.scatter(*gp_s,s=270,marker='*',c='cyan',edgecolor='blue',lw=1.3,zorder=7,label='참 창시작(GT)')
    ax.scatter(*gp_e,s=270,marker='P',c='cyan',edgecolor='blue',lw=1.3,zorder=7,label='참 지금(GT)')
    ax.set_xlim(900,3350); ax.set_ylim(1260,400); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='lower left',fontsize=8.5,ncol=4,framealpha=.9)
    ax.set_title(f'{nm} [{r["scene"]}]  실시간 인과 · 창 [t-{W_eff:.0f}s, t] · φ={phi} 시동확정 · 미래데이터 0\n'
                 f'지금(t={ts}s) 오차 {err_e:.2f}m  /  창시작 오차 {err_s:.2f}m',fontsize=12.5)

    def place(tq, Pq):
        """그 시각 스캔을 그 시각 추정위치에 얹어 지도픽셀로."""
        pts,yaw=L._scan(tq); hh=np.deg2rad(phi)+yaw+RT.BETA; c,s=np.cos(hh),np.sin(hh)
        return (Pq[0]+(pts[:,0]*c-pts[:,1]*s)*L.PPM, Pq[1]-(pts[:,0]*s+pts[:,1]*c)*L.PPM)
    def fitof(ex,ey):
        dd=np.clip(L.DFpx[np.clip(np.round(ey).astype(int),0,L.H-1),
                          np.clip(np.round(ex).astype(int),0,L.W-1)],0,10)
        return float(np.sort(dd)[:max(1,int(len(dd)*.7))].mean()/L.PPM)

    # B) 지금 시각 단일 스캔
    axz=fig.add_subplot(g[1,0]); axz.imshow(im,cmap='gray',vmin=0,vmax=255)
    ex,ey=place(t_now,P_e); axz.plot(ex,ey,'.',ms=1.7,c='red')
    axz.plot(*P_e,'o',ms=13,mfc='none',mec='magenta',mew=3)
    R=330; axz.set_xlim(P_e[0]-R,P_e[0]+R); axz.set_ylim(P_e[1]+R*.62,P_e[1]-R*.62)
    axz.set_xticks([]);axz.set_yticks([])
    axz.set_title(f'① 지금 시각 스캔 1프레임\n벽정합 {fitof(ex,ey):.3f}m',fontsize=10)

    # C) ★누적: 창 안 모든 프레임을 odom 으로 정렬해 겹쳐 그림 (belief 가 실제로 쓰는 것)
    axc2=fig.add_subplot(g[1,1]); axc2.imshow(im,cmap='gray',vmin=0,vmax=255)
    frs=r['frames']; AX=[]; AY=[]; CC=[]
    for u,tq in enumerate(frs):
        i=np.argmin(np.abs(L.T-tq))
        odq=np.array([L.X[i]-L.X[i_n], L.Y[i]-L.Y[i_n]])
        Pq=P_e+L.PPM*(RT._A(np.deg2rad(phi))@odq)
        exq,eyq=place(tq,Pq); AX.append(exq); AY.append(eyq)
        CC.append(np.full(len(exq), u/max(len(frs)-1,1)))
    AX=np.concatenate(AX); AY=np.concatenate(AY); CC=np.concatenate(CC)
    axc2.scatter(AX,AY,s=1.1,c=CC,cmap='viridis',alpha=.75,linewidths=0)
    axc2.plot(traj[:,0],traj[:,1],'-',c='deepskyblue',lw=1.6,alpha=.8)
    axc2.plot(*P_e,'o',ms=13,mfc='none',mec='magenta',mew=3)
    axc2.plot(*P_s,'o',ms=11,mfc='none',mec='lime',mew=2.5)
    R2=520; axc2.set_xlim(P_e[0]-R2,P_e[0]+R2); axc2.set_ylim(P_e[1]+R2*.62,P_e[1]-R2*.62)
    axc2.set_xticks([]);axc2.set_yticks([])
    axc2.set_title(f'② 창 전체 누적 — {len(frs)}프레임 odom 정렬 중첩\n'
                   f'벽정합 {fitof(AX,AY):.3f}m  (보라=과거 → 노랑=지금)',fontsize=10)

    ax4=fig.add_subplot(g[1,2]); ob=Z._ob_near(t_now-L.T[0])
    if ob:
        cal=Z.cal_past(t_now-W); ec=sm.expected(hd_at(t_now))
        idx=int(np.argmin(np.hypot(sm.G[:,0]-P_e[0],sm.G[:,1]-P_e[1])))
        ks=[k for k in SHOW if k in ob]; y=np.arange(len(ks))
        ax4.barh(y+.19,[cal.z(ob,k) for k in ks],height=.36,color='#d1495b',label='VLM 관측 z')
        ax4.barh(y-.19,[ec[k][idx] if k in ec else np.nan for k in ks],height=.36,
                 color='#3d84a8',label='지도 기대 e')
        ax4.set_yticks(y); ax4.set_yticklabels(ks,fontsize=8.5); ax4.invert_yaxis()
        ax4.set_xlim(0,1); ax4.legend(fontsize=8,loc='lower right'); ax4.grid(axis='x',alpha=.25)
    ax4.set_title(f'지금 시각 — 이미지가 본 것 vs 지도 기대',fontsize=10)

    for k,(tq,lb,col) in enumerate([(t_beg,f'창 시작 t-{W_eff:.0f}s','green'),(t_now,'지금 t','magenta')]):
        axi=fig.add_subplot(g[2,k]); rel=tq-L.T[0]
        fp,ft=frame_near(rel)
        if fp:
            axi.imshow(Image.open(fp))
            for sp in axi.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(3)
            extra='' if abs(ft-rel)<1e-6 else f' (최근접 t={ft:g}s)'
        else:
            axi.text(.5,.5,f't={rel:g}s 부근 프레임 없음',ha='center',va='center'); extra=''
        axi.set_xticks([]);axi.set_yticks([]); axi.set_title(f'{lb} 입력 영상{extra}',fontsize=10,color=col)

    axc=fig.add_subplot(g[2,2]); ws=[10,20,30,45,60]; ee=[]
    for w in ws:
        rr=Z.localize(t_now,W=float(w),phis=[phi])
        ee.append(float(np.hypot(*(rr['pos']-gp_e))/L.PPM))
    axc.plot(ws,ee,'o-',c='crimson',ms=5); axc.axhline(3,ls=':',c='green',lw=1.2)
    axc.axvline(W,ls='--',c='gray',lw=1)
    axc.set_yscale('log'); axc.set_xlabel('창 길이(초)',fontsize=9.5)
    axc.set_ylabel('오차(m)',fontsize=9.5); axc.grid(alpha=.25)
    axc.set_title('과거를 얼마나 보면 좁혀지나',fontsize=10)

    plt.suptitle(f'실시간 위치추정 — {nm}   지금 오차 {err_e:.2f} m',fontsize=14.5,y=.985)
    out=f'/home/user/semloc_ws/out/rt_{nm}.png'
    plt.savefig(out,dpi=84,bbox_inches='tight'); plt.close()
    summ.append((nm,err_e)); print(f'저장: {out}  지금 {err_e:.2f}m / 창시작 {err_s:.2f}m',flush=True)
e=np.array([x[1] for x in summ])
print(f'\n  중앙 {np.median(e):.2f}m | 3m이내 {int((e<3).sum())}/{len(e)}')
