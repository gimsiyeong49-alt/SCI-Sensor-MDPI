# '벽을 반대편으로 통과했나' 판정: 벽 안 구간마다 들어가기 직전·나온 직후 칸이 벽을 안 지나고 얼마나 돌아야 이어지는가
import numpy as np, csv, json, math, sys
from skimage.graph import MCP_Geometric
S='<작업폴더>'   # 원래 작업 폴더(공유본에서는 지움)
GT='reports/_00_SIGN_PIPELINE_20260920/31_벽구간이벤트/36_채점용정답'
def dense(tr):
    seg=np.diff(tr,axis=0); n=np.maximum(np.ceil(np.hypot(seg[:,0],seg[:,1])).astype(int),1)
    return np.vstack([tr[i]+seg[i]*(np.arange(n[i])[:,None]/n[i]) for i in range(len(seg))]+[tr[-1:]])
def passes(W,tr,ppm,R_m=6.0,detour_m=2.0,ratio=3.0):
    """벽 통과 횟수와 각 통과의 (직선 m, 우회 m)"""
    P=np.rint(dense(tr)).astype(int); H,Wd=W.shape
    ok=(P[:,0]>=0)&(P[:,0]<Wd)&(P[:,1]>=0)&(P[:,1]<H); P=P[ok]
    inw=W[P[:,1],P[:,0]]; ev=[]; i=0; n=len(P)
    while i<n:
        if not inw[i]: i+=1; continue
        j=i
        while j<n and inw[j]: j+=1
        if i==0 or j>=n: i=j; continue
        a,b=P[i-1],P[j]; eu=np.hypot(*(a-b))/ppm
        R=int(R_m*ppm); x0,y0=max(0,min(a[0],b[0])-R),max(0,min(a[1],b[1])-R); x1,y1=min(Wd,max(a[0],b[0])+R),min(H,max(a[1],b[1])+R)
        cost=np.where(W[y0:y1,x0:x1],-1.0,1.0)          # 벽은 못 지나감(음수=통행 불가)
        m=MCP_Geometric(cost); acc,_=m.find_costs([(a[1]-y0,a[0]-x0)],[(b[1]-y0,b[0]-x0)])
        geo=acc[b[1]-y0,b[0]-x0]/ppm
        if not np.isfinite(geo) or geo>max(ratio*eu, eu+detour_m): ev.append((eu,geo))
        i=j
    return ev
if __name__=='__main__':
    sys.path.insert(0,S+'/go2_ro/realtime'); import pose_tools as PT
    E=S+'/go2_ro/exp_7F'; cands=json.load(open(E+'/out/floor.json'))['cands']
    rows=list(csv.DictReader(open(E+'/data/odom_link.csv'))); T=np.array([float(r['t_sec']) for r in rows])
    print('[오늘 415 실험 · 층별 후보 · 로봇 지도]')
    for k,c in enumerate(cands):
        m=c['map']; W=np.load(f'{S}/walltr_robot1005_{m}F.npy')
        tr=np.array([PT.map_pose_px(c,float(r['fastlio_x']),float(r['fastlio_y']),float(r['fastlio_yaw_rad']))[:2] for r,t in zip(rows,T) if c['t']-60<=t<=c['t']])
        ev=passes(W,tr,c['ppm'])
        print(f'  {k+1}위 {m}층: 벽 통과 {len(ev)}번 '+(' '.join(f'(직선 {e:.2f} m→우회 {"막힘" if not np.isfinite(g) else f"{g:.1f} m"})' for e,g in ev[:3])))
    print('\n[정답 경로 60초 창 · 8차 지도] 벽 통과가 1번 이상 잡힌 창 비율 (정답이니 0% 여야 좋음)')
    for mm in range(1,10):
        F=f'{mm}F'; W=np.load(f'{S}/walltr_v8_{F}.npy'); ppm=json.load(open(f'{GT}/{F}/summary.json'))['ppm']
        g=np.array([[float(r['relative_sec']),float(r['x_px']),float(r['y_px'])] for r in csv.DictReader(open(f'{GT}/{F}/warp_clean/gt_warp.csv'))])
        hit=[];tot=0
        for t1 in np.arange(60,g[-1,0],10.0):
            s=(g[:,0]>=t1-60)&(g[:,0]<=t1)
            if s.sum()<20: continue
            tot+=1; hit.append(len(passes(W,g[s,1:][::2],ppm))>0)
        print(f'  {F}: {100*np.mean(hit):5.1f}%  ({sum(hit)}/{tot})')
