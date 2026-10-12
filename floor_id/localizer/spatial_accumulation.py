"""Map-free accumulation with independent-frame support, retaining every input.

A raw observation is confirmed only when other acquisition times observe the
same location. This is evidence selection, not a deformation or a scan repair.
Unconfirmed points remain available to callers alongside the confirmed cloud.
"""
import numpy as np
from scipy.spatial import cKDTree
from sensor_frame import rotation


def accumulate(frames, beta, mirror=1, radius_m=.20, min_observations=3):
    if beta is None or mirror not in (-1,1):
        raise ValueError('A calibrated rotation and handedness are required')
    if radius_m <= 0 or min_observations < 2:
        raise ValueError('Positive support radius and at least two observations required')
    if not frames or np.any(np.diff([f['t'] for f in frames]) <= 0):
        raise ValueError('Use distinct, strictly time-ordered observations')
    anchor=np.asarray(frames[-1]['xy'],float)
    clouds=[];sources=[]
    for i,fr in enumerate(frames):
        pts=np.asarray(fr['pts'],float)
        if pts.ndim!=2 or pts.shape[1]!=2 or not np.isfinite(pts).all():
            raise ValueError('Scan points must be finite Nx2 coordinates')
        clouds.append(np.asarray(fr['xy'])-anchor+(pts*[1,mirror])@rotation(fr['yaw']+beta).T)
        sources.append(np.full(len(pts),i,dtype=np.int32))
    raw=np.vstack(clouds);frame_id=np.concatenate(sources)
    votes=np.zeros(len(raw),dtype=np.int32)
    # Each acquisition contributes at most one vote, irrespective of its point
    # count or the number of original scans merged inside an S3 observation.
    for cloud in clouds:
        if len(cloud):
            distance=cKDTree(cloud).query(raw,k=1,distance_upper_bound=radius_m)[0]
            votes += np.isfinite(distance)
    confirmed=votes>=min_observations
    return dict(raw=raw,confirmed=raw[confirmed],unconfirmed=raw[~confirmed],
                confirmed_mask=confirmed,frame_id=frame_id,observations=votes,
                anchor=anchor,radius_m=float(radius_m),min_observations=int(min_observations),
                frame_times=np.array([f['t'] for f in frames]))
