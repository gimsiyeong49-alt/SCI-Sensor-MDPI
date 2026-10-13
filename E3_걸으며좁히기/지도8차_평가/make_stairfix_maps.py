#!/usr/bin/env python3
"""계단 앞 벽을 관통 판정에서 빼는 지도 만들기(2026-10-12). 원본 지도는 그대로 두고 새 폴더에 복사본을 만든다.
위치추정의 벽 관통 판정은 '벽 − 문 마스크'(metric_fit_v2._maps 의 wall_tr)로 잰다. 그래서 계단 둘레의 벽을 문 마스크(maps/door_mask.png)에 더하면
코드를 고치지 않고 지도만으로 계단 쪽 관통을 없앨 수 있다. 점수용 벽(base_map)과 의미지도(sem/)는 건드리지 않는다.
  규칙(지도만 본다, 정답 경로는 안 씀): 계단 마스크(sem/4F_stair_mask.png)의 덩어리마다, 그 덩어리 bounding box 짧은 변 × FRAC 안의 벽 픽셀을 문 마스크에 더한다.
  안내도는 계단 앞 방화셔터를 벽처럼 그리지만 평소엔 열려 있어 로봇이 지나다닌다(지도빈곳점검_20261012, 정보관 3·5·6·7·8·9층).
  --mode shutter(기본, v2): 그중 방(sem/01_fused_room_mask.png)이나 건물 밖(mcl/building_footprint.png 바깥)에 NEAR px 안으로 붙은 벽은 남긴다.
        셔터는 양쪽이 복도·계단 공간이고, 계단실 옆 벽은 방에, 외벽은 건물 밖에 붙는다.
  --mode all(v1): 계단 둘레 벽을 전부 연다(계단실 옆 벽까지 열려 과함).
  --mode dangling(v4): 셔터 모양만 연다. 벽(위치추정과 같은 가짜 벽 제거 뒤)의 뼈대에서 '한쪽 끝이 떠 있는 곧은 조각'(끝점이 있는 가지) 중
        계단 둘레(짧은 변 × FRAC) 안에 있고 길이 MINLEN px 이상, 곧음(끝점 거리/길이) 0.8 이상인 것을 그 벽 두께만큼 연다.
        계단실 벽은 양 끝이 모서리로 이어진 닫힌 테두리라 남는다. 계단 마스크를 INNER px 줄인 칸 안의 남은 벽도 연다(로봇이 계단에 올라선 경우, 테두리 벽은 남김).
  --mode shutter_line(v6): dangling 가지 중 그 계단의 긴 변과 나란하고(15° 안) 길이가 긴 변 × 0.6 이상이며 계단 bbox 밖에 있는 것만 연다.
        계단 빗금 안쪽은 열지 않는다. 복도 셔터는 지도 생성에서 이미 지워져 남은 것은 가운데 계단 앞 셔터 선뿐이다(2026-10-12 사용자).
        v5 에서 셔터가 아닌데 열린 곳(계단실 옆 세로 조각·모서리 조각·빗금 안쪽 벽, 사용자 지적 22곳)을 거르려는 조건이다.
        v7: 그중 조각의 절반 넘게가 방 마스크에서 NEAR px 안이면 뺀다(방 칸막이, 소융대 3층 312|313 에서 잘못 열림).
    python3 scripts/make_stairfix_maps.py --dst /data2/floor_id_maps_stairfix_v2_20261012          # v2
    python3 scripts/make_stairfix_maps.py --mode all --dst /data2/floor_id_maps_stairfix_20261012  # v1
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects, distance_transform_edt, binary_dilation, binary_erosion, convolve

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/data/kci_work/collab_laptop_20261012/onboard_robot_20261001/floor_id/maps")
ap.add_argument("--dst", default="/data2/floor_id_maps_stairfix_20261012")
ap.add_argument("--frac", type=float, default=0.6)
ap.add_argument("--min-area", type=int, default=200, help="이보다 작은 계단 덩어리는 무시(잡티)")
ap.add_argument("--mode", default="shutter", choices=["shutter", "all", "dangling", "shutter_line"])
ap.add_argument("--inner", type=int, default=4, help="dangling: 계단 마스크를 이만큼 안으로 줄인 칸의 벽만 연다")
ap.add_argument("--minlen", type=int, default=30, help="dangling: 이보다 짧은 가지는 벽 모서리 잔가지로 보고 무시")
ap.add_argument("--near", type=int, default=12, help="shutter: 방·건물 밖에서 이 px 안의 벽은 남김(벽 두께 8~12 px)")
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
if dst.exists():
    sys.exit(f"{dst} 이 이미 있다(덮어쓰지 않음)")
shutil.copytree(src, dst)
sys.path.insert(0, str(src.parent / "localizer"))


def dangling_open(base, st, zone, slab=None, room=None):
    from skimage.morphology import skeletonize
    from wall_clean import clean_wall
    wall, _ = clean_wall(base)
    sk = skeletonize(wall)
    nb = convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    junc = binary_dilation(sk & (nb >= 3), iterations=1)
    br, nbr = label(sk & ~junc, structure=np.ones((3, 3)))
    endp = sk & (nb == 1)
    thick = distance_transform_edt(wall)
    droom = distance_transform_edt(~room) if room is not None else None
    out = np.zeros_like(wall)
    for i, sl in enumerate(find_objects(br)):
        m = br[sl] == i + 1
        ys, xs = np.nonzero(m); ys = ys + sl[0].start; xs = xs + sl[1].start
        n = len(ys)
        if n < a.minlen or not endp[ys, xs].any():
            continue
        if zone[ys, xs].mean() < 0.5 or st[ys, xs].mean() > 0.5:
            continue
        pts = np.stack([xs, ys], 1); d = np.max(np.linalg.norm(pts[:, None] - pts[None, ::max(n // 50, 1)], axis=-1))
        if d / n < 0.8:
            continue
        if slab is not None:                      # v6: 가장 가까운 계단과 나란하고 그 폭만큼 긴, 계단 밖 선만
            cx, cy = xs.mean(), ys.mean(); best = None
            for j, ss in enumerate(find_objects(slab)):
                if (slab[ss] == j + 1).sum() < a.min_area:
                    continue
                dd = np.hypot(np.clip(cx, ss[1].start, ss[1].stop) - cx, np.clip(cy, ss[0].start, ss[0].stop) - cy)
                if best is None or dd < best[0]:
                    best = (dd, ss)
            if best is None or best[0] == 0:
                continue
            ss = best[1]; bw, bh = ss[1].stop - ss[1].start, ss[0].stop - ss[0].start
            ang = np.degrees(np.arctan2(*np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0][::-1])) % 180
            want = 0.0 if bw >= bh else 90.0
            if min(abs(ang - want), 180 - abs(ang - want)) > 15 or d < 0.6 * max(bw, bh):
                continue
            if droom is not None and np.mean(droom[ys, xs] <= a.near) > 0.5:   # 방에 붙은 조각(방 칸막이)은 셔터가 아님(v7, 소융대 3층 312|313)
                continue
        seg = np.zeros_like(wall); seg[ys, xs] = True
        out |= binary_dilation(seg, iterations=int(np.ceil(thick[ys, xs].max())) + 2) & wall
    if slab is None:
        out |= wall & binary_erosion(st, iterations=a.inner)      # 계단 빗금 칸 안쪽만(테두리 벽은 남김)
    return out


rep = []
for b in ("kci", "gong", "sw"):
    for fl in sorted((dst / f"{b}_maps_perfloor").iterdir()):
        dm_p = fl / "maps/door_mask.png"; st_p = dst / f"{b}_semmaps/{fl.name}/sem/4F_stair_mask.png"
        if not dm_p.exists() or not st_p.exists():
            continue
        base = np.array(Image.open(fl / "maps/base_map.png").convert("L")) < 128
        door_img = Image.open(dm_p); door = np.array(door_img.convert("L")) > 127
        st = np.array(Image.open(st_p).convert("L")) > 127
        lab, n = label(st); zone = np.zeros_like(st)
        for i, s in enumerate(find_objects(lab)):
            comp = lab == i + 1
            if comp.sum() < a.min_area:
                continue
            r = a.frac * min(s[0].stop - s[0].start, s[1].stop - s[1].start)
            y0, y1 = max(s[0].start - int(r) - 1, 0), min(s[0].stop + int(r) + 1, st.shape[0])
            x0, x1 = max(s[1].start - int(r) - 1, 0), min(s[1].stop + int(r) + 1, st.shape[1])
            d = distance_transform_edt(~comp[y0:y1, x0:x1])
            zone[y0:y1, x0:x1] |= d <= r
        add = base & zone & ~door
        if a.mode in ("dangling", "shutter_line"):
            room = None
            if a.mode == "shutter_line":
                room = np.array(Image.open(dst / f"{b}_semmaps/{fl.name}/sem/01_fused_room_mask.png").convert("L")) > 127
            add = dangling_open(base, st, zone, lab if a.mode == "shutter_line" else None, room) & ~door
        elif a.mode == "shutter":
            sem = dst / f"{b}_semmaps/{fl.name}"
            room = np.array(Image.open(sem / "sem/01_fused_room_mask.png").convert("L")) > 127
            outside = ~(np.array(Image.open(sem / "mcl/building_footprint.png").convert("L")) > 127)
            add &= ~binary_dilation(room | outside, iterations=a.near)
        new = door | add
        out = np.where(new, 255, 0).astype(np.uint8)
        Image.fromarray(out, "L").save(dm_p)
        rep.append(dict(건물=b, 층=fl.name, 계단덩어리=int(sum((lab == i + 1).sum() >= a.min_area for i in range(n))),
                        더한_벽px=int(add.sum()), 벽대비=round(float(add.sum() / max(base.sum(), 1)) * 100, 2)))
        print(rep[-1], flush=True)
(dst / "STAIRFIX_README.json").write_text(json.dumps(dict(원본=str(src), mode=a.mode, near=a.near, minlen=a.minlen, inner=a.inner, frac=a.frac, min_area=a.min_area, 층=rep), ensure_ascii=False, indent=1))
