#!/usr/bin/env python3
"""사람 검수 목록의 셔터 자리를 문 마스크에 더한 지도 만들기(2026-10-12). 원본 수정판은 그대로 두고 새 폴더에 복사한다.
목록 항목마다 그 점에서 가장 가까운(SEARCH px 안) '한쪽 끝이 떠 있는 벽 조각'(벽 뼈대의 끝점 가지)을 찾아 그 벽 두께만큼 문 마스크에 더한다.
정답 경로는 쓰지 않는다(사람이 안내도를 보고 정한 자리).
    python3 scripts/apply_map_review.py --src /data2/floor_id_maps_stairfix_v7_20261012 --dst /data2/floor_id_maps_v8_review_20261012 --review configs/map_review/검수목록_20261012.json
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt, label

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True); ap.add_argument("--review", required=True)
ap.add_argument("--search", type=int, default=30)
a = ap.parse_args()
src, dst = Path(a.src), Path(a.dst)
if dst.exists():
    sys.exit(f"{dst} 이 이미 있다(덮어쓰지 않음)")
shutil.copytree(src, dst)
sys.path.insert(0, "/data/kci_work/collab_laptop_20261012/onboard_robot_20261001/floor_id/localizer")
from wall_clean import clean_wall  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402
log = []
for it in json.load(open(a.review)):
    fl = dst / f"{it['building']}_maps_perfloor/{it['floor']}/maps"
    base = np.array(Image.open(fl / "base_map.png").convert("L")) < 128; wall, _ = clean_wall(base)
    door = np.array(Image.open(fl / "door_mask.png").convert("L")) > 127
    sk = skeletonize(wall); nb = convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - sk
    junc = binary_dilation(sk & (nb >= 3), iterations=1); br, _ = label(sk & ~junc, structure=np.ones((3, 3)))
    endp = sk & (nb == 1); px, py = it["point_px"]
    ys, xs = np.nonzero(br > 0); d = np.hypot(xs - px, ys - py); best = None
    for k in np.argsort(d):
        if d[k] > a.search:
            break
        lab = br[ys[k], xs[k]]; m = br == lab
        if endp[m].any():
            best = m; break
    if best is None:
        log.append(dict(it, 결과="못 찾음")); print("못 찾음:", it["floor"], it["point_px"]); continue
    thick = distance_transform_edt(wall); t = int(np.ceil(thick[best].max())) + 2
    add = binary_dilation(best, iterations=t) & wall & ~door
    Image.fromarray(np.where(door | add, 255, 0).astype(np.uint8), "L").save(fl / "door_mask.png")
    log.append(dict(it, 결과=f"벽 {int(add.sum())} px 더함", 가지길이px=int(best.sum()))); print(it["floor"], it["point_px"], "→", log[-1]["결과"])
(dst / "REVIEW_README.json").write_text(json.dumps(dict(원본=str(src), 검수목록=a.review, 항목=log), ensure_ascii=False, indent=1))
