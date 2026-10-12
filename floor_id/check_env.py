#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_env.py — 온보드에서 돌기 전 환경을 점검한다. 문제가 있으면 무엇을 깔아야 하는지 알려준다."""
import sys
from pathlib import Path
PKG = Path(__file__).resolve().parent
ok = True


def chk(name, cond, hint=''):
    global ok
    print(f"  [{'OK ' if cond else '실패'}] {name}" + ('' if cond else f'  → {hint}'))
    ok = ok and cond


print(f'python {sys.version.split()[0]}')
chk('python 3.8 이상', sys.version_info >= (3, 8), 'python3.10 권장')
for mod, hint in (('numpy', 'pip install numpy'), ('scipy', 'pip install scipy'),
                  ('yaml', 'pip install pyyaml'), ('PIL', 'pip install pillow')):
    try:
        __import__(mod); chk(mod, True)
    except ImportError:
        chk(mod, False, hint)
try:
    import cv2
    chk(f'opencv {cv2.__version__}', True)
    chk('cv2.ximgproc (세선화)', hasattr(cv2, 'ximgproc'), 'pip install opencv-contrib-python')
except ImportError:
    chk('opencv', False, 'pip install opencv-contrib-python')

print('패키지 파일')
chk('config/buildings.yaml', (PKG / 'config' / 'buildings.yaml').exists())
chk('localizer/node_local.py', (PKG / 'localizer' / 'node_local.py').exists())
try:
    import yaml
    cfg = yaml.safe_load((PKG / 'config' / 'buildings.yaml').read_text(encoding='utf-8'))
    for k, b in cfg['buildings'].items():
        root = PKG / b['maps_root']
        miss = [m for m in range(1, int(b['n_maps']) + 1) if not (root / f'{m}F').is_dir()]
        chk(f"지도 {k} ({b['name']}) {b['n_maps']}층", not miss, f'없는 층 {miss}')
except Exception as e:
    chk('설정 읽기', False, str(e))

print('\n' + ('준비 완료. docs/USAGE.md 를 보고 실행할 것.' if ok else '위 실패 항목을 먼저 해결할 것.'))
sys.exit(0 if ok else 1)
