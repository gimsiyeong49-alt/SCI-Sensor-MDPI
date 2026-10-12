"""Raw rosbag location: USB only. Extracted samples/caches stay on the laptop."""
from pathlib import Path

USB_DATASET = Path('/media/collab/USB/flower_dataset')


def usb_bag(floor):
    floor = int(floor)
    if floor not in range(1, 10):
        raise ValueError('floor must be between 1 and 9')
    folder = '3flower_ v2' if floor == 3 else f'{floor}flower'
    name = '4F_full_walk_0.db3' if floor == 4 else f'{floor}flower_0.db3'
    return str(USB_DATASET / folder / name)
