import glob
import numpy as np
from PIL import Image

import time

import torch
from torch.utils.data import Dataset, DataLoader

import ast
import inspect

from os.path import dirname
__dir__ = dirname(__file__)

########################
# Profiling...

def timer(units='ms'):

    UNITS = {'h': 1.0/60.0, 'm': 1.0/60.0, 's': 1.0, 'ms': 1e3, 'us': 1e6}
    assert units in UNITS, f'The given units {units} is not supported, please use h, m, s, ms, or us'

    def contains_explicit_return(func):
        return any(isinstance(node, ast.Return) for node in ast.walk(ast.parse(inspect.getsource(func))))

    def get_time():
        torch.cuda.synchronize()
        return time.time()

    def decorator(func):
        if contains_explicit_return(func):
            def wrapper(*args):
                start = get_time()
                result = func(*args)
                end = get_time()
                func_time = (end - start) * UNITS[units]
                return func_time, result
        else:
            def wrapper(*args):
                start = get_time()
                func(*args)
                end = get_time()
                func_time = (end - start) * UNITS[units]
                return func_time
        return wrapper
    return decorator

@timer()
def timed(func, arg):
    return func(arg)

########################
# Scoring...

def iou_score(a, b):
    SMOOTH = 1
    intersection = torch.logical_and(a, b).sum()
    union = torch.logical_or(a, b).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.item()

def parameter_errors(a, b):
    a_center = np.array([a[0], a[1]])
    a_radius = a[2]
    
    b_center = np.array([b[0], b[1]])
    b_radius = b[2]

    center_error = np.linalg.norm(a_center - b_center)
    radius_error = abs(a_radius - b_radius)

    return center_error, radius_error

########################
# Data handling...

class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.width = 854
        self.height = 480

        self.areas = [
            (400, 250, 360),
            (340, 200, 370),
            (450, 230, 250),
        ]

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, index):
        area_x, area_y, area_r = self.areas[index]

        coords = torch.stack(torch.meshgrid(torch.arange(0, self.height), torch.arange(0, self.width), indexing="ij"))
        center = torch.Tensor([area_y, area_x]).reshape((2, 1, 1))

        mask = torch.where(torch.linalg.norm(abs(coords - center), dim=0) > area_r, 0, 1).unsqueeze(0)
        img = mask * torch.randint(0, 255, (3, self.height, self.width))

        img = img.to(dtype=torch.uint8).contiguous()
        mask = mask.to(dtype=torch.uint8).contiguous()

        return img, mask, (area_x, area_y, area_r)

class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.img_paths = sorted(glob.glob(f'{__dir__}/data/*_img.png'))
        self.seg_paths = sorted(glob.glob(f'{__dir__}/data/*_seg.png'))
        self.length = len(self.img_paths)

        assert len(self.img_paths) == len(self.seg_paths), f'Number of images ({len(self.img_paths)}) doesn\'t match the number of segmentations ({len(self.seg_paths)})!'
        assert len(self.img_paths) > 0, f'No samples found!'

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img, seg = self.img_paths[index], self.seg_paths[index]
        img, seg = tuple(map(lambda x: torch.from_numpy(np.array(Image.open(x))), (img, seg)))
        return img.permute(2, 0, 1), seg.unsqueeze(0)

class TestDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=False) -> None:
        super().__init__(dataset=dataset, batch_size=None, num_workers=10, pin_memory=True, shuffle=shuffle)
