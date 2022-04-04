from dataclasses import dataclass
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

########################
# Scoring...

def iou_score(a, b):
    SMOOTH = 1
    intersection = torch.logical_and(a, b).sum()
    union = torch.logical_or(a, b).sum()
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou.item()

########################
# Data handling...

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
    def __init__(self, dataset=TestDataset(), shuffle=False) -> None:
        super().__init__(dataset=dataset, batch_size=None, num_workers=10, pin_memory=True, shuffle=shuffle)