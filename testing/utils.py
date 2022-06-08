import json
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

def perimeter_distance_score(a, b):
    a_center = np.array([a[0], a[1]])
    a_radius = a[2]

    b_center = np.array([b[0], b[1]])
    b_radius = b[2]

    d = np.linalg.norm(a_center - b_center)

    ab = abs(a_radius - b_radius + d)
    ba = abs(b_radius - a_radius + d)

    return (ab + ba) / 2

def mean_border_distance(circle_a, circle_b, frame_size):

    def closest_point_on_circle(px, py, cx, cy, cr):
        dx, dy = px - cx, py - cy
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm, dy / norm
        return cx + dx * cr, cy + dy * cr

    def in_bounds(x, y, width, height):
        return x > 0 and x < width and y > 0 and y < height

    def distance(a_x, a_y, b_x, b_y):
        return np.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)

    def clamp_to_frame(a_x, a_y, b_x, b_y, width, height):
        m = (b_y - a_y) / (b_x - a_x)
        c =  b_y - m * b_x
        b_x = max(0, min(b_x, width-1))
        b_y = m*b_x + c
        b_y = max(0, min(b_y, height-1))
        b_x = (b_y-c) / m
        return b_x, b_y

    a_x, a_y, a_r = circle_a
    b_x, b_y, b_r = circle_b
    width, height = frame_size

    distances = []

    for theta in np.linspace(0, 2 * np.pi, 360):
        p1_x = a_x + np.cos(theta) * a_r
        p1_y = a_y + np.sin(theta) * a_r
        p2_x, p2_y = closest_point_on_circle(p1_x, p1_y, b_x, b_y, b_r)

        p1_in = in_bounds(p1_x, p1_y, width, height)
        p2_in = in_bounds(p2_x, p2_y, width, height)

        if not p1_in and not p2_in:
            continue
        elif not p1_in:
            p1_x, p1_y = clamp_to_frame(p2_x, p2_y, p1_x, p1_y, width, height)
        elif not p2_in:
            p2_x, p2_y = clamp_to_frame(p1_x, p1_y, p2_x, p2_y, width, height)

        dist = distance(p1_x, p1_y, p2_x, p2_y)
        distances.append(dist)

    return sum(distances) / len(distances)

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
            None,
        ]

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, index):
        area = self.areas[index]

        if area != None:
            area_x, area_y, area_r = self.areas[index]
            coords = torch.stack(torch.meshgrid(torch.arange(0, self.height), torch.arange(0, self.width), indexing="ij"))
            center = torch.Tensor([area_y, area_x]).reshape((2, 1, 1))
            mask = torch.where(torch.linalg.norm(abs(coords - center), dim=0) < area_r, 0, 1).unsqueeze(0)
        else:
            mask = torch.zeros(1, self.height, self.width)

        img = 255 * (1 - mask).expand((3, self.height, self.width))
        img = img.to(dtype=torch.uint8).contiguous()
        mask = mask.to(dtype=torch.uint8).contiguous()

        return img, mask, area

class TestDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        with open(f'{__dir__}/data/samples.json') as samples_file:
            samples = json.load(samples_file)
        self.img_paths = [f'{__dir__}/data/{s["image_file"]}' for s in samples]
        self.seg_paths = [f'{__dir__}/data/{s["segmentation_file"]}' for s in samples]
        self.content_areas = [s["content_area"] for s in samples]
        self.length = len(samples)

    def __len__(self):
        return 2 * self.length

    def __getitem__(self, index):

        crop = bool(index % 2)
        index //= 2

        img = torch.from_numpy(np.array(Image.open(self.img_paths[index]))).permute(2, 0, 1)

        if crop:
            x_low, x_high = int(img.shape[1] * 0.3), int(img.shape[1] * 0.7)
            y_low, y_high = int(img.shape[2] * 0.3), int(img.shape[2] * 0.7)
            img = img[:, x_low:x_high, y_low:y_high]
            return img, None
        else:
            return img, self.content_areas[index]

class TestDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=False) -> None:
        super().__init__(dataset=dataset, batch_size=None, num_workers=10, pin_memory=True, shuffle=shuffle)
