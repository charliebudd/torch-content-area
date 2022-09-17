import time
import torch

UNITS = {'h': 1.0/60.0, 'm': 1.0/60.0, 's': 1.0, 'ms': 1e3, 'us': 1e6}

def get_time():
    torch.cuda.synchronize()
    return time.time()

class Timer():
    def __init__(self, units='ms'):
        assert units in UNITS, f'The given units {units} is not supported, please use h, m, s, ms, or us'
        self.units = units

    def __enter__(self):
        self.start = get_time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = get_time()
        self.time = (self.end - self.start) * UNITS[self.units]
