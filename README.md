# Torch Content Area
A PyTorch tool kit for estimating the circular content area in endoscopic footage.

[![Build Status](https://github.com/charliebudd/torch-content-area/actions/workflows/build.yml/badge.svg)](https://github.com/charliebudd/torch-content-area/actions/workflows/build.yml)

![Example GIF](example.gif?raw=true)

## Installation
To install the latest version, simply run...
```
pip install torchcontentarea
```

## Usage
```python
from torchvision.io import read_image
from torchcontentarea import estimate_area, get_points, fit_area

# Image in NCHW format, byte/uint8 type is expected
image = read_image("my_image.png").unsqueeze(0)

# Either directly estimate area from image...
area = estimate_area(image, strip_count=16)

# ...or get the set of points and then fit the area.
points = get_points(image, strip_count=16)
area = fit_area(points, image.shape[2:4])
```

## Performance
Performance is measured against the CholecECA subset of the [Endoscopic Content Area (ECA) dataset](https://github.com/charliebudd/eca-dataset).
<!-- performance stats start -->

Performance Results (handcrafted cpu)...
- Avg Time (Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz): 2.501ms
- Avg Error (Hausdorff Distance): 3.535
- Miss Rate (Error > 15): 2.1%
- Bad Miss Rate (Error > 25): 1.0%

Performance Results (learned cpu)...
- Avg Time (Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz): 4.662ms
- Avg Error (Hausdorff Distance): 4.388
- Miss Rate (Error > 15): 2.6%
- Bad Miss Rate (Error > 25): 1.4%

Performance Results (handcrafted cuda)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 0.171ms
- Avg Error (Hausdorff Distance): 4.289
- Miss Rate (Error > 15): 2.4%
- Bad Miss Rate (Error > 25): 1.3%

Performance Results (learned cuda)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 1.349ms
- Avg Error (Hausdorff Distance): 4.641
- Miss Rate (Error > 15): 2.6%
- Bad Miss Rate (Error > 25): 1.3% 
<!-- performance stats end -->

