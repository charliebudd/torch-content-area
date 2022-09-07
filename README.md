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
First, import the `ContentAreaInference` class and create an instance...
```
from torchcontentarea import ContentAreaInference

content_area = ContentAreaInference()
```
Then, either infer the content area mask directly...
```
mask = content_area.infer_mask(image)
```
Or return a description of the content area which may be adjusted before drawing the mask...
```
area = content_area.infer_area(image)
area = edit_area(area)
mask = content_area.draw_mask(image, area)
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

