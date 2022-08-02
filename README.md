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
Performance is tested against CholecECA, a subset of the [Endoscopic Content Area (ECA) dataset](https://github.com/charliebudd/eca-dataset)...
<!-- performance stats start -->

Performance Results (handcrafted)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 0.184ms
- Avg Error (Hausdorff Distance): 3.704
- Miss Rate (Error > 15): 2.0%
- Bad Miss Rate (Error > 25): 1.1%

Performance Results (learned)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 2.857ms
- Avg Error (Hausdorff Distance): 4.365
- Miss Rate (Error > 15): 2.4%
- Bad Miss Rate (Error > 25): 1.2% 
<!-- performance stats end -->

