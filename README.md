# Torch Content Area
A PyTorch tool kit for segmenting the circular content area in endoscopic footage. This implementation is based off of that found in ["Detection of circular content area in endoscopic videos"](http://www-itec.uni-klu.ac.at/bib/files/CircleDetection.pdf)

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
Performance is tested against the [dataset](testing/data) included in this repo...
<!-- performance stats start -->
- Avg Time (NVIDIA GeForce GTX 980 Ti): 0.157ms
- Avg Error (Perimeter Distance): 2.026px
- Misses (Error > 5px): 4.3%
- Bad Misses (Error > 10px): 0.8%
- Classification Accuracy: 96.9% 
<!-- performance stats end -->

