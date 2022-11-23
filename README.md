# Torch Content Area
A PyTorch tool kit for estimating the circular content area in endoscopic footage. This implementation is released alongside our publication:

<ul><b>Rapid and robust endoscopic content area estimation: A lean GPU-based pipeline and curated benchmark dataset</b>,<br>
    Charlie Budd, Luis C. Garcia-Peraza-Herrera, Martin Huber, Sebastien Ourselin, Tom Vercauteren.<br>
    [ <a href="https://arxiv.org/abs/2210.14771">arXiv</a> ]
</ul>

If you make use of this work, please cite the paper.

[![Build Status](https://github.com/charliebudd/torch-content-area/actions/workflows/build.yml/badge.svg)](https://github.com/charliebudd/torch-content-area/actions/workflows/build.yml)

![Example GIF](example.gif?raw=true)

## Installation
For Linux users, to install the latest version, simply run...
```
pip install torchcontentarea
```
For Windows users, or if you encounter any issues, try building from source by running...
```
pip install git+https://github.com/charliebudd/torch-content-area
```
***Note:*** *this will require that you have CUDA installed and that its version matches the version of CUDA used to build your installation of PyTorch.*

## Usage
```python
from torchvision.io import read_image
from torchcontentarea import estimate_area, get_points, fit_area

# Grayscale or RGB image in NCHW or CHW format, byte/uint8 type is expected
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
- Avg Time (Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz): 2.157 ± 0.632ms
- Avg Error (Hausdorff Distance): 3.595
- Miss Rate (Error > 15): 2.1%
- Bad Miss Rate (Error > 25): 1.0%

Performance Results (learned cpu)...
- Avg Time (Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz): 4.764 ± 2.268ms
- Avg Error (Hausdorff Distance): 4.560
- Miss Rate (Error > 15): 2.5%
- Bad Miss Rate (Error > 25): 1.4%

Performance Results (handcrafted cuda)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 0.164 ± 0.066ms
- Avg Error (Hausdorff Distance): 3.618
- Miss Rate (Error > 15): 2.1%
- Bad Miss Rate (Error > 25): 1.1%

Performance Results (learned cuda)...
- Avg Time (NVIDIA GeForce GTX 980 Ti): 1.209 ± 0.640ms
- Avg Error (Hausdorff Distance): 4.641
- Miss Rate (Error > 15): 2.6%
- Bad Miss Rate (Error > 25): 1.3% 
<!-- performance stats end -->

