# torch content area
A tool kit for segmenting the endoscopic content area in laparoscopy footage.

## performance
Performance is tested against the [dataset included in this repo](testing/data). The follow results were acheived when running on an NVIDIA Quadro RTX 3000...
- Avg Time: 1.285ms
- Avg Score (IoU): 0.980
- Misses (IoU < 0.95): 8.1%