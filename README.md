# torch content area
A tool kit for segmenting the endoscopic content area in laparoscopy footage. This implementation is based off of that found in ["Detection of circular content area in endoscopic videos"](http://www-itec.uni-klu.ac.at/bib/files/CircleDetection.pdf)

## performance
Performance is tested against the [dataset](testing/data) included in this repo. The follow results were achieved when running on an NVIDIA Quadro RTX 3000...
- Avg Time: 1.285ms
- Avg Score (IoU): 0.980
- Misses (IoU < 0.95): 8.1%
