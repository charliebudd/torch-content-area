import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import content_area

import time

plt.rcParams["figure.figsize"] = (20, 5)

image = torchvision.io.read_image("examples/hard.png").cuda()
# gray = torchvision.io.read_image("examples/easy.png", torchvision.io.ImageReadMode.GRAY).cuda()
mask = torch.zeros_like(image[0:1, ...])

image = image.contiguous()
mask = mask.contiguous()

time_sum = 0
time_count = 50
for i in range(time_count):
    start = time.time()
    content_area.get_area_mask(image, mask)
    time_sum += time.time() - start
print("time(ms): ", 1e3 * time_sum / time_count)

image = image.permute(1, 2, 0).cpu().numpy()
mask = mask.permute(1, 2, 0).cpu().numpy()

heightgap = 135
lx = [247, 175, 510, 322, 271, 393, 381, 508]
rx = [1480, 1778, 1475, 1846, 1451, 1672, 1408, 1507]
lx = [0, 0, 503, 111, 231, 393, 359, 319]
rx = [1704, 1779, 1824, 1847, 1848, 1829, 1789, 1720]
y = [(i + 0.5) * heightgap for i in range(len(lx))]
from matplotlib.patches import Circle
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(image)
for xx, yy in zip(lx, y):
    circ = Circle((xx,yy),10)
    ax.add_patch(circ)
for xx, yy in zip(rx, y):
    circ = Circle((xx,yy),10)
    ax.add_patch(circ)
plt.show()

plt.subplot(121).axis('off')
plt.imshow(image)
plt.subplot(122).axis('off')
plt.imshow(mask)
plt.show()

mask = mask[..., 0] * 255
mask = np.stack([mask, mask, mask], axis=-1)

plt.imshow((image * 0.8 + mask * 0.2) / 255)
plt.show()