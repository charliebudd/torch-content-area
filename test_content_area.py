import torch
import random as rand
import matplotlib.pyplot as plt

import content_area

def draw_circle(width, height, x, y, r):

    range_x = torch.range(0, height - 1)
    range_y = torch.range(0, width - 1)
    grid_x, grid_y = torch.meshgrid(range_x, range_y)

    distance_squared = (grid_x - x) ** 2 + (grid_y - y) ** 2
    mask = distance_squared > (r * r)

    return mask.unsqueeze(0)

image_width = 1920
image_height = 1080

center_offset_min = 0.4
center_offset_max = 0.6
radius_range_min = 0.4
radius_range_max = 0.5

# rand.seed(0)
circle_x = rand.randint(int(image_height * center_offset_min), int(image_height * center_offset_max))
circle_y = rand.randint(int(image_width * center_offset_min), int(image_width * center_offset_max))
circle_r = rand.randint(int(image_width * radius_range_min), int(image_width * radius_range_min))

image = (torch.ones([3, image_height, image_width]) * 255).to(dtype=torch.uint8).cuda()
circle = draw_circle(image_width, image_height, circle_x, circle_y, circle_r).to(dtype=torch.uint8).cuda()

masked_image = image * (1 - circle)

out_mask = torch.zeros_like(circle)
# area = content_area.get_area(masked_image.contiguous())
content_area.get_area_mask(masked_image.contiguous(), out_mask.contiguous())

# print(area)
# print(circle_y, circle_x, circle_r)


heightgap = 135
lx = [462, 339, 262, 217, 198, 202, 231, 288]
rx = [1468, 1591, 1668, 1713, 1732, 1728, 1699, 1642]
y = [(i + 0.5) * heightgap for i in range(len(lx))]
from matplotlib.patches import Circle
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(masked_image.permute(1, 2, 0).cpu())
for xx, yy in zip(lx, y):
    circ = Circle((xx,yy),10)
    ax.add_patch(circ)
for xx, yy in zip(rx, y):
    circ = Circle((xx,yy),10)
    ax.add_patch(circ)
plt.show()



# print(torch.max(out_mask))
# print(torch.max(circle))

i = torch.sum(torch.logical_and(circle, out_mask))
u = torch.sum(torch.logical_or(circle, out_mask))
iou = torch.ones_like(i) if u == torch.zeros_like(i) else i / u
print(iou.item())

plt.imshow(masked_image.permute(1, 2, 0).cpu())
plt.show()

plt.imshow(circle.permute(1, 2, 0).cpu())
plt.show()

plt.imshow(out_mask.permute(1, 2, 0).cpu())
plt.show()



plt.imshow(abs(circle - out_mask).permute(1, 2, 0).cpu())
plt.show()
