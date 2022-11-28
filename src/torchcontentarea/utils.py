import torch
from math import sqrt, floor

def draw_area(area, image, bias=0):
    image_size = image.shape[-2:]
    device = image.device
    mesh = torch.meshgrid(torch.arange(0, image_size[0], device=device), torch.arange(0, image_size[1], device=device))
    dist = torch.sqrt((mesh[0] - area[1])**2 + (mesh[1] - area[0])**2)
    mask = torch.where(dist < (area[2] + bias), 1, 0).to(dtype=torch.uint8)
    return mask

def get_crop(area, image_size, aspect_ratio=None, bias=0):

    i_h, i_w = image_size
    a_x, a_y, a_r = area[:3]

    if aspect_ratio == None:
        aspect_ratio = i_w / i_h

    inscribed_height = 2 * (a_r + bias - 2) / sqrt(1 + aspect_ratio * aspect_ratio)
    inscribed_width = inscribed_height * aspect_ratio

    left = max(a_x - inscribed_width / 2, 0)
    right = min(a_x + inscribed_width / 2, i_w)
    top = max(a_y - inscribed_height / 2, 0)
    bottom = min(a_y + inscribed_height / 2, i_h)

    x_scale = (right - left)
    y_scale = (bottom - top) * aspect_ratio

    scale = min(x_scale, y_scale)

    w = int(floor(scale))
    h = int(floor(scale / aspect_ratio))

    x = int(left + (right - left) / 2 - w / 2)
    y = int(top + (bottom - top) / 2 - h / 2)

    crop = y, y+h, x, x+w

    return crop

def crop_area(area, image, aspect_ratio=None, bias=0):

    crop = get_crop(area, image.shape[-2:], aspect_ratio, bias)
    
    cropped_image = image[..., crop[0]:crop[1], crop[2]:crop[3]]

    return cropped_image
