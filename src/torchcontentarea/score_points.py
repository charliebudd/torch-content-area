import torch
import math
from torch.nn.functional import pad

model = None

norm_means = (0.3441, 0.2251, 0.2203)
norm_stds = (0.2381, 0.1994, 0.1939)

def meshgrid(tensors):
    if torch.__version__ >= "1.10":
        return torch.meshgrid(tensors, indexing="ij")
    else:
        return torch.meshgrid(tensors)

def normalise(tensor, mean, std):
    mean = torch.as_tensor(mean, device=tensor.device)
    std = torch.as_tensor(std, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor

def add_coords(tensor):
    x, y = torch.arange(0, tensor.shape[1], device=tensor.device) / tensor.shape[1], torch.arange(0, tensor.shape[2], device=tensor.device) / tensor.shape[2]
    xx, yy = meshgrid([x, y])
    coords = torch.stack([xx, yy], dim=0) - 0.5
    tensor = torch.cat([tensor, coords], dim=0)
    return tensor

def get_strip_scores(image):

    global model
    if model == None:
        dir = "/".join(__file__.split("/")[:-1])
        model = torch.jit.load(f"{dir}/models/kernel_3_8.pt")
    kernel_size = 7
    strip_count = 16

    new_image = add_coords(normalise(image / 255, norm_means, norm_stds))
    
    image_height = new_image.shape[1]
    strip_heigths = [int(1 + (image_height - 2) / (1.0 + math.exp(-(i - strip_count / 2.0 + 0.5)/(strip_count / 8.0)))) for i in range(strip_count)]

    half_kernel = int(kernel_size // 2)
    strips = torch.stack([new_image[:, y-half_kernel:y+half_kernel+1, :] for y in strip_heigths], dim=0)
    edges = torch.sigmoid(model(strips)).squeeze()
    edges = pad(edges, [half_kernel, half_kernel, 0, 0])

    return edges
