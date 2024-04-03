import torch
from typing import Sequence, Optional
    
DEG2RAD=0.01745329251
RAD2DEG=1.0/DEG2RAD

GRAY_SCALE_WEIGHTS = torch.tensor([
    0.2126, 0.7152, 0.0722
])

SOBEL_KERNEL = torch.tensor([
    [[
        [-0.25, 0.0, 0.25],
        [-0.5, 0.0, 0.5],
        [-0.25, 0.0, 0.25],
    ]],
    [[
        [-0.25, -0.5, -0.25],
        [0.0, 0.0, 0.0],
        [0.25, 0.5, 0.25],
    ]],
])

def get_points_handcrafted(image: torch.Tensor, strip_count: int=16, feature_thresholds: Sequence[float]=(20, 30, 25)) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using handcrafted feature extraction.
    """
    
    batched = len(image.shape) == 4
    device = image.device
    if not batched:
        image.unsqueeze(0)
    if image.dtype.is_floating_point:
        image = image * 255.0
    if image.size(1) != 1:
        image = (image * GRAY_SCALE_WEIGHTS[None, :, None, None]).sum(dim=1)
    image = image.float()
    B, H, W = image.shape

    strip_indices = torch.arange(strip_count, device=device)
    strip_heights = (1 + (H - 2) / (1.0 + torch.exp(-(strip_indices - strip_count / 2.0 + 0.5) / (strip_count / 8.0)))).long()
    
    indices = torch.cat([strip_heights-1, strip_heights, strip_heights+1])
    strips = image[:, indices, :].reshape(B, 3, strip_count, W).permute(0, 2, 1, 3)
    strips = torch.cat([strips[..., :W//2], strips.flip(-1)[..., :W//2]], dim=1)
    
    ys = torch.cat([strip_heights, strip_heights])[None, :, None].repeat(B, 1, W//2-2)
    xs = (torch.arange(W//2-2) + 1)[None, None, :].repeat(B, strip_count, 1)
    xs = torch.cat([xs, W - 1 - xs], dim=1)
    
    grad = torch.conv2d(strips.reshape(B*2*strip_count, 1, 3, -1), SOBEL_KERNEL.to(device)).reshape(B, 2*strip_count, 2, -1)
    strips = strips[:, :, 1, 1:-1]
    
    grad_x = grad[:, :, 0]
    grad_y = grad[:, :, 1]
    grad_x[:, strip_count:] *= -1
    
    grad = torch.sqrt(grad_x**2 + grad_y**2)
    
    max_preceeding_intensity = torch.cummax(strips, dim=-1)[0]
    
    center_dir_x = (0.5 * W) - xs
    center_dir_y = (0.5 * H) - ys
    center_dir_norm = torch.sqrt(center_dir_x * center_dir_x + center_dir_y * center_dir_y)

    dot = torch.where(grad == 0, -1, (center_dir_x * grad_x + center_dir_y * grad_y) / (center_dir_norm * grad))
    dot = torch.clamp(dot, -0.99, 0.99)
    angle = RAD2DEG * torch.acos(dot)
    
    edge_score = torch.tanh(grad / feature_thresholds[0])
    angle_score = 1.0 - torch.tanh(angle / feature_thresholds[1])
    intensity_score = 1.0 - torch.tanh(max_preceeding_intensity / feature_thresholds[2])
    
    
    point_scores = edge_score * angle_score * intensity_score
    
    point_scores, indices = torch.max(point_scores, dim=-1)
    
    point_scores = point_scores.squeeze()
    ys = torch.gather(ys, 2, indices[:, :, None]).squeeze()
    xs = torch.gather(xs, 2, indices[:, :, None]).squeeze()
    
    result = torch.stack([xs, ys, point_scores], dim=1).reshape(B, strip_count*2, 3).permute(0, 2, 1)
    
    if not batched:
        result = result[0]
    
    return result

def get_points_learned(image: torch.Tensor, strip_count: int=16, model: Optional[torch.jit.ScriptModule]=None,  model_patch_size: int=7) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using learned feature extraction.
    """
    raise NotImplementedError("The learned method is not implemented in the python fallback. The handcrafted method should provide better performance, but to use the learned method you will need to install the compiled extension.")
