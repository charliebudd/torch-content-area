import torch
from math import sqrt
from typing import Sequence

MAX_CENTER_DIST=0.2
MIN_RADIUS=0.2
MAX_RADIUS=0.8
RANSAC_ATTEMPTS=32
RANSAC_ITERATIONS=3
RANSAC_INLIER_THRESHOLD=3

def check_circle(x, y, r, w, h):

    x_diff = x - 0.5 * w
    y_diff = y - 0.5 * h
    diff = sqrt(x_diff * x_diff + y_diff * y_diff)

    valid = True
    valid &= diff < MAX_CENTER_DIST * w
    valid &= r > MIN_RADIUS * w
    valid &= r < MAX_RADIUS * w
    
    return valid

def calculate_circle(points):
    
    ax, ay = points[:2, 0]
    bx, by = points[:2, 1]
    cx, cy = points[:2, 2]
    
    offset = bx * bx + by * by

    bc = 0.5 * (ax * ax + ay * ay - offset)
    cd = 0.5 * (offset - cx * cx - cy * cy)

    det = (ax - bx) * (by - cy) - (bx - cx) * (ay - by)
 
    if abs(det) > 1e-8:
        idet = 1.0 / det
        x = (bc * (by - cy) - cd * (ay - by)) * idet
        y = (cd * (ax - bx) - bc * (bx - cx)) * idet
        r = sqrt((bx - x) * (bx - x) + (by - y) * (by - y))
        return x, y, r
    else:
        return None

def get_circle(points):
    
    if points.size(1) == 3:
        return calculate_circle(points)
    else:
        lhs = torch.zeros(3, 3).to(points.device)
        rhs = torch.zeros(3).to(points.device)

        for p_x, p_y, _ in points.T:
            lhs[0, 0] += p_x * p_x
            lhs[0, 1] += p_x * p_y
            lhs[1, 1] += p_y * p_y
            lhs[0, 2] += p_x
            lhs[1, 2] += p_y
            lhs[2, 2] += 1

            rhs[0] += p_x * p_x * p_x + p_x * p_y * p_y
            rhs[1] += p_x * p_x * p_y + p_y * p_y * p_y
            rhs[2] += p_x * p_x + p_y * p_y

        lhs[1, 0] = lhs[0, 1]
        lhs[2, 0] = lhs[0, 2]
        lhs[2, 1] = lhs[1, 2]

        try:
            L = torch.linalg.cholesky(lhs)
            y = torch.linalg.solve(L, rhs)
            x = torch.linalg.solve(L.T, y)
        except:
            return None
            
        A, B, C = x[0], x[1], x[2]

        x = A / 2.0
        y = B / 2.0
        r = torch.sqrt(4.0 * C + A * A + B * B) / 2.0
        
        return x, y, r

def fit_area(points: torch.Tensor, image_size: Sequence[int], confidence_thresholds: Sequence[float]=(0.03, 0.06)) -> torch.Tensor:
    """
    Finds candidate edge points and corresponding scores in the given image(s) using handcrafted feature extraction.
    """

    batched = len(points.shape) == 3
    
    if not batched:
        points = points.unsqueeze(0)

    areas = []

    for point_batch in points:
        
        point_batch = point_batch[:, point_batch[2] > confidence_thresholds[0]]
        
        if point_batch.size(1) < 3:
            areas.append(torch.zeros(4))
            continue
        
        best_circle = torch.zeros(4)

        for _ in range(RANSAC_ATTEMPTS):
             
            indices = torch.randperm(point_batch.size(1))[:3]
            inliers = point_batch[:, indices]

            for _ in range(RANSAC_ITERATIONS):
                circle = get_circle(inliers)
                
                if circle is None:
                    x, y, r = 0, 0, 0
                    circle_score = 0
                    break
                
                x, y, r = circle
                
                dx = x - point_batch[0]
                dy = y - point_batch[1]
                
                error = torch.abs(torch.sqrt(dx**2 + dy**2) - r)
                
                inliers = point_batch[:, error < RANSAC_INLIER_THRESHOLD]
                circle_score = inliers[2].sum()
                
            circle_score /= point_batch.size(1)

            circle_valid = check_circle(x, y, r, image_size[1], image_size[0])

            if circle_valid and circle_score > best_circle[3]:
                best_circle = torch.tensor([x, y, r, circle_score])
                
        areas.append(best_circle)
                
    areas = torch.stack(areas)
    
    if not batched:
        areas = areas[0]

    return areas
