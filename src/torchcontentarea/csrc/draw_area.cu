#include <cuda_runtime.h>
#include "content_area_inference.cuh"

#define BLOCK_SIZE 32
#define GRID_SIZE(d) ((d / BLOCK_SIZE) + 1)

__global__ void draw_circle(Area area, uint8* mask, const uint mask_height, const uint mask_width)
{
    uint mask_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint mask_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (mask_x > mask_width || mask_y > mask_height)
    {
        return;
    }

    int x_diff = mask_x - area.circle.x;
    int y_diff = mask_y - area.circle.y;

    int diff = x_diff * x_diff + y_diff * y_diff;

    bool in_area = diff < area.circle.r * area.circle.r;

    mask[mask_x + mask_y * mask_width] = in_area ? 1 : 0;
}


__global__ void draw_rectangle(Area area, uint8* mask, const uint mask_height, const uint mask_width)
{
    uint mask_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint mask_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (mask_x > mask_width || mask_y > mask_height)
    {
        return;
    }

    bool in_area = true;

    in_area &= mask_x > area.rectangle.x && mask_x < area.rectangle.x + area.rectangle.w; 
    in_area &= mask_y > area.rectangle.y && mask_y < area.rectangle.y + area.rectangle.h; 

    mask[mask_x + mask_y * mask_width] = in_area ? 1 : 0;
}


void ContentAreaInference::draw_area(Area area, uint8* mask, const uint mask_height, const uint mask_width)
{
    dim3 grid(GRID_SIZE(mask_width), GRID_SIZE(mask_height));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    switch(area.type)
    {
        case(Area::Circle): draw_circle<<<grid, block>>>(area, mask, mask_height, mask_width); break;
        case(Area::Rectangle): draw_rectangle<<<grid, block>>>(area, mask, mask_height, mask_width); break;
        case(Area::None): break;
    }
}
