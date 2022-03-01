#include <cuda_runtime.h>
#include "content_area_inference.cuh"

#define BLOCK_SIZE 32
#define GRID_SIZE(d) ((d / BLOCK_SIZE) + 1)

__global__ void draw_circle(const ContentArea area, uint8* mask, const uint mask_height, const uint mask_width)
{
    uint mask_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint mask_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (mask_x >= mask_width || mask_y >= mask_height)
    {
        return;
    }

    int x_diff = mask_x - area.x;
    int y_diff = mask_y - area.y;

    int diff = x_diff * x_diff + y_diff * y_diff;

    bool in_area = diff < area.r * area.r;

    mask[mask_x + mask_y * mask_width] = in_area ? 1 : 0;
}

void ContentAreaInference::draw_area(const ContentArea area, uint8* mask, const uint mask_height, const uint mask_width)
{
    dim3 grid(GRID_SIZE(mask_width), GRID_SIZE(mask_height));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    switch(area.type)
    {
        case(ContentAreaType::None): break;
        case(ContentAreaType::Circle): draw_circle<<<grid, block>>>(area, mask, mask_height, mask_width); break;
    }
}
