#include <cuda_runtime.h>
#include "content_area_inference.cuh"

#include <torch/extension.h>

#define BLOCK_SIZE 32
#define GRID_SIZE(d) ((d / BLOCK_SIZE) + 1)

__global__ void crop_resize_nearest(const uint8* src_image, uint8* dst_image, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, const int x, const int y, const int w, const int h)
{
    uint dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
    {
        return;
    }

    uint src_x = round(x + w * float(dst_x) / dst_width);
    uint src_y = round(y + h * float(dst_y) / dst_height);

    uint8 src_r = src_image[src_x + src_y * src_width + 0 * src_width * src_height];
    uint8 src_g = src_image[src_x + src_y * src_width + 1 * src_width * src_height];
    uint8 src_b = src_image[src_x + src_y * src_width + 2 * src_width * src_height];

    dst_image[dst_x + dst_y * dst_width + 0 * dst_width * dst_height] = src_r;
    dst_image[dst_x + dst_y * dst_width + 1 * dst_width * dst_height] = src_g;
    dst_image[dst_x + dst_y * dst_width + 2 * dst_width * dst_height] = src_b;
}

__global__ void crop_resize_bilinear(const uint8* src_image, uint8* dst_image, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, const int x, const int y, const int w, const int h)
{
    uint dst_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint dst_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (dst_x >= dst_width || dst_y >= dst_height)
    {
        return;
    }

    float src_x = x + w * float(dst_x) / dst_width;
    float src_y = y + h * float(dst_y) / dst_height;

    int x_lower = floor(src_x);
    float x_remainder = src_x - x_lower;

    int y_lower = floor(src_y);
    float y_remainder = src_y - y_lower;

    int coords_x[4] {x_lower, x_lower, x_lower+1, x_lower+1};
    int coords_y[4] {y_lower, y_lower+1, y_lower, y_lower+1};

    float coeffs[4]
    {
        (1 - x_remainder) * (1 - y_remainder),
        (1 - x_remainder) * y_remainder,
        x_remainder * (1 - y_remainder),
        x_remainder * y_remainder,
    };

    uint8 color[3] {0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int c = 0; c < 3; c++)
        {
            color[c] += coeffs[i] * src_image[coords_x[i] + coords_y[i] * src_width + c * src_width * src_height];
        }
    }
    
    #pragma unroll
    for (int c = 0; c < 3; c++)
    {
        dst_image[dst_x + dst_y * dst_width + c * dst_width * dst_height] = color[c];
    }
}

void calculate_optimal_crop(const ContentArea area, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, int* x, int* y, int* w, int* h)
{
    float aspect_ratio = float(dst_width) / dst_height;

    int inscribed_height = 2 * area.r / sqrt(1 + aspect_ratio * aspect_ratio);
    int inscribed_width = inscribed_height * aspect_ratio;

    int left = max(int(area.x) - inscribed_width / 2, 0);
    int right = min(int(area.x) + inscribed_width / 2, src_width);
    int top = max(int(area.y) - inscribed_height / 2, 0);
    int bottom = min(int(area.y) + inscribed_height / 2, src_height);

    float x_scale = (right - left);
    float y_scale = (bottom - top) * aspect_ratio;

    float scale = min(x_scale, y_scale);

    *w = (int)floor(scale);
    *h = (int)floor(scale / aspect_ratio);
    
    *x = (int)(left + (right - left) / 2 - *w / 2);
    *y = (int)(top + (bottom - top) / 2 - *h / 2);
}

void ContentAreaInference::crop_area(const ContentArea area, const uint8* src_image, uint8* dst_image, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, const InterpolationMode interpolation_mode)
{
    // Finding crop to perform...
    int x, y, w, h;
    switch(area.type)
    {
        case(ContentAreaType::None):
            x = 0; y = 0; w = src_width; h = src_height;
            break;
        case(ContentAreaType::Circle): 
            calculate_optimal_crop(area, src_width, src_height, dst_width, dst_height, &x, &y, &w, &h);
            break;
    }
    
    // Performing crop...
    dim3 grid(GRID_SIZE(dst_width), GRID_SIZE(dst_height));
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    switch(interpolation_mode)
    {
        case(InterpolationMode::Nearest):
            crop_resize_nearest<<<grid, block>>>(src_image, dst_image, src_width, src_height, dst_width, dst_height, x, y, w, h); 
            break;
        case(InterpolationMode::Bilinear): 
            crop_resize_bilinear<<<grid, block>>>(src_image, dst_image, src_width, src_height, dst_width, dst_height, x, y, w, h); 
            break;
        case(InterpolationMode::Bicubic): 
            // crop_resize_bicubic<<<grid, block>>>(src_image, dst_image, src_width, src_height, dst_width, dst_height, x, y, w, h); 
            break;
    }
}
