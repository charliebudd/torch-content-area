#include "infer_area_cuda.cuh"

__device__ int get_strip_height(int strip_index, int strip_count, int image_height)
{
    return 1 + (image_height - 2) / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f) / (strip_count / 8.0f)));
}

__global__ void create_strips(const uint8* g_image, const uint image_width, const uint image_height, const uint strip_count, const uint strip_width, float* g_strips)
{
    int strip_index = blockIdx.y;
    int strip_offset = strip_index * 5 * image_width * strip_width;
    int strip_height = get_strip_height(strip_index, strip_count, image_height);

    int image_x = threadIdx.x + blockIdx.x * blockDim.x;
    int strip_y = threadIdx.y;

    if (image_x >= image_width)
        return;

    int image_y = strip_height + strip_y - (strip_width - 1) / 2;

    int image_pixel_index = image_x + image_y * image_width;
    int strip_pixel_index = strip_offset + image_x + strip_y * image_width;

    float r = (g_image[image_pixel_index + 0 * image_width * image_height]/255.0f - 0.3441f) / 0.2381f;
    float g = (g_image[image_pixel_index + 1 * image_width * image_height]/255.0f - 0.2251f) / 0.1994f;
    float b = (g_image[image_pixel_index + 2 * image_width * image_height]/255.0f - 0.2203f) / 0.1939f;
    float x = ((float)image_x / image_width) - 0.5f;
    float y = ((float)image_y / image_height) - 0.5f;

    g_strips[strip_pixel_index + 0 * image_width * strip_width] = r;
    g_strips[strip_pixel_index + 1 * image_width * strip_width] = g;
    g_strips[strip_pixel_index + 2 * image_width * strip_width] = b;
    g_strips[strip_pixel_index + 3 * image_width * strip_width] = x;
    g_strips[strip_pixel_index + 4 * image_width * strip_width] = y;
}

void make_strips(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const uint strip_width, float* strips)
{
    dim3 grid((image_width - 1 / 128) + 1, strip_count);
    dim3 block(128, strip_width);
    create_strips<<<grid, block>>>(image, image_width, image_height, strip_count, strip_width, strips);
}
