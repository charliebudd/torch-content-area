#include <cuda_runtime.h>
#include "../common.hpp"

namespace cuda
{
    __device__ int get_strip_height(int strip_index, int strip_count, int image_height)
    {
        return 1 + (image_height - 2) / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f) / (strip_count / 8.0f)));
    }

    template<int c, typename T>
    __global__ void make_strips_kernel(const T* g_image_batch, const int image_width, const int image_height, const int strip_count, const int strip_width, float* g_strips_batch)
    {
        const T* g_image = g_image_batch + blockIdx.z * c * image_width * image_height;
        float* g_strips = g_strips_batch + blockIdx.z * strip_count * 5 * image_width * strip_width;

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

        float r, g, b;

        float norm = std::is_floating_point<T>::value ? 1.0f : 255.0f;

        if (c == 3)
        {
            r = (g_image[image_pixel_index + 0 * image_width * image_height]/norm - 0.3441f) / 0.2381f;
            g = (g_image[image_pixel_index + 1 * image_width * image_height]/norm - 0.2251f) / 0.1994f;
            b = (g_image[image_pixel_index + 2 * image_width * image_height]/norm - 0.2203f) / 0.1939f;
        }
        else
        {
            r = (g_image[image_pixel_index]/norm - 0.3441f) / 0.2381f;
            g = (g_image[image_pixel_index]/norm - 0.2251f) / 0.1994f;
            b = (g_image[image_pixel_index]/norm - 0.2203f) / 0.1939f;
        }

        float x = ((float)image_x / image_width) - 0.5f;
        float y = ((float)image_y / image_height) - 0.5f;

        g_strips[strip_pixel_index + 0 * image_width * strip_width] = r;
        g_strips[strip_pixel_index + 1 * image_width * strip_width] = g;
        g_strips[strip_pixel_index + 2 * image_width * strip_width] = b;
        g_strips[strip_pixel_index + 3 * image_width * strip_width] = x;
        g_strips[strip_pixel_index + 4 * image_width * strip_width] = y;
    }

    void make_strips(Image image, const int batch_count, const int channel_count, const int image_height, const int image_width, const int strip_count, const int strip_width, float* strips)
    {
        dim3 grid(((image_width - 1) / 128) + 1, strip_count, batch_count);
        dim3 block(128, strip_width);
        KERNEL_DISPATCH_IMAGE_FORMAT(make_strips_kernel, ARG(grid, block), image, image_width, image_height, strip_count, strip_width, strips);
    }
}
