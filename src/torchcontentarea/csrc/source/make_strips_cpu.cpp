#include <cmath>
#include "../common.hpp"

namespace cpu
{
    void make_strips(const uint8* image, const int image_height, const int image_width, const int strip_count, const int strip_width, float* strips)
    {
        for (int strip_index = 0; strip_index < strip_count; ++strip_index)
        {
            int strip_height = 1 + (image_height - 2) / (1.0f + std::exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));
            int strip_offset = strip_index * 5 * image_width * strip_width;

            for (int image_x = 0; image_x < image_width; ++image_x)
            {
                for (int strip_y = 0; strip_y < strip_width; ++strip_y)
                {
                    int image_y = strip_height + strip_y - (strip_width - 1) / 2;

                    int image_pixel_index = image_x + image_y * image_width;
                    int strip_pixel_index = strip_offset + image_x + strip_y * image_width;

                    float r = (image[image_pixel_index + 0 * image_width * image_height]/255.0f - 0.3441f) / 0.2381f;
                    float g = (image[image_pixel_index + 1 * image_width * image_height]/255.0f - 0.2251f) / 0.1994f;
                    float b = (image[image_pixel_index + 2 * image_width * image_height]/255.0f - 0.2203f) / 0.1939f;
                    float x = ((float)image_x / image_width) - 0.5f;
                    float y = ((float)image_y / image_height) - 0.5f;

                    strips[strip_pixel_index + 0 * image_width * strip_width] = r;
                    strips[strip_pixel_index + 1 * image_width * strip_width] = g;
                    strips[strip_pixel_index + 2 * image_width * strip_width] = b;
                    strips[strip_pixel_index + 3 * image_width * strip_width] = x;
                    strips[strip_pixel_index + 4 * image_width * strip_width] = y;
                }
            }
        }
    }
}
