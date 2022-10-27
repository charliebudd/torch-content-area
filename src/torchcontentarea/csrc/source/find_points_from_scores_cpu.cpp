#include <math.h>
#include "../common.hpp"

namespace cpu
{
    void find_points_from_strip_scores(const float* strips, const int image_height, const int image_width, const int strip_count, const int model_patch_size, int* points_x, int* points_y, float* point_score)
    {
        int half_patch_size = model_patch_size / 2;
        int strip_width = image_width - 2 * half_patch_size;

        for (int strip_index = 0; strip_index < strip_count; ++strip_index)
        {
            int image_y = 1 + (image_height - 2) / (1.0f + std::exp(-(strip_index - strip_count / 2.0f + 0.5f) / (strip_count / 8.0f)));

            float best_score = 0.0f;
            int best_index = 0;

            for (int strip_x = 0; strip_x < strip_width / 2; ++strip_x)
            {
                float point_score = strips[strip_x + strip_index * strip_width];

                if (point_score > best_score)
                {
                    best_score = point_score;
                    best_index = strip_x + half_patch_size;
                }
            }

            points_x[strip_index * 2] = best_index;
            points_y[strip_index * 2] = image_y;
            point_score[strip_index * 2] = best_score;
            
            best_score = 0.0f;
            best_index = 0;

            for (int strip_x = strip_width / 2; strip_x < strip_width; ++strip_x)
            {
                float point_score = strips[strip_x + strip_index * strip_width];

                if (point_score > best_score)
                {
                    best_score = point_score;
                    best_index = strip_x + half_patch_size;
                }
            }

            points_x[1 + strip_index * 2] = best_index;
            points_y[1 + strip_index * 2] = image_y;
            point_score[1 + strip_index * 2] = best_score;
        }
    }
}