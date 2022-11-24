#include <cmath>
#include "../common.hpp"

namespace cpu
{
    void find_points_from_strip_scores(const float* strips, const int batch_count, const int image_height, const int image_width, const int strip_count, const int model_patch_size, float* points_x, float* points_y, float* point_score)
    {
        for (int batch_index = 0; batch_index < batch_count; ++batch_index)
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
                    float point_score = strips[strip_x + strip_index * strip_width + batch_index * strip_count * strip_width];

                    if (point_score > best_score)
                    {
                        best_score = point_score;
                        best_index = strip_x + half_patch_size;
                    }
                }

                points_x[strip_index + batch_index * 3 * 2 * strip_count] = best_index;
                points_y[strip_index + batch_index * 3 * 2 * strip_count] = image_y;
                point_score[strip_index + batch_index * 3 * 2 * strip_count] = best_score;
                
                best_score = 0.0f;
                best_index = 0;

                for (int strip_x = strip_width / 2; strip_x < strip_width; ++strip_x)
                {
                    float point_score = strips[strip_x + strip_index * strip_width + batch_index * strip_count * strip_width];

                    if (point_score > best_score)
                    {
                        best_score = point_score;
                        best_index = strip_x + half_patch_size;
                    }
                }

                points_x[strip_index + strip_count + batch_index * 3 * 2 * strip_count] = best_index;
                points_y[strip_index + strip_count + batch_index * 3 * 2 * strip_count] = image_y;
                point_score[strip_index + strip_count + batch_index * 3 * 2 * strip_count] = best_score;
            }
        }
    }
}
