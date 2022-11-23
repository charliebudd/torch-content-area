#include <cmath>
#include "../common.hpp"

namespace cpu
{
   // =========================================================================
    // General functionality...

    template<int c, typename T>
    float load_element(const T* data, const int index, const int color_stride)
    {
        float value = c == 1 ? data[index] : 0.2126f * data[index + 0 * color_stride] + 0.7152f * data[index + 1 * color_stride] + 0.0722f * data[index + 2 * color_stride];
        return std::is_floating_point<T>::value ? 255 * value : value;
    }

    template<int c, typename T>
    float load_sobel_strip(const T* data, const int index, const int spatial_stride, const int color_stride)
    {
        return  0.25 * load_element<c, T>(data, index - spatial_stride, color_stride) + 0.5 * load_element<c, T>(data, index, color_stride) + 0.25 * load_element<c, T>(data, index + spatial_stride, color_stride);
    }

    // =========================================================================
    // Main function...

    template<int c, typename T>
    void find_points(const T* images, const int batch_count, const int image_height, const int image_width, const int strip_count, FeatureThresholds feature_thresholds, float* points_x, float* points_y, float* point_scores)
    {
        for (int batch_index = 0; batch_index < batch_count; ++batch_index)
        {
            const T* image = images + batch_index * c * image_height * image_width;

            for (int strip_index = 0; strip_index < strip_count; ++strip_index)
            {
                int image_y = 1 + (image_height - 2) / (1.0f + std::exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));

                for (int point_index = 0; point_index < 2; ++point_index)
                {
                    bool flip = point_index > 0;

                    float max_preceeding_intensity = 0.0f;
                    float best_score = 0.0f;
                    int best_index = 0;

                    for (int x = 1; x < image_width / 2; ++x)
                    {
                        int image_x = flip ? image_width - 1 - x : x;

                        float intensity = c == 3 ? load_element<c, T>(image, image_x + image_y * image_width, image_width * image_height) : image[image_x + image_y * image_width];
                        max_preceeding_intensity = max_preceeding_intensity < intensity ? intensity : max_preceeding_intensity;

                        float left  = load_sobel_strip<c, T>(image, (image_x - 1) + image_y * image_width, image_width, image_width * image_height);
                        float right = load_sobel_strip<c, T>(image, (image_x + 1) + image_y * image_width, image_width, image_width * image_height);
                        float top = load_sobel_strip<c, T>(image, image_x + (image_y - 1) * image_width, 1, image_width * image_height);
                        float bot = load_sobel_strip<c, T>(image, image_x + (image_y + 1) * image_width, 1, image_width * image_height);

                        float grad_x = right - left;
                        float grad_y = bot - top;
                        float grad = sqrt(grad_x * grad_x + grad_y * grad_y);

                        float center_dir_x = (0.5f * image_width) - (float)image_x;
                        float center_dir_y = (0.5f * image_height) - (float)image_y;
                        float center_dir_norm = sqrt(center_dir_x * center_dir_x + center_dir_y * center_dir_y);

                        float dot = grad == 0 ? -1 : (center_dir_x * grad_x + center_dir_y * grad_y) / (center_dir_norm * grad);
                        float angle = RAD2DEG * acos(dot);

                        // ============================================================
                        // Final scoring...

                        float edge_score = tanh(grad / feature_thresholds.edge);
                        float angle_score = 1.0f - tanh(angle / feature_thresholds.angle);
                        float intensity_score = 1.0f - tanh(max_preceeding_intensity / feature_thresholds.intensity);

                        float point_score = edge_score * angle_score * intensity_score;

                        if (point_score > best_score)
                        {
                            best_score = point_score;
                            best_index = image_x;
                        }
                    }

                    if (best_index < DISCARD_BORDER || best_index >= image_width - DISCARD_BORDER)
                    {
                        best_score = 0.0f;
                    }

                    points_x[strip_index + point_index * strip_count + batch_index * 3 * 2 * strip_count] = best_index;
                    points_y[strip_index + point_index * strip_count + batch_index * 3 * 2 * strip_count] = image_y;
                    point_scores[strip_index + point_index * strip_count + batch_index * 3 * 2 * strip_count] = best_score;
                }
            }
        }
    }

    void find_points(Image image, const int batch_count, const int channel_count, const int image_height, const int image_width, const int strip_count, FeatureThresholds feature_thresholds, float* points_x, float* points_y, float* point_scores)
    {
        FUNCTION_CALL_IMAGE_FORMAT(find_points, image, batch_count, image_height, image_width, strip_count, feature_thresholds, points_x, points_y, point_scores);
    }
} 
