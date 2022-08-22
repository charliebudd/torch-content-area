#include <math.h>
#include "infer_area_cpu.h"

#define DISCARD_BORDER 3
#define DEG2RAD 0.01745329251f
#define RAD2DEG (1.0f / DEG2RAD)

#define INTENSITY_THRESHOLD 25
#define EDGE_THRESHOLD 20
#define ANGLE_THRESHOLD 30
#define EDGE_CONFIDENCE_THRESHOLD 0.03

// =========================================================================
// General functionality...

float load_grayscale(const uint8* data, const uint index, const uint color_stride)
{
    return 0.2126f * data[index + 0 * color_stride] + 0.7152f * data[index + 1 * color_stride] + 0.0722f * data[index + 2 * color_stride];
}

float load_sobel_strip(const uint8* data, const uint index, const uint spatial_stride, const uint color_stride)
{
    return  0.25 * load_grayscale(data, index - spatial_stride, color_stride) + 0.5 * load_grayscale(data, index, color_stride) + 0.25 * load_grayscale(data, index + spatial_stride, color_stride);
}

// =========================================================================
// Main function...

void find_points_cpu(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, uint* points_x, uint* points_y, float* point_scores)
{
    float image_patch[3][3];

    for (int strip_index = 0; strip_index < strip_count; ++strip_index)
    {
        int image_y = 1 + (image_height - 2) / (1.0f + std::exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));

        float max_preceeding_intensity = 0.0f;
        float best_score = 0.0f;
        int best_index = 0;

        for (int image_x = 1; image_x < image_width / 2; ++image_x)
        {
            float intensity = load_grayscale(image, image_x + image_y * image_width, image_width * image_height);
            max_preceeding_intensity = max_preceeding_intensity < intensity ? intensity : max_preceeding_intensity;

            float left  = load_sobel_strip(image, (image_x - 1) + image_y * image_width, image_width, image_width * image_height);
            float right = load_sobel_strip(image, (image_x + 1) + image_y * image_width, image_width, image_width * image_height);
            float top = load_sobel_strip(image, image_x + (image_y - 1) * image_width, 1, image_width * image_height);
            float bot = load_sobel_strip(image, image_x + (image_y + 1) * image_width, 1, image_width * image_height);

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

            float edge_score = tanh(grad / EDGE_THRESHOLD);
            float angle_score = 1.0f - tanh(angle / ANGLE_THRESHOLD);
            float intensity_score = 1.0f - tanh(max_preceeding_intensity / INTENSITY_THRESHOLD);

            float point_score = edge_score * angle_score * intensity_score;

            if (point_score > best_score)
            {
                best_score = point_score;
                best_index = image_x;
            }
        }

        if (best_index <= DISCARD_BORDER || best_score < EDGE_CONFIDENCE_THRESHOLD)
        {
            best_index = INVALID_POINT;
            best_score = 0.0f;
        }

        points_x[2 * strip_index] = best_index;
        points_y[2 * strip_index] = image_y;
        point_scores[2 * strip_index] = best_score;

        max_preceeding_intensity = 0.0f;
        best_score = 0.0f;
        best_index = 0;

        for (int image_x = image_width - 2; image_x > image_width / 2; --image_x)
        {
            float intensity = load_grayscale(image, image_x + image_y * image_width, image_width * image_height);
            max_preceeding_intensity = max_preceeding_intensity < intensity ? intensity : max_preceeding_intensity;

            float left  = load_sobel_strip(image, (image_x - 1) + image_y * image_width, image_width, image_width * image_height);
            float right = load_sobel_strip(image, (image_x + 1) + image_y * image_width, image_width, image_width * image_height);
            float top = load_sobel_strip(image, image_x + (image_y - 1) * image_width, 1, image_width * image_height);
            float bot = load_sobel_strip(image, image_x + (image_y + 1) * image_width, 1, image_width * image_height);

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

            float edge_score = tanh(grad / EDGE_THRESHOLD);
            float angle_score = 1.0f - tanh(angle / ANGLE_THRESHOLD);
            float intensity_score = 1.0f - tanh(max_preceeding_intensity / INTENSITY_THRESHOLD);

            float point_score = edge_score * angle_score * intensity_score;

            if (point_score > best_score)
            {
                best_score = point_score;
                best_index = image_x;
            }
        }

        if (best_index >= image_width - 1 - DISCARD_BORDER || best_score < EDGE_CONFIDENCE_THRESHOLD)
        {
            best_index = INVALID_POINT;
            best_score = 0.0f;
        }

        points_x[1 + 2 * strip_index] = best_index;
        points_y[1 + 2 * strip_index] = image_y;
        point_scores[1 + 2 * strip_index] = best_score;
    }
}
