#pragma once
#include "common.hpp"

namespace cpu
{
    void find_points(const uint8* image, const int batch_count, const int channel_count, const int image_height, const int image_width, const int strip_count, const FeatureThresholds feature_thresholds, float* points_x, float* points_y, float* point_score);
    void make_strips(const uint8* image, const int batch_count, const int image_height, const int image_width, const int strip_count, const int strip_width, float* strips);
    void find_points_from_strip_scores(const float* strips, const int batch_count, const int image_width, const int image_height, const int strip_count, const int model_patch_size, float* points_x, float* points_y, float* point_score);
    void fit_circle(const float* points_x, const float* points_y, const float* points_score, const int batch_count, const int point_count, const ConfidenceThresholds confidence_thresholds, const int image_height, const int image_width, float* results);
}
