#pragma once
#include <cuda_runtime.h>
#include "../infer_area.h"

void find_points(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const FeatureThresholds feature_thresholds, uint* points_x, uint* points_y, float* point_score);

void make_strips(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const uint strip_width, float* strips);
void find_points_from_strip_scores(const float* strips, const uint image_width, const uint image_height, const uint strip_count, const uint model_patch_size, uint* points_x, uint* points_y, float* point_score);

void fit_circle(const uint* points_x, const uint* points_y, const float* points_score, const uint point_count, const ConfidenceThresholds confidence_thresholds, const uint image_height, const uint image_width, float* results);
