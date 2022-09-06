#include "../infer_area.h"

void find_points_cpu(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, FeatureThresholds feature_thresholds, uint* points_x, uint* points_y, float* point_score);

void make_strips_cpu(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const uint strip_width, float* strips);
void find_points_from_strip_scores_cpu(const float* strips, const uint image_width, const uint image_height, const uint strip_count, const uint model_patch_size, uint* points_x, uint* points_y, float* point_score);

void fit_circle_cpu(uint* points_x, uint* points_y, float* points_score, const uint point_count, ConfidenceThresholds confidence_thresholds, const uint image_height, const uint image_width, float* results);
