#include <cuda_runtime.h>

#define MAX_POINT_COUNT 32
#define INVALID_POINT -1

typedef unsigned char uint8;

void make_strips(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const uint strip_width, float* strips);

void find_points(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, uint* points_x, uint* points_y, float* point_score);

void find_points_from_strip_scores(const float* strips, const uint image_width, const uint image_height, const uint strip_count, const uint model_patch_size, uint* points_x, uint* points_y, float* point_score);

void fit_circle(const uint* points_x, const uint* points_y, const float* points_score, const uint point_count, const uint image_height, const uint image_width, float* results);
