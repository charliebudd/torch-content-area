#define MAX_POINT_COUNT 32
#define INVALID_POINT -1

typedef unsigned int uint;
typedef unsigned char uint8;

void find_points_cpu(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, uint* points_x, uint* points_y, float* point_score);

void make_strips_cpu(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const uint strip_width, float* strips);
void find_points_from_strip_scores_cpu(const float* strips, const uint image_width, const uint image_height, const uint strip_count, const uint model_patch_size, uint* points_x, uint* points_y, float* point_score);

void fit_circle_cpu(uint* points_x, uint* points_y, float* points_score, const uint point_count, const uint image_height, const uint image_width, float* results);
