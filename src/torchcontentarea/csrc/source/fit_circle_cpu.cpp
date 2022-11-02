#include <cmath>
#include "../common.hpp"

// =========================================================================
// General functionality...

namespace cpu
{

    int fast_rand(int& seed)
    { 
        seed = 214013 * seed + 2531011; 
        return (seed >> 16) & 0x7FFF; 
    }

    void rand_triplet(int seed, int seed_stride, int max, int* triplet) 
    {
        triplet[0] = fast_rand(seed) % max;
        do {triplet[1] = fast_rand(seed) % max;} while (triplet[1] == triplet[0]);
        do {triplet[2] = fast_rand(seed) % max;} while (triplet[2] == triplet[0] || triplet[2] == triplet[1]);
    }

    bool check_circle(float x, float y, float r, int image_width, int image_height)
    {
        float x_diff = x - 0.5 * image_width;
        float y_diff = y - 0.5 * image_height;
        float diff = sqrt(x_diff * x_diff + y_diff * y_diff);

        bool valid = true;
        valid &= diff < MAX_CENTER_DIST * image_width;
        valid &= r > MIN_RADIUS * image_width;
        valid &= r < MAX_RADIUS * image_width;
        
        return valid;
    }

    bool calculate_circle(float ax, float ay, float bx, float by, float cx, float cy, float* x, float* y, float* r)
    {
        float offset = bx * bx + by * by;

        float bc = 0.5f * (ax * ax + ay * ay - offset);
        float cd = 0.5f * (offset - cx * cx - cy * cy);

        float det = (ax - bx) * (by - cy) - (bx - cx) * (ay - by);

        bool valid = abs(det) > 1e-8; 

        if (valid)
        {
            float idet = 1.0f / det;

            *x = (bc * (by - cy) - cd * (ay - by)) * idet;
            *y = (cd * (ax - bx) - bc * (bx - cx)) * idet;
            *r = sqrt((bx - *x) * (bx - *x) + (by - *y) * (by - *y));
        }

        return valid;
    }

    bool Cholesky3x3(float lhs[3][3], float rhs[3])
    {
        float sum;
        float diagonal[3];

        sum = lhs[0][0];

        if (sum <= 0.f) 
            return false;

        diagonal[0] = sqrt(sum);

        sum = lhs[0][1];
        lhs[1][0] = sum / diagonal[0];

        sum = lhs[0][2];
        lhs[2][0] = sum / diagonal[0];

        sum = lhs[1][1] - lhs[1][0] * lhs[1][0];

        if (sum <= 0.f) 
            return false;

        diagonal[1] = sqrt(sum);

        sum = lhs[1][2] - lhs[1][0] * lhs[2][0];
        lhs[2][1] = sum / diagonal[1];

        sum = lhs[2][2] - lhs[2][1] * lhs[2][1] - lhs[2][0] * lhs[2][0];

        if (sum <= 0.f)
            return false;

        diagonal[2] = sqrt(sum);

        sum = rhs[0];
        rhs[0] = sum / diagonal[0];

        sum = rhs[1] - lhs[1][0] * rhs[0];
        rhs[1] = sum / diagonal[1];

        sum = rhs[2] - lhs[2][1] * rhs[1] - lhs[2][0] * rhs[0];
        rhs[2] = sum / diagonal[2];

        sum = rhs[2];
        rhs[2] = sum / diagonal[2];

        sum = rhs[1] - lhs[2][1] * rhs[2];
        rhs[1] = sum / diagonal[1];

        sum = rhs[0] - lhs[1][0] * rhs[1] - lhs[2][0] * rhs[2];
        rhs[0] = sum / diagonal[0];

        return true;
    }

    void get_circle(int point_count, int* indices, int* points_x, int* points_y, float* circle_x, float* circle_y, float* circle_r)
    {
        if (point_count == 3)
        {
            int a = indices[0];
            int b = indices[1];
            int c = indices[2];

            calculate_circle(points_x[a], points_y[a], points_x[b], points_y[b], points_x[c], points_y[c], circle_x, circle_y, circle_r);
        }
        else
        {

            float lhs[3][3] {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            float rhs[3] {0, 0, 0};

            for (int i = 0; i < point_count; i++)
            {
                float p_x = points_x[indices[i]];
                float p_y = points_y[indices[i]];

                lhs[0][0] += p_x * p_x;
                lhs[0][1] += p_x * p_y;
                lhs[1][1] += p_y * p_y;
                lhs[0][2] += p_x;
                lhs[1][2] += p_y;
                lhs[2][2] += 1;

                rhs[0] += p_x * p_x * p_x + p_x * p_y * p_y;
                rhs[1] += p_x * p_x * p_y + p_y * p_y * p_y;
                rhs[2] += p_x * p_x + p_y * p_y;
            }

            Cholesky3x3(lhs, rhs);

            float A=rhs[0], B=rhs[1], C=rhs[2];

            *circle_x = A / 2.0f;
            *circle_y = B / 2.0f;
            *circle_r = std::sqrt(4.0f * C + A * A + B * B) / 2.0f;
        }
    }
        
    // =========================================================================
    // Main function...

    void fit_circle(const int* points_x, const int* points_y, const float* points_score, const int point_count, const ConfidenceThresholds confidence_thresholds, const int image_height, const int image_width, float* results)
    {
        int* compacted_points = (int*)malloc(3 * point_count * sizeof(int));
        int* compacted_points_x = compacted_points + 0 * point_count;
        int* compacted_points_y = compacted_points + 1 * point_count;
        float* compacted_points_s = (float*)compacted_points + 2 * point_count;

        // Point compaction...
        int real_point_count = 0;
        for (int i = 0; i < point_count; ++i)
        {
            if (points_score[i] > confidence_thresholds.edge)
            {
                compacted_points_x[real_point_count] = points_x[i]; 
                compacted_points_y[real_point_count] = points_y[i]; 
                compacted_points_s[real_point_count] = points_score[i]; 
                real_point_count += 1;
            }
        }
    
        results[0] = 0.0f;
        results[1] = 0.0f;
        results[2] = 0.0f;
        results[3] = 0.0f;

        // Early out...
        if (real_point_count < 3)
        {
            return;
        }

        // Ransac attempts...
        for (int ransac_attempt = 0; ransac_attempt < RANSAC_ATTEMPTS; ++ransac_attempt)
        {
            int inlier_count = 3;
            int inliers[MAX_POINT_COUNT];
            rand_triplet(ransac_attempt * 42342, RANSAC_ATTEMPTS, real_point_count, inliers);

            float circle_x, circle_y, circle_r;
            float circle_score = 0.0f;

            for (int i = 0; i < RANSAC_ITERATIONS; i++)
            {
                get_circle(inlier_count, inliers, compacted_points_x, compacted_points_y, &circle_x, &circle_y, &circle_r);
                
                inlier_count = 0;
                circle_score = 0.0f;

                for (int point_index = 0; point_index < real_point_count; point_index++)
                {
                    int edge_x = compacted_points_x[point_index];
                    int edge_y = compacted_points_y[point_index];
                    float edge_score = compacted_points_s[point_index];

                    float delta_x = circle_x - edge_x;
                    float delta_y = circle_y - edge_y;

                    float delta = std::sqrt(delta_x * delta_x + delta_y * delta_y);
                    float error = std::abs(circle_r - delta);

                    if (error < RANSAC_INLIER_THRESHOLD)
                    {
                        circle_score += edge_score;

                        inliers[inlier_count] = point_index;
                        inlier_count++;
                    }
                }

                circle_score /= point_count;
            }

            bool circle_valid = check_circle(circle_x, circle_y, circle_r, image_width, image_height);

            if (circle_valid && circle_score > results[3])
            {
                results[0] = circle_x;
                results[1] = circle_y;
                results[2] = circle_r;
                results[3] = circle_score;
            }
        }
        
        free(compacted_points);
    }
}
