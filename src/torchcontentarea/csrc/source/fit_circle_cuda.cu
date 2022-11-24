#include <cuda_runtime.h>
#include "../common.hpp"

namespace cuda
{
    // =========================================================================
    // General functionality...

    __device__ int fast_rand(int& seed)
    { 
        seed = 214013 * seed + 2531011; 
        return (seed >> 16) & 0x7FFF; 
    }

    __device__ void rand_triplet(int seed, int seed_stride, int max, int* triplet) 
    {
        triplet[0] = fast_rand(seed) % max;
        do {triplet[1] = fast_rand(seed) % max;} while (triplet[1] == triplet[0]);
        do {triplet[2] = fast_rand(seed) % max;} while (triplet[2] == triplet[0] || triplet[2] == triplet[1]);
    }

    __device__ bool check_circle(float x, float y, float r, int image_width, int image_height)
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

    __device__ bool calculate_circle(float ax, float ay, float bx, float by, float cx, float cy, float* x, float* y, float* r)
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

    __device__ bool Cholesky3x3(float lhs[3][3], float rhs[3])
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

    __device__ void get_circle(int point_count, int* indices, int* points_x, int* points_y, float* circle_x, float* circle_y, float* circle_r)
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
            *circle_r = sqrt(4.0f * C + A * A + B * B) / 2.0f;
        }
    }

    // =========================================================================
    // Kernels...

    template<int warp_count>
    __global__ void fit_circle_kernel(const float* g_edge_x_batch, const float* g_edge_y_batch, const float* g_edge_scores_batch, float* g_circle_batch, const int point_count, const ConfidenceThresholds confidence_thresholds, const int image_height, const int image_width)
    {
        extern __shared__ int s_edge_info[];
        __shared__ int s_valid_point_count;
        __shared__ float s_score_reduction_buffer[warp_count];
        __shared__ float s_x_reduction_buffer[warp_count];
        __shared__ float s_y_reduction_buffer[warp_count];
        __shared__ float s_r_reduction_buffer[warp_count];

        const float* g_edge_x = g_edge_x_batch + blockIdx.x * 3 * point_count;
        const float* g_edge_y = g_edge_y_batch + blockIdx.x * 3 * point_count;
        const float* g_edge_scores = g_edge_scores_batch + blockIdx.x * 3 * point_count;
        float* g_circle = g_circle_batch + blockIdx.x * 4;

        int* s_edge_x = (int*)(s_edge_info + 0 * point_count);
        int* s_edge_y = (int*)(s_edge_info + 1 * point_count);
        float* s_edge_scores = (float*)(s_edge_info + 2 * point_count);

        const int warp_index = threadIdx.x >> 5;
        const int lane_index = threadIdx.x & 31;

        // Loading points to shared memory...
        if (threadIdx.x < point_count)
        {
            s_edge_x[threadIdx.x] = g_edge_x[threadIdx.x];
            s_edge_y[threadIdx.x] = g_edge_y[threadIdx.x];
            s_edge_scores[threadIdx.x] = g_edge_scores[threadIdx.x];
        }

        // Point compaction...
        bool has_point = threadIdx.x < point_count ? s_edge_scores[threadIdx.x] > confidence_thresholds.edge : false;
        int preceeding_count = has_point;

        #pragma unroll
        for (int d=1; d < 32; d<<=1) 
        {
            float other_count = __shfl_up_sync(0xffffffff, preceeding_count, d);

            if (lane_index >= d) 
            {
                preceeding_count += other_count;
            }
        }

        if (lane_index == 31)
        {
            s_score_reduction_buffer[warp_index] = preceeding_count;
        }

        __syncthreads();

        if (warp_index == 0)
        {
            int warp_sum = lane_index < warp_count ? s_score_reduction_buffer[lane_index] : 0;

            #pragma unroll
            for (int d=1; d < 32; d<<=1) 
            {
                float other_warp_sum = __shfl_up_sync(0xffffffff, warp_sum, d);

                if (lane_index >= d && other_warp_sum > warp_sum) 
                {
                    warp_sum = other_warp_sum;
                }
            }

            if (lane_index < warp_count)
            {
                s_score_reduction_buffer[lane_index] = warp_sum;
            }
        }

        __syncthreads();

        if (warp_index > 0)
        {
            preceeding_count +=  s_score_reduction_buffer[warp_index-1];
        }

        if (has_point)
        {
            s_edge_x[preceeding_count - 1] = s_edge_x[threadIdx.x];
            s_edge_y[preceeding_count - 1] = s_edge_y[threadIdx.x];
            s_edge_scores[preceeding_count - 1] = s_edge_scores[threadIdx.x];
        }

        if (threadIdx.x == blockDim.x - 1)
        {
            s_valid_point_count = preceeding_count;
        }

        __syncthreads();

        if (s_valid_point_count < 3)
        {   
            if (threadIdx.x < 4)
            {
                g_circle[threadIdx.x] = 0.0;
            }

            return;
        }

        int inlier_count = 3;
        int inliers[MAX_POINT_COUNT];
        rand_triplet(threadIdx.x * 42342, RANSAC_ATTEMPTS, s_valid_point_count, inliers);

        float circle_x, circle_y, circle_r;
        float circle_score = 0.0f;

        for (int i = 0; i < RANSAC_ITERATIONS; i++)
        {
            get_circle(inlier_count, inliers, s_edge_x, s_edge_y, &circle_x, &circle_y, &circle_r);
            
            inlier_count = 0;
            circle_score = 0.0f;

            for (int point_index = 0; point_index < s_valid_point_count; point_index++)
            {
                int edge_x = s_edge_x[point_index];
                int edge_y = s_edge_y[point_index];
                float edge_score = s_edge_scores[point_index];

                float delta_x = circle_x - edge_x;
                float delta_y = circle_y - edge_y;

                float delta = sqrt(delta_x * delta_x + delta_y * delta_y);
                float error = abs(circle_r - delta);

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

        if (!circle_valid)
        {
            circle_score = 0;
        }

        //#################################
        // Reduction

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            float other_circle_score = __shfl_down_sync(0xffffffff, circle_score, offset);
            float other_circle_x = __shfl_down_sync(0xffffffff, circle_x, offset);
            float other_circle_y = __shfl_down_sync(0xffffffff, circle_y, offset);
            float other_circle_r = __shfl_down_sync(0xffffffff, circle_r, offset);

            if (other_circle_score > circle_score)
            {
                circle_score = other_circle_score;
                circle_x = other_circle_x;
                circle_y = other_circle_y;
                circle_r = other_circle_r;
            }
        }

        if (lane_index == 0)
        {
            s_score_reduction_buffer[warp_index] = circle_score;
            s_x_reduction_buffer[warp_index] = circle_x;
            s_y_reduction_buffer[warp_index] = circle_y;
            s_r_reduction_buffer[warp_index] = circle_r;
        }

        __syncthreads();

        if (warp_index == 0 && lane_index < warp_count)
        {
            circle_score = s_score_reduction_buffer[warp_index];
            circle_x = s_x_reduction_buffer[warp_index];
            circle_y = s_y_reduction_buffer[warp_index];
            circle_r = s_r_reduction_buffer[warp_index];

            #pragma unroll
            for (int offset = warp_count / 2; offset > 0; offset /= 2)
            {
                float other_circle_score = __shfl_down_sync(0xffffffff, circle_score, offset);
                float other_circle_x = __shfl_down_sync(0xffffffff, circle_x, offset);
                float other_circle_y = __shfl_down_sync(0xffffffff, circle_y, offset);
                float other_circle_r = __shfl_down_sync(0xffffffff, circle_r, offset);

                if (other_circle_score > circle_score)
                {
                    circle_score = other_circle_score;
                    circle_x = other_circle_x;
                    circle_y = other_circle_y;
                    circle_r = other_circle_r;
                }
            }

            if (lane_index == 0)
            {
                g_circle[0] = circle_x;
                g_circle[1] = circle_y;
                g_circle[2] = circle_r;
                g_circle[3] = circle_score;
            }
        }
    }

    // =========================================================================
    // Main function...

    #define ransac_threads RANSAC_ATTEMPTS
    #define ransac_warps (1 + (RANSAC_ATTEMPTS - 1) / 32)

    void fit_circle(const float* points_x, const float* points_y, const float* points_score, const int batch_count, const int point_count, const ConfidenceThresholds confidence_thresholds, const int image_height, const int image_width, float* results)
    {
        dim3 grid(batch_count);
        dim3 block(point_count);
        fit_circle_kernel<ransac_warps><<<grid, block, 3 * point_count * sizeof(int)>>>(points_x, points_y, points_score, results, point_count, confidence_thresholds, image_height, image_width);
    }

}
