#include <cuda_runtime.h>
#include "content_area_inference.cuh"

// TODO, expose params
#define INTENSITY_THRESHOLD 25
#define EDGE_THRESHOLD 20
#define ANGLE_THRESHOLD 30

#define EDGE_CONFIDENCE_THRESHOLD 0.03
#define CONFIDENCE_THRESHOLD 0.06

#define HYBRID_EDGE_CONFIDENCE_THRESHOLD 0.03
#define HYBRID_CONFIDENCE_THRESHOLD 0.06

#define MAX_CENTER_DIST 0.2 // * image width
#define MIN_RADIUS 0.2 // * image width
#define MAX_RADIUS 0.8 // * image width

#define DISCARD_BORDER 3
#define RANSAC_INLIER_THRESHOLD 3
#define RANSAC_ITERATIONS 3
#define MAX_POINT_COUNT 32
#define INVALID_POINT -1
#define DEG2RAD 0.01745329251f
#define RAD2DEG (1.0f / DEG2RAD)

#define WARP_COUNT(x) ((x - 1) >> 5) + 1

// =========================================================================
// General functionality...

__device__ float load_grayscale(const uint8* data, const uint index, const uint x_stride, const uint y_stride)
{
    return 0.2126f * data[index + 0 * x_stride * y_stride] + 0.7152f * data[index + 1 * x_stride * y_stride] + 0.0722f * data[index + 2 * x_stride * y_stride];
}

__device__ float sobel_filter(const float* data, const uint index, const uint x_stride, const uint y_stride, float* x_grad, float* y_grad)
{
    float left  = 0.25f * data[index - x_stride - y_stride] + 0.5f * data[index - x_stride] + 0.25f * data[index - x_stride + y_stride];
    float right = 0.25f * data[index + x_stride - y_stride] + 0.5f * data[index + x_stride] + 0.25f * data[index + x_stride + y_stride];
    *x_grad = right - left;

    float top = 0.25f * data[index - x_stride - y_stride] + 0.5f * data[index - y_stride] + 0.25f * data[index + x_stride - y_stride];
    float bot = 0.25f * data[index - x_stride + y_stride] + 0.5f * data[index + y_stride] + 0.25f * data[index + x_stride + y_stride];
    *y_grad = bot - top;

    return sqrt(*x_grad * *x_grad + *y_grad * *y_grad);
}

__device__ uint fast_rand(uint& seed)
{ 
    seed = 214013 * seed + 2531011; 
    return (seed >> 16) & 0x7FFF; 
}

__device__ void rand_triplet(uint seed, uint seed_stride, uint max, int* triplet) 
{
    triplet[0] = fast_rand(seed) % max;
    do {triplet[1] = fast_rand(seed) % max;} while (triplet[1] == triplet[0]);
    do {triplet[2] = fast_rand(seed) % max;} while (triplet[2] == triplet[0] || triplet[2] == triplet[1]);
}

__device__ bool check_circle(float x, float y, float r, uint image_width, uint image_height)
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

__device__ void fit_circle(int point_count, int* indices, uint* points_x, uint* points_y, float* circle_x, float* circle_y, float* circle_r)
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
__global__ void find_points(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, float* g_edge_scores, const uint image_width, const uint image_height, const uint strip_count)
{
    constexpr uint warp_size = 32;
    constexpr uint thread_count = warp_count * warp_size;
    
    __shared__ float s_image_strip[thread_count * 3];
    __shared__ float s_cross_warp_operation_buffer[warp_count];
    __shared__ float s_cross_warp_operation_buffer_2[warp_count];

    uint warp_index = threadIdx.x >> 5;
    uint lane_index = threadIdx.x & 31;

    bool flip = blockIdx.x == 1;

    // ============================================================
    // Load strip into shared memory...

    int image_x = flip ? image_width - 1 - threadIdx.x : threadIdx.x;

    int strip_index = blockIdx.y;
    int strip_height = 1 + (image_height - 2) / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));
    
    #pragma unroll
    for (int y = 0; y < 3; y++)
    {
        int image_element_index = image_x + (strip_height + (y - 1)) * image_width;
        s_image_strip[threadIdx.x + y * thread_count] = load_grayscale(g_image, image_element_index, image_width, image_height);
    }
    
    __syncthreads();
    
    // ============================================================
    // Calculate largest preceeding intensity...

    float max_preceeding_intensity = s_image_strip[threadIdx.x + thread_count];

    #pragma unroll
    for (int d=1; d < 32; d<<=1) 
    {
        float other_intensity = __shfl_up_sync(0xffffffff, max_preceeding_intensity, d);

        if (lane_index >= d && other_intensity > max_preceeding_intensity) 
        {
            max_preceeding_intensity = other_intensity;
        }
    }

    if (lane_index == warp_size - 1)
    {
        s_cross_warp_operation_buffer[warp_index] = max_preceeding_intensity;
    }
    
    __syncthreads();

    if (warp_index == 0)
    {
        float warp_max = lane_index < warp_count ? s_cross_warp_operation_buffer[lane_index] : 0;

        #pragma unroll
        for (int d=1; d < 32; d<<=1) 
        {
            float other_max = __shfl_up_sync(0xffffffff, warp_max, d);

            if (lane_index >= d && other_max > warp_max) 
            {
                warp_max = other_max;
            }
        }

        if (lane_index < warp_count)
        {
            s_cross_warp_operation_buffer[lane_index] = warp_max;
        }
    }

    __syncthreads();

    if (warp_index > 0)
    {
        float other_intensity = s_cross_warp_operation_buffer[warp_index-1];
        max_preceeding_intensity = other_intensity > max_preceeding_intensity ? other_intensity : max_preceeding_intensity;
    }

    // ============================================================
    // Applying sobel kernel to image patch...

    float x_grad = 0;
    float y_grad = 0;
    float grad = 0;

    if (threadIdx.x > 0 && threadIdx.x < thread_count - 1)
    {
        grad = sobel_filter(s_image_strip, threadIdx.x + thread_count, 1, thread_count, &x_grad, &y_grad);
    }
    
    // ============================================================
    // Calculating angle between gradient vector and center vector...

    float center_dir_x = (image_width / 2.0f) - (float)image_x;
    float center_dir_y = (image_height / 2.0f) - (float)strip_height;
    float center_dir_norm = sqrt(center_dir_x * center_dir_x + center_dir_y * center_dir_y);

    x_grad = flip ? -x_grad : x_grad;

    float dot = grad == 0 ? -1 : (center_dir_x * x_grad + center_dir_y * y_grad) / (center_dir_norm * grad);
    float angle = RAD2DEG * acos(dot);

    // ============================================================
    // Final scoring...

    float edge_score = tanh(grad / EDGE_THRESHOLD);
    float angle_score = 1.0f - tanh(angle / ANGLE_THRESHOLD);
    float intensity_score = 1.0f - tanh(max_preceeding_intensity / INTENSITY_THRESHOLD);

    float point_score = edge_score * angle_score * intensity_score;
    
    // ============================================================
    // Reduction to find the best edge...

    
    bool is_valid = threadIdx.x > DISCARD_BORDER && point_score >= EDGE_CONFIDENCE_THRESHOLD;

    int best_edge_x = is_valid ? image_x : INVALID_POINT;
    float best_edge_score = is_valid ? point_score : 0.0f;
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        int other_edge_x = __shfl_down_sync(0xffffffff, best_edge_x, offset);
        float other_edge_score = __shfl_down_sync(0xffffffff, best_edge_score, offset);

        if (other_edge_score > best_edge_score)
        {
            best_edge_x = other_edge_x;
            best_edge_score = other_edge_score;
        }
    }

    if (lane_index == 0)
    {
        s_cross_warp_operation_buffer[warp_index] = best_edge_x;
        s_cross_warp_operation_buffer_2[warp_index] = best_edge_score;
    }

    __syncthreads();

    // block reduction....
    if (warp_index == 0 && lane_index < warp_count)
    {
        best_edge_x = s_cross_warp_operation_buffer[lane_index];
        best_edge_score = s_cross_warp_operation_buffer_2[lane_index];

        #pragma unroll
        for (int offset = warp_count >> 1 ; offset > 0; offset >>= 1)
        {
            int other_edge_x = __shfl_down_sync(0xffffffff, best_edge_x, offset);
            float other_edge_score = __shfl_down_sync(0xffffffff, best_edge_score, offset);

            if (other_edge_score > best_edge_score)
            {
                best_edge_x = other_edge_x;
                best_edge_score = other_edge_score;
            }
        }

        if (lane_index == 0)
        {
            uint point_index = flip ? strip_index : strip_index + strip_count;
            g_edge_x[point_index] = best_edge_x;
            g_edge_y[point_index] = strip_height;
            g_edge_scores[point_index] = best_edge_score;
        }
    }
}


template<int warp_count>
__global__ void rand_ransac(const uint* g_edge_x, const uint* g_edge_y, const float* g_edge_scores, float* g_circle, const uint point_count, const uint image_height, const uint image_width)
{
    __shared__ uint s_edge_x[MAX_POINT_COUNT];
    __shared__ uint s_edge_y[MAX_POINT_COUNT];
    __shared__ float s_edge_scores[MAX_POINT_COUNT];
    __shared__ int s_valid_point_count;

    __shared__ float s_score_reduction_buffer[warp_count];
    __shared__ float s_x_reduction_buffer[warp_count];
    __shared__ float s_y_reduction_buffer[warp_count];
    __shared__ float s_r_reduction_buffer[warp_count];

    const uint warp_index = threadIdx.x >> 5;
    const uint lane_index = threadIdx.x & 31;

    // Loading points to shared memory...
    if (threadIdx.x < point_count)
    {
        s_edge_x[threadIdx.x] = g_edge_x[threadIdx.x];
        s_edge_y[threadIdx.x] = g_edge_y[threadIdx.x];
        s_edge_scores[threadIdx.x] = g_edge_scores[threadIdx.x];
    }

    // Point compaction...
    bool has_point = threadIdx.x < point_count && s_edge_x[threadIdx.x] != INVALID_POINT;
    int preceeding_count = has_point;

    if (warp_index == 0)
    {
        #pragma unroll
        for (int d=1; d < 32; d<<=1) 
        {
            float other_count = __shfl_up_sync(0xffffffff, preceeding_count, d);

            if (lane_index >= d) 
            {
                preceeding_count += other_count;
            }
        }

        if (has_point)
        {
            s_edge_x[preceeding_count - 1] = s_edge_x[threadIdx.x];
            s_edge_y[preceeding_count - 1] = s_edge_y[threadIdx.x];
            s_edge_scores[preceeding_count - 1] = s_edge_scores[threadIdx.x];
        }

        if (lane_index == 31)
        {
            s_valid_point_count = preceeding_count;
        }
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
    rand_triplet(threadIdx.x * 42342, blockDim.x, s_valid_point_count, inliers);

    float circle_x, circle_y, circle_r;
    float circle_score = 0.0f;

    for (int i = 0; i < RANSAC_ITERATIONS; i++)
    {
        fit_circle(inlier_count, inliers, s_edge_x, s_edge_y, &circle_x, &circle_y, &circle_r);
        
        inlier_count = 0;
        circle_score = 0.0f;

        for (int point_index = 0; point_index < s_valid_point_count; point_index++)
        {
            int edge_x = s_edge_x[point_index];
            int edge_y = s_edge_y[point_index];
            float edge_score = s_edge_scores[point_index];

            if (edge_x != INVALID_POINT)
            {  
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

#define warp_size 32
#define find_points_warp_count 8

#define corner_triangle_size 7
#define corner_triangle_threads corner_triangle_size * (corner_triangle_size + 1) / 2

#define ransac_threads 32
#define ransac_warps WARP_COUNT(ransac_threads)

ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    // #########################################################
    // Finding candididate points...

    dim3 find_points_grid(2, m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, m_dev_edge_scores, image_width, image_height, m_height_samples);

    // #########################################################
    // Fitting circle to candididate points...

    dim3 rand_ransac_grid(1);
    dim3 rand_ransac_block(ransac_threads);
    rand_ransac<ransac_warps><<<rand_ransac_grid, rand_ransac_block>>>(m_dev_edge_x, m_dev_edge_y, m_dev_edge_scores, m_dev_circle, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results...

    cudaMemcpy(m_hst_circle, m_dev_circle, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float circle_x = m_hst_circle[0];
    float circle_y = m_hst_circle[1];
    float circle_r = m_hst_circle[2];
    float confidence_score = m_hst_circle[3];

    // #########################################################
    // Returning...

    if (confidence_score >= CONFIDENCE_THRESHOLD)
    {
        return ContentArea(circle_x, circle_y, circle_r);
    }
    else
    {
        return ContentArea();
    }
}

template<int warp_count>
__global__ void find_best_edge(const float* g_score_strips, uint* g_edge_x, uint* g_edge_y, float* g_edge_scores, const uint image_width, const uint image_height, const uint strip_count)
{
    __shared__ float s_cross_warp_operation_buffer[warp_count];
    __shared__ float s_cross_warp_operation_buffer_2[warp_count];

    uint warp_index = threadIdx.x >> 5;
    uint lane_index = threadIdx.x & 31;

    bool flip = blockIdx.x == 1;

    // ============================================================
    // Load strip into shared memory...

    int image_x = flip ? image_width - 1 - threadIdx.x : threadIdx.x;

    int strip_index = blockIdx.y;
    int strip_height = 1 + (image_height - 2) / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));
    
    float point_score = g_score_strips[image_x + strip_index * image_width];
   
    bool is_valid = threadIdx.x > DISCARD_BORDER && point_score >= HYBRID_EDGE_CONFIDENCE_THRESHOLD;

    int best_edge_x = is_valid ? image_x : INVALID_POINT;
    float best_edge_score = is_valid ? point_score : 0.0f;
    
    // warp reduction....
    #pragma unroll
    for (int offset = 32 >> 1; offset > 0; offset >>= 1)
    {
        int other_edge_x = __shfl_down_sync(0xffffffff, best_edge_x, offset);
        float other_edge_score = __shfl_down_sync(0xffffffff, best_edge_score, offset);

        if (other_edge_score > best_edge_score)
        {
            best_edge_x = other_edge_x;
            best_edge_score = other_edge_score;
        }
    }

    if (lane_index == 0)
    {
        s_cross_warp_operation_buffer[warp_index] = best_edge_x;
        s_cross_warp_operation_buffer_2[warp_index] = best_edge_score;
    }

    __syncthreads();

    // block reduction....
    if (warp_index == 0 && lane_index < warp_count)
    {
        best_edge_x = s_cross_warp_operation_buffer[lane_index];
        best_edge_score = s_cross_warp_operation_buffer_2[lane_index];

        #pragma unroll
        for (int offset = warp_count >> 1 ; offset > 0; offset >>= 1)
        {
            int other_edge_x = __shfl_down_sync(0xffffffff, best_edge_x, offset);
            float other_edge_score = __shfl_down_sync(0xffffffff, best_edge_score, offset);

            if (other_edge_score > best_edge_score)
            {
                best_edge_x = other_edge_x;
                best_edge_score = other_edge_score;
            }
        }

        if (lane_index == 0)
        {
            uint point_index = flip ? strip_index : strip_index + strip_count;
            g_edge_x[point_index] = best_edge_x;
            g_edge_y[point_index] = strip_height;
            g_edge_scores[point_index] = best_edge_score;
        }
    }
}


ContentArea ContentAreaInference::infer_area_hybrid(float* strips, const uint image_height, const uint image_width, const uint strip_count)
{
    dim3 find_points_grid(2, strip_count);
    dim3 find_points_block(32 * find_points_warp_count);
    find_best_edge<find_points_warp_count><<<find_points_grid, find_points_block>>>(strips, m_dev_edge_x, m_dev_edge_y, m_dev_edge_scores, image_width, image_height, strip_count);
    
    dim3 rand_ransac_grid(1);
    dim3 rand_ransac_block(ransac_threads);
    rand_ransac<ransac_warps><<<rand_ransac_grid, rand_ransac_block>>>(m_dev_edge_x, m_dev_edge_y, m_dev_edge_scores, m_dev_circle, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results...

    cudaMemcpy(m_hst_circle, m_dev_circle, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float circle_x = m_hst_circle[0];
    float circle_y = m_hst_circle[1];
    float circle_r = m_hst_circle[2];
    float confidence_score = m_hst_circle[3];

    // #########################################################
    // Returning...

    if (confidence_score >= HYBRID_CONFIDENCE_THRESHOLD)
    {
        return ContentArea(circle_x, circle_y, circle_r);
    }
    else
    {
        return ContentArea();
    }
}

std::vector<std::vector<float>> ContentAreaInference::get_debug()
{
    cudaMemcpy(m_hst_buffer, m_dev_buffer, m_buffer_size, cudaMemcpyDeviceToHost);

    float circle_x = m_hst_circle[0];
    float circle_y = m_hst_circle[1];
    float circle_r = m_hst_circle[2];
    float confidence_score = m_hst_circle[3];

    std::vector<float> points_x, points_y, points_scores, circle;

    for (int i = 0; i < m_point_count; i++)
    {
        points_x.push_back(m_hst_edge_x[i]);
        points_y.push_back(m_hst_edge_y[i]);
        points_scores.push_back(m_hst_edge_scores[i]);
    }

    circle.push_back(circle_x);
    circle.push_back(circle_y);
    circle.push_back(circle_r);
    circle.push_back(confidence_score);

    std::vector<std::vector<float>> result = std::vector<std::vector<float>>();
    result.push_back(points_x);
    result.push_back(points_y);
    result.push_back(points_scores);
    result.push_back(circle);

    return result;
}
