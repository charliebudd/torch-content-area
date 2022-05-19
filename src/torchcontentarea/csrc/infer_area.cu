#include <cuda_runtime.h>
#include "content_area_inference.cuh"

// TODO, expose params
#define MAX_CENTER_DIST 0.15 // * image width
#define MIN_RADIUS 0.2 // * image width
#define MAX_RADIUS 0.6 // * image width
#define EDGE_THRESHOLD 4
#define STRICT_EDGE_THRESHOLD 10.5
#define ANGLE_THRESHOLD 26
#define INLIER_THRESHOLD 3.0f
#define MAX_RANSAC_ITERATIONS 10
#define DISCARD_BORDER 4

#define MAX_POINT_COUNT 32
#define INVALID_POINT -1
#define DEG2RAD 0.01745329251f

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

__device__ int fast_rand(int seed) 
{ 
    seed = 214013 * seed + 2531011; 
    return (seed >> 16) & 0x7FFF; 
}

__device__ void rand_triplet(int seed, int seed_stride, int max, int* triplet) 
{ 
    triplet[0] = fast_rand(0.3 * seed) % max; seed += 3 * seed_stride;
    do {triplet[1] = fast_rand(0.3 * seed) % max; seed += 3 * seed_stride;} while (triplet[1] == triplet[0]);
    do {triplet[2] = fast_rand(0.3 * seed) % max; seed += 3 * seed_stride;} while (triplet[2] == triplet[0] || triplet[2] == triplet[1]);
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


// =========================================================================
// Kernels...

template<int warp_count>
__global__ void border_guess(const uint8* g_image, const uint image_width, const uint image_height, float* g_x, float* g_xx)
{
    constexpr uint warp_size = 32;
    constexpr uint thread_count = warp_count * warp_size;

    __shared__ float x_reduction[warp_count];
    __shared__ float xx_reduction[warp_count];
    
    uint thread_index = threadIdx.x + threadIdx.y * blockDim.x;
    uint warp_index = thread_index >> 5;
    uint lane_index = thread_index & 31;

    int image_x, image_y;

    switch (blockIdx.x)
    {
        case (0):
        {
            image_x = threadIdx.x;
            image_y = threadIdx.y;
        }
        break;
        
        case (1):
        {
            image_x = (image_width - 1) - threadIdx.x;
            image_y = threadIdx.y;
        }
        break;
        
        case (2):
        {
            image_x = threadIdx.x;
            image_y = (image_height - 1) - threadIdx.y;
        }
        break;
        
        case (3):
        {
            image_x = (image_width - 1) - threadIdx.x;
            image_y = (image_height - 1) - threadIdx.y;
        }
        break;
        
        case (4):
        {  
            float stride_x = (image_width - 2 * blockDim.x) / blockDim.x;
            float stride_y = (image_height - 2 * blockDim.y) / blockDim.y;
    
            image_x = (image_width / 2 - stride_x * 0.5 * blockDim.x) + stride_x * (threadIdx.x + 0.5);
            image_y = (image_height / 2 - stride_y * 0.5 * blockDim.y) + stride_y * (threadIdx.y + 0.5);
        }
        break;
    }

    
    float x = 0;
    float xx = 0;

    for (int c = 0; c < 3; c++)
    {
        float value = g_image[image_x + image_y * image_width + c * image_width * image_height];

        x += value;
        xx += value * value;
    }
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        x += __shfl_down_sync(0xffffffff, x, offset);
        xx += __shfl_down_sync(0xffffffff, xx, offset);
    }

    if (lane_index == 0)
    {
        x_reduction[warp_index] = x;
        xx_reduction[warp_index] = xx;
    }

    __syncthreads();

    // block reduction....
    if (warp_index == 0 && lane_index < warp_count)
    {
        x = x_reduction[lane_index];
        xx = xx_reduction[lane_index];

        #pragma unroll
        for (int offset = warp_count >> 1 ; offset > 0; offset >>= 1)
        {
            x += __shfl_down_sync(0xffffffff, x, offset);
            xx += __shfl_down_sync(0xffffffff, xx, offset);
        }

        if (lane_index == 0)
        {
            int count = 3 * blockDim.x * blockDim.y;
            g_x[blockIdx.x] = x / count;
            g_xx[blockIdx.x] = xx / count;
        }
    }
}

template<int warp_count>
__global__ void find_points(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, const uint image_width, const uint image_height, const uint strip_count)
{
    constexpr uint warp_size = 32;
    constexpr uint thread_count = warp_count * warp_size;
    
    uint warp_index = threadIdx.x >> 5;
    uint lane_index = threadIdx.x & 31;

    __shared__ float s_image_strip[thread_count * 3];
    __shared__ uint s_first_edge_reduction_buffer[warp_count];
    __shared__ uint s_last_edge_reduction_buffer[warp_count];

    // ============================================================
    // Load strip into shared memory with linear interpolation...

    float remainder = threadIdx.x * image_width / (float)thread_count;
    int image_x = remainder;
    remainder -= image_x;

    int strip_index = blockIdx.x;
    int strip_height = image_height / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f)));
    
    #pragma unroll
    for (int y = 0; y < 3; y++)
    {
        int image_element_index = image_x + (strip_height + (y - 1)) * image_width;

        float a = load_grayscale(g_image, image_element_index, image_width, image_height);
        float b = load_grayscale(g_image, image_element_index + 1, image_width, image_height);

        s_image_strip[threadIdx.x + y * thread_count] = (a * (1 - remainder) + b * remainder) / 2;
    }
    
    __syncthreads();

    // ============================================================
    // Applying sobel kernel to image patch...

    float x_grad = 0;
    float y_grad = 0;
    float grad = 0;

    if (threadIdx.x > 0 && threadIdx.x < thread_count - 1)
    {
        grad = sobel_filter(s_image_strip, threadIdx.x + thread_count, 1, thread_count, &x_grad, &y_grad);
    }

    bool is_edge = grad > EDGE_THRESHOLD;

    // ============================================================
    // Reduction to find the first and last edge in strip...

    int first_edge = is_edge ? image_x : INVALID_POINT;
    int last_edge = is_edge ? image_x : INVALID_POINT;
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        int other_first_edge = __shfl_down_sync(0xffffffff, first_edge, offset);
        int other_last_edge = __shfl_down_sync(0xffffffff, last_edge, offset);

        if (first_edge == INVALID_POINT || (other_first_edge != INVALID_POINT && other_first_edge < first_edge))
        {
            first_edge = other_first_edge;
        }
        
        if (last_edge == INVALID_POINT || (other_last_edge != INVALID_POINT && other_last_edge > last_edge))
        {
            last_edge = other_last_edge;
        }
    }

    if (lane_index == 0)
    {
        s_first_edge_reduction_buffer[warp_index] = first_edge;
        s_last_edge_reduction_buffer[warp_index] = last_edge;
    }

    __syncthreads();

    // block reduction....
    if (warp_index == 0 && lane_index < warp_count)
    {
        first_edge = s_first_edge_reduction_buffer[lane_index];
        last_edge = s_last_edge_reduction_buffer[lane_index];

        #pragma unroll
        for (int offset = warp_count >> 1 ; offset > 0; offset >>= 1)
        {
            int other_first_edge = __shfl_down_sync(0xffffffff, first_edge, offset);
            int other_last_edge = __shfl_down_sync(0xffffffff, last_edge, offset);

            if (first_edge == INVALID_POINT || (other_first_edge != INVALID_POINT && other_first_edge < first_edge))
            {
                first_edge = other_first_edge;
            }
            
            if (last_edge == INVALID_POINT || (other_last_edge != INVALID_POINT && other_last_edge > last_edge))
            {
                last_edge = other_last_edge;
            }
        }

        if (lane_index == 0)
        {
            g_edge_x[strip_index] = first_edge;
            g_edge_y[strip_index] = strip_height;

            g_edge_x[strip_index + strip_count] = last_edge;
            g_edge_y[strip_index + strip_count] = strip_height;
        }
    }
}

__global__ void refine_points(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, float* g_norm_x, float* g_norm_y, const uint image_width, const uint image_height, const uint strip_count)
{
    constexpr uint warp_size = 32;

    __shared__ float s_image_strip[warp_size * 3];
    
    // ============================================================
    // Get approximate point position...

    int point_index = blockIdx.x;

    uint point_x = g_edge_x[point_index];
    uint point_y = g_edge_y[point_index];

    if (point_x == INVALID_POINT || point_x < DISCARD_BORDER || point_x > image_width - (DISCARD_BORDER+1))
        return;

    point_x = max(warp_size / 2, min(point_x, image_width - 1 - warp_size / 2));

    // ============================================================
    // Load patch centered around point position into shared memory...

    int image_x = point_x + (threadIdx.x - warp_size / 2);
    
    for (int y = 0; y < 3; y++)
    {
        int image_index = image_x + (point_y + (y - 1)) * image_width;
        s_image_strip[threadIdx.x + y * warp_size] = load_grayscale(g_image, image_index, image_width, image_height);
    }
    
    __syncthreads();
    
    // ============================================================
    // Apply sobel filter...

    float x_grad = 0;
    float y_grad = 0;
    float grad = 0;

    if (threadIdx.x > 0 && threadIdx.x < warp_size - 1)
    {
        grad = sobel_filter(s_image_strip, threadIdx.x + warp_size, 1, warp_size, &x_grad, &y_grad);
    }

    float center_dir_x = (image_width / 2.0f) - (float)image_x;
    float center_dir_y = (image_height / 2.0f) - (float)point_y;

    float center_dir_norm = sqrt(center_dir_x * center_dir_x + center_dir_y * center_dir_y);

    float dot = (center_dir_x * x_grad + center_dir_y * y_grad) / (center_dir_norm * grad);

    bool is_edge = grad > STRICT_EDGE_THRESHOLD &&  dot > cos(ANGLE_THRESHOLD * DEG2RAD);

    x_grad /= grad;
    y_grad /= grad;

    // ============================================================
    // Reduction to find the first and last edge in strip...

    bool flip = point_index >= strip_count;
    int edge = is_edge ? image_x : INVALID_POINT;
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        int other_edge = __shfl_down_sync(0xffffffff, edge, offset);
        float other_x_grad = __shfl_down_sync(0xffffffff, x_grad, offset);
        float other_y_grad = __shfl_down_sync(0xffffffff, y_grad, offset);

        if (edge == INVALID_POINT || (other_edge != INVALID_POINT && flip == other_edge > edge))
        {
            edge = other_edge;
            x_grad = other_x_grad;
            y_grad = other_y_grad;
        }
    }

    if (threadIdx.x == 0)
    {
        g_edge_x[point_index] = edge;
        g_norm_x[point_index] = x_grad;
        g_norm_y[point_index] = y_grad;
    }
}

template<int warp_count>
__global__ void rand_ransac(const uint* g_edge_x, const uint* g_edge_y, const float* g_norm_x, const float* g_norm_y, float* g_circle, const uint point_count, const uint image_height, const uint image_width)
{
    __shared__ uint s_edge_x[MAX_POINT_COUNT];
    __shared__ uint s_edge_y[MAX_POINT_COUNT];
    __shared__ float s_norm_x[MAX_POINT_COUNT];
    __shared__ float s_norm_y[MAX_POINT_COUNT];

    __shared__ float s_score_reduction_buffer[warp_count];
    __shared__ float s_x_reduction_buffer[warp_count];
    __shared__ float s_y_reduction_buffer[warp_count];
    __shared__ float s_r_reduction_buffer[warp_count];

    if (threadIdx.x < point_count)
    {
        s_edge_x[threadIdx.x] = g_edge_x[threadIdx.x];
        s_edge_y[threadIdx.x] = g_edge_y[threadIdx.x];
        s_norm_x[threadIdx.x] = g_norm_x[threadIdx.x];
        s_norm_y[threadIdx.x] = g_norm_y[threadIdx.x];
    }
    
    __syncthreads();

    const uint warp_index = threadIdx.x >> 5;
    const uint lane_index = threadIdx.x & 31;

    int inlier_count = 3;
    int inliers[MAX_POINT_COUNT];
    rand_triplet(threadIdx.x, blockDim.x, point_count, inliers);

    float circle_x, circle_y, circle_r;
    float circle_score = 0.0f;

    for (int i = 0; i < MAX_RANSAC_ITERATIONS; i++)
    {
        fit_circle(inlier_count, inliers, s_edge_x, s_edge_y, &circle_x, &circle_y, &circle_r);
        
        inlier_count = 0;
        circle_score = 0.0f;

        for (int point_index = 0; point_index < point_count; point_index++)
        {
            int edge_x = s_edge_x[point_index];
            int edge_y = s_edge_y[point_index];

            if (edge_x != INVALID_POINT && edge_x > DISCARD_BORDER && edge_x < image_width - (DISCARD_BORDER+1))
            {  
                float delta_x = circle_x - edge_x;
                float delta_y = circle_y - edge_y;

                float delta = sqrt(delta_x * delta_x + delta_y * delta_y);
                float error = abs(circle_r - delta);

                if (error < INLIER_THRESHOLD)
                {
                    float norm_x = s_norm_x[point_index];
                    float norm_y = s_norm_y[point_index];

                    float dot = (delta_x * norm_x + delta_y * norm_y) / delta;

                    circle_score += dot;

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
#define find_points_warp_count 16

#define border_guess_sample_size 8
#define border_guess_warp_count WARP_COUNT(border_guess_sample_size * border_guess_sample_size)

#define ransac_threads 128
#define ransac_warps WARP_COUNT(ransac_threads)

ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    // #########################################################
    // Guess if border...
    
    dim3 border_guess_grid(5);
    dim3 border_guess_block(border_guess_sample_size, border_guess_sample_size);
    border_guess<border_guess_warp_count><<<border_guess_grid, border_guess_block>>>(image, image_width, image_height, m_dev_x_sums, m_dev_xx_sums);

    // #########################################################
    // Finding candididate points...

    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(warp_size);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, m_dev_norm_x, m_dev_norm_y, image_width, image_height, m_height_samples);

    // #########################################################
    // Fitting circle to candididate points...

    dim3 rand_ransac_grid(1);
    dim3 rand_ransac_block(ransac_threads);
    rand_ransac<ransac_warps><<<rand_ransac_grid, rand_ransac_block>>>(m_dev_edge_x, m_dev_edge_y, m_dev_norm_x, m_dev_norm_y, m_dev_circle, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results...

    cudaMemcpy(m_hst_buffer, m_dev_buffer, m_buffer_size, cudaMemcpyDeviceToHost);

    // #########################################################
    // Checking border...

    float corner_x_mean = (m_hst_x_sums[0] + m_hst_x_sums[1] + m_hst_x_sums[2] + m_hst_x_sums[3]) / 4;
    float corner_xx_mean = (m_hst_xx_sums[0] + m_hst_xx_sums[1] + m_hst_xx_sums[2] + m_hst_xx_sums[3]) / 4;

    float middle_x_mean = m_hst_x_sums[4];
    float middle_xx_mean = m_hst_xx_sums[4];

    float corner_std = sqrt(corner_xx_mean - corner_x_mean * corner_x_mean);
    float middle_std = sqrt(middle_xx_mean - middle_x_mean * middle_x_mean);

    bool border = abs(corner_x_mean - middle_x_mean) > (corner_std + middle_std) && corner_x_mean < middle_x_mean;
    float border_score = tanh(abs(corner_x_mean - middle_x_mean) / (corner_std + middle_std));
    
    // #########################################################
    // Checking circle...

    float circle_x = m_hst_circle[0];
    float circle_y = m_hst_circle[1];
    float circle_r = m_hst_circle[2];
    float circle_score = m_hst_circle[3];

    // #########################################################
    // Returning...

    float confidence_score = circle_score * border_score;

    if (confidence_score >= 0.15)
    {
        return ContentArea(circle_x, circle_y, circle_r);
    }
    else
    {
        return ContentArea();
    }
}

std::vector<std::vector<float>> ContentAreaInference::get_debug(uint8* image, const uint image_height, const uint image_width)
{
    infer_area(image, image_height, image_width);
   
    // #########################################################
    // Checking border...

    float corner_x_mean = (m_hst_x_sums[0] + m_hst_x_sums[1] + m_hst_x_sums[2] + m_hst_x_sums[3]) / 4;
    float corner_xx_mean = (m_hst_xx_sums[0] + m_hst_xx_sums[1] + m_hst_xx_sums[2] + m_hst_xx_sums[3]) / 4;

    float middle_x_mean = m_hst_x_sums[4];
    float middle_xx_mean = m_hst_xx_sums[4];

    float corner_std = sqrt(corner_xx_mean - corner_x_mean * corner_x_mean);
    float middle_std = sqrt(middle_xx_mean - middle_x_mean * middle_x_mean);

    bool border = abs(corner_x_mean - middle_x_mean) > (corner_std + middle_std) && corner_x_mean < middle_x_mean;
    float border_score = tanh(abs(corner_x_mean - middle_x_mean) / (corner_std + middle_std));
    
    // #########################################################
    // Checking circle...

    float circle_x = m_hst_circle[0];
    float circle_y = m_hst_circle[1];
    float circle_r = m_hst_circle[2];
    float circle_score = m_hst_circle[3];

    // #########################################################
    // Returning...

    float confidence_score = circle_score * border_score;

    // #########################################################
    // Checking border...

    std::vector<float> points_x, points_y, norm_x, norm_y, circle, scores;

    for (int i = 0; i < m_point_count; i++)
    {
        points_x.push_back(m_hst_edge_x[i]);
        points_y.push_back(m_hst_edge_y[i]);
        norm_x.push_back(m_hst_norm_x[i]);
        norm_y.push_back(m_hst_norm_y[i]);
    }

    circle.push_back(circle_x);
    circle.push_back(circle_y);
    circle.push_back(circle_r);

    scores.push_back(border_score);
    scores.push_back(circle_score);
    scores.push_back(confidence_score);

    std::vector<std::vector<float>> result = std::vector<std::vector<float>>();
    result.push_back(points_x);
    result.push_back(points_y);
    result.push_back(norm_x);
    result.push_back(norm_y);
    result.push_back(circle);
    result.push_back(scores);

    return result;
}
