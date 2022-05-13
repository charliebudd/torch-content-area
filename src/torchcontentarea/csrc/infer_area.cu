#include <cuda_runtime.h>
#include "content_area_inference.cuh"

// TODO, expose params
#define MAX_CENTER_DIST 0.15 // * image width
#define MIN_RADIUS 0.2 // * image width
#define MAX_RADIUS 0.6 // * image width
#define EDGE_THRESHOLD 4
#define STRICT_EDGE_THRESHOLD 10.5
#define ANGLE_THRESHOLD 26
#define INLIER_THRESHOLD 4.0f
#define MAX_RANSAC_ITERATIONS 10
#define VALID_POINT_THRESHOLD 0.03

#define MAX_POINT_COUNT 32
#define DEG2RAD 0.01745329251f

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
    int strip_height = (strip_index + 0.5) * image_height / strip_count;
    // More points at the extremes...
    // strip_height = image_height / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f)));

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

    int first_edge = is_edge ? image_x : image_width;
    int last_edge = is_edge ? image_x : -1;
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        int other_first_edge = __shfl_down_sync(0xffffffff, first_edge, offset);
        int other_last_edge = __shfl_down_sync(0xffffffff, last_edge, offset);

        if (other_first_edge < first_edge)
        {
            first_edge = other_first_edge;
        }
        
        if (other_last_edge > last_edge)
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

            if (other_first_edge < first_edge)
            {
                first_edge = other_first_edge;
            }
            
            if (other_last_edge > last_edge)
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

    if (point_x < 1 || point_x > image_width - 2)
        return;

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
    int edge = is_edge ? image_x : (flip ? 0 : image_width);
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        int other_edge = __shfl_down_sync(0xffffffff, edge, offset);
        float other_x_grad = __shfl_down_sync(0xffffffff, x_grad, offset);
        float other_y_grad = __shfl_down_sync(0xffffffff, y_grad, offset);

        if ((flip && other_edge > edge) || (!flip && other_edge < edge))
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

__host__ __device__ bool calculate_circle(float ax, float ay, float bx, float by, float cx, float cy, float* x, float* y, float* r)
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

__host__ __device__ bool check_circle(float x, float y, float r, uint image_width, uint image_height)
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

__host__ __device__ uint triangle_size(const uint n)
{
    return n * (n - 1) / 2;
}

// Calculates the ij indices for element k of a n x n square 
__device__ void square_indices(const int k, const int n, uint* i, uint* j)
{
    *i = n - 2 - int(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    *j = k + *i + 1 -  n * (n - 1) / 2 + (n - *i) * ((n - *i) - 1) / 2;
}

__global__ void check_triples(const uint* g_edge_x, const uint* g_edge_y, float* g_edge_scores, const uint point_count, const uint image_height, const uint image_width)
{
    __shared__ uint s_edge_x[MAX_POINT_COUNT];
    __shared__ uint s_edge_y[MAX_POINT_COUNT];
    __shared__ float s_score_reduction_buffer[MAX_POINT_COUNT];

    if (threadIdx.x < point_count)
    {
        s_edge_x[threadIdx.x] = g_edge_x[threadIdx.x];
        s_edge_y[threadIdx.x] = g_edge_y[threadIdx.x];
    }
    
    __syncthreads();

    const uint warp_count = ((blockDim.x - 1) >> 5) + 1;
    const uint warp_index = threadIdx.x >> 5;
    const uint lane_index = threadIdx.x & 31;

    uint a_index, b_index, c_index;
    a_index = blockIdx.x;
    square_indices(threadIdx.x, point_count, &b_index, &c_index);

    float ax = s_edge_x[a_index];
    float ay = s_edge_y[a_index];
    
    float bx = s_edge_x[b_index];
    float by = s_edge_y[b_index];
    
    float cx = s_edge_x[c_index];
    float cy = s_edge_y[c_index];

    bool valid = ax > 1 && ax < image_width - 2 && bx > 1 && bx < image_width - 2 && cx > 1 && cx < image_width - 2;

    float x, y, r;
    valid &= calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);
    valid &= check_circle(x, y, r, image_width, image_height);

    float score = 0.0f;

    if (valid)
    {
        score = 0;

        for (int i=0; i < point_count; i++)
        {
            float diff_x = (s_edge_x[i] - x);
            float diff_y = (s_edge_y[i] - y);

            float diff = sqrt(diff_x * diff_x + diff_y * diff_y);
            diff = abs(diff - r);

            if (diff < INLIER_THRESHOLD)
            {
                score += 1 - diff / INLIER_THRESHOLD;
            }
        }
    }

    //#################################
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        score += __shfl_down_sync(0xffffffff, score, offset);
    }

    if (lane_index == 0)
    { 
        s_score_reduction_buffer[warp_index] = score;
    }

    // Syncing between warps
    __syncthreads();

    // Block reduction
    if (warp_index == 0 && lane_index < warp_count)
    {
        score = s_score_reduction_buffer[lane_index];
        
        #pragma unroll
        for (int offset = warp_count / 2; offset > 0; offset /= 2)
        {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }

        // Outputting result
        if (lane_index == 0)
        {
            g_edge_scores[a_index] = score / ((blockDim.x - (point_count - 1)) * point_count);
        }
    }
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

void fit_circle(int point_count, int* indices, uint* points_x, uint* points_y, float* circle_x, float* circle_y, float* circle_r)
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

void random_triplet(int max, int& a, int& b, int& c)
{
    a = rand() % max;
    do b = rand() % max; while (b == a);
    do c = rand() % max; while (c == a || c == b);
}

void circle_selection(const uint point_count, uint* points_x, uint* points_y, float* circle_x, float* circle_y, float* circle_r)
{
    int best_inlier_count = 0;

    int* new_inliers = new int[point_count];
    int* old_inliers = new int[point_count];

    int new_inlier_count = 0;
    int old_inlier_count = 0;

    for (int i = 0; i < MAX_RANSAC_ITERATIONS; i++)
    {
        new_inlier_count = 3;
        random_triplet(point_count, new_inliers[0], new_inliers[1], new_inliers[2]);

        float new_circle_x, new_circle_y, new_circle_r;

        for (int j = 0; j < MAX_RANSAC_ITERATIONS; j++)
        {
            fit_circle(new_inlier_count, new_inliers, points_x, points_y, &new_circle_x, &new_circle_y, &new_circle_r);
            
            if (new_inlier_count == point_count)
            {
                break;
            }

            std::swap(new_inliers, old_inliers);

            old_inlier_count = new_inlier_count;
            new_inlier_count = 0;

            bool no_change = true;

            for (int point_index = 0; point_index < point_count; point_index++)
            {
                float delta_x = points_x[point_index] - new_circle_x;
                float delta_y = points_y[point_index] - new_circle_y;

                float delta = sqrt(delta_x * delta_x + delta_y * delta_y);
                float error = abs(new_circle_r - delta);

                if (error < INLIER_THRESHOLD)
                {
                    no_change &= old_inliers[new_inlier_count] == point_index;
                    new_inliers[new_inlier_count] = point_index;
                    new_inlier_count++;
                }
            }

            no_change &= new_inlier_count == old_inlier_count;

            if (new_inlier_count < 3 || no_change)
            {
                break;
            }
        }

        if (new_inlier_count > best_inlier_count)
        {
            best_inlier_count = new_inlier_count;

            *circle_x = new_circle_x;
            *circle_y = new_circle_y;
            *circle_r = new_circle_r;
        }

        if (new_inlier_count == point_count)
        {
            break;
        }
    }

    delete new_inliers;
    delete old_inliers;
}

#define warp_size 32
#define find_points_warp_count 16

ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    // #########################################################
    // Finding candididate points...

    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(warp_size);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, m_dev_norm_x, m_dev_norm_y, image_width, image_height, m_height_samples);

    // #########################################################
    // Evaluating candidate points...

    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, m_dev_scores, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results...

    cudaMemcpy(m_hst_buffer, m_dev_buffer, m_buffer_size, cudaMemcpyDeviceToHost);

    // #########################################################
    // Removing invalid points...

    int valid_point_count = 0;

    for (int i = 0; i < m_point_count; i++)
    {
        if (m_hst_edge_x[i] != -1 && m_hst_scores[i] > VALID_POINT_THRESHOLD)
        {
            m_hst_edge_x[valid_point_count] = m_hst_edge_x[i];
            m_hst_edge_y[valid_point_count] = m_hst_edge_y[i];
            m_hst_scores[valid_point_count] = m_hst_scores[i];
            valid_point_count += 1;
        }
    }
    
    // #########################################################
    // Fitting final circle...

    if (valid_point_count < 3)
    {
        return ContentArea();
    }
    else if (valid_point_count == 3)
    {    
        int ax = m_hst_edge_x[0];
        int ay = m_hst_edge_y[0];
        
        int by = m_hst_edge_y[1];
        int bx = m_hst_edge_x[1];
        
        int cx = m_hst_edge_x[2];
        int cy = m_hst_edge_y[2];

        float x, y, r;
        calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

        return ContentArea(x, y, r);
    }
    else
    {
        float x, y, r;
        circle_selection(valid_point_count, m_hst_edge_x, m_hst_edge_y, &x, &y, &r);

        return ContentArea(x, y, r);
    }
}

std::vector<std::vector<int>> ContentAreaInference::get_points(uint8* image, const uint image_height, const uint image_width)
{
    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(warp_size);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, m_dev_norm_x, m_dev_norm_y, image_width, image_height, m_height_samples);
    
    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, m_dev_scores, m_point_count, image_height, image_width);

    cudaMemcpy(m_hst_buffer, m_dev_buffer, m_buffer_size, cudaMemcpyDeviceToHost);

    std::vector<int> points_x, points_y, norm_x, norm_y, scores;
    for (int i = 0; i < m_point_count; i++)
    {
        points_x.push_back(m_hst_edge_x[i]);
        points_y.push_back(m_hst_edge_y[i]);
        norm_x.push_back(m_hst_norm_x[i]);
        norm_y.push_back(m_hst_norm_y[i]);
        scores.push_back(((int*)m_hst_scores)[i]);
    }

    std::vector<std::vector<int>> result = std::vector<std::vector<int>>();
    result.push_back(points_x);
    result.push_back(points_y);
    result.push_back(norm_x);
    result.push_back(norm_y);
    result.push_back(scores);

    return result;
}
