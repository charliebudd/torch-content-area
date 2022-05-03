#include <cuda_runtime.h>
#include "content_area_inference.cuh"

// TODO, expose params
#define MAX_CENTER_DIST 0.15 // * image width
#define MIN_RADIUS 0.2 // * image width
#define MAX_RADIUS 0.6 // * image width
#define EDGE_THRESHOLD 4
#define STRICT_EDGE_THRESHOLD 10.5
#define ANGLE_THRESHOLD 26

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
__global__ void find_points(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, const uint image_width, const uint image_height, const uint height_gap, const uint strip_count)
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
    int strip_height = (strip_index + 0.5) * height_gap;

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

    float x, y, r;
    bool valid = calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

    float x_diff = x - 0.5 * image_width;
    float y_diff = y - 0.5 * image_height;
    float diff = sqrt(x_diff * x_diff + y_diff * y_diff);

    // Filter out bad circles
    valid &= ax > 1;
    valid &= ax < image_width - 2;
    valid &= bx > 1;
    valid &= bx < image_width - 2;
    valid &= cx > 1;
    valid &= cx < image_width - 2;
    valid &= diff < MAX_CENTER_DIST * image_width;
    valid &= r > MIN_RADIUS * image_width;
    valid &= r < MAX_RADIUS * image_width;

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

            constexpr float dist = 4.0f;

            if (diff < dist)
            {
                score += 1 - diff / dist;
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

float distance_score(const uint height_samples, const uint i, const uint j)
{
    float x_diff = (i >= height_samples) != (j >= height_samples);
    float y_diff = abs(float(i % height_samples) - float(j % height_samples)) / height_samples;

    return sqrt((x_diff * x_diff + y_diff * y_diff) / 2);
}

void select_final_triple(const uint point_count, const float* scores, int* indices)
{
    float best_score = -1.0f;

    uint height_samples = point_count / 2;

    for (int i = 0; i < point_count; i++)
    {
        float score_i = scores[i];

        // if (score_i == 0)
        //     continue;

        for (int j = i+1; j < point_count; j++)
        {
            float score_j = scores[j];

            // if (score_j == 0)
            //     continue;

            for (int k = j+1; k < point_count; k++)
            {
                float score_k = scores[k];

                // if (score_k == 0)
                //     continue;

                float dist_score = distance_score(height_samples, i, j) + distance_score(height_samples, i, k) + distance_score(height_samples, j, k);
                float score = (score_i + score_j + score_k);

                score = score * (dist_score + 15);

                if (score > best_score)
                {
                    best_score = score;
                    
                    indices[0] = i;
                    indices[1] = j;
                    indices[2] = k;
                }
            }
        }
    }
}

#define warp_size 32
#define find_points_warp_count 16

ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    // #########################################################
    // Some useful values...

    uint height_gap = image_height / m_height_samples;

    // #########################################################
    // Finding candididate points...
    // A thread block for each point

    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, height_gap, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(warp_size);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, image_width, image_height, m_height_samples);

    // #########################################################
    // Evaluating candidate points...
    // A thread block for each point (left and right)
    // A thread per combination of the other two points in each triple.

    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, (float*)m_dev_scores, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results...

    cudaMemcpy(m_hst_block, m_dev_block, m_buffer_size, cudaMemcpyDeviceToHost);

    // #########################################################
    // Choosing the final points and calculating circle...

    int indices[3] {0, 1, 2};
    select_final_triple(m_point_count, (float*)m_hst_scores, indices);

    int ax = m_hst_edge_x[indices[0]];
    int ay = m_hst_edge_y[indices[0]];
    
    int by = m_hst_edge_y[indices[1]];
    int bx = m_hst_edge_x[indices[1]];
    
    int cx = m_hst_edge_x[indices[2]];
    int cy = m_hst_edge_y[indices[2]];

    float x, y, r;
    calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

    // #########################################################
    // Constructing final area to return...

    return ContentArea(x, y, r);
}


std::vector<std::vector<int>> ContentAreaInference::get_points(uint8* image, const uint image_height, const uint image_width)
{
    uint height_gap = image_height / m_height_samples;

    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * find_points_warp_count);
    find_points<find_points_warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, height_gap, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(warp_size);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, image_width, image_height, m_height_samples);
    
    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, (float*)m_dev_scores, m_point_count, image_height, image_width);

    cudaMemcpy(m_hst_block, m_dev_block, m_buffer_size, cudaMemcpyDeviceToHost);

    int indices[3];
    select_final_triple(m_point_count, (float*)m_hst_scores, indices);

    std::vector<int> points_x, points_y, norm_x, norm_y, scores;
    for (int i = 0; i < m_point_count; i++)
    {
        points_x.push_back(m_hst_edge_x[i]);
        points_y.push_back(m_hst_edge_y[i]);
        norm_x.push_back(m_hst_norm_x[i]);
        norm_y.push_back(m_hst_norm_y[i]);
        scores.push_back(m_hst_scores[i]);
    }

    std::vector<int> final_indices;
    for (int i = 0; i < 3; i++)
    {
        final_indices.push_back(indices[i]);
    }

    std::vector<std::vector<int>> result = std::vector<std::vector<int>>();
    result.push_back(points_x);
    result.push_back(points_y);
    result.push_back(norm_x);
    result.push_back(norm_y);
    result.push_back(scores);
    result.push_back(final_indices);

    return result;
}
