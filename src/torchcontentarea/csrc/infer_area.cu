#include <cuda_runtime.h>
#include "content_area_inference.cuh"

#define MAX_CENTER_DIST_X 0.3
#define MAX_CENTER_DIST_Y 0.3
#define MIN_RADIUS 0.2
#define MAX_RADIUS 0.6

__device__  float normed_euclidean(float* a, float* b)
{
    #define EUCLID_NORM 441.67f // max possible value... sqrt(3 * 255 ^ 2)
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])) / EUCLID_NORM;
}

template<int warp_count>
__global__ void find_points(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, const uint image_width, const uint image_height, const uint height_gap, const uint strip_count)
{
    constexpr uint warp_size = 32;
    constexpr uint thread_count = warp_count * warp_size;
    constexpr uint kernel_size = 3;
    constexpr uint kernel_offset = (kernel_size - 1) / 2;
    constexpr uint channel_count = 3;
    
    uint warp_index = threadIdx.x >> 5;
    uint lane_index = threadIdx.x & 31;

    __shared__ uint8 s_image_strip[thread_count * kernel_size * channel_count];
    __shared__ uint s_first_edge_reduction_buffer[warp_count];
    __shared__ uint s_last_edge_reduction_buffer[warp_count];

    // ============================================================
    // Load strip into shared memory, interpolating using nearest neighbour...

    int image_x = threadIdx.x * image_width / thread_count;
    int strip_index = blockIdx.x;
    int strip_height = (strip_index + 0.5) * height_gap;

    for (int y = 0; y < kernel_size; y++)
    {
        for (int c = 0; c < channel_count; c++)
        {
            int strip_element_index = threadIdx.x + y * thread_count + c * kernel_size * thread_count;
            int image_element_index = image_x + (strip_height + (y - kernel_offset)) * image_width + c * image_width * image_height;

            s_image_strip[strip_element_index] = g_image[image_element_index];
        }
    }
    
    __syncthreads();

    // ============================================================
    // Loading patch into local memory, 0 if out of bounds...

    uint8 image_patch[kernel_size][kernel_size][channel_count];

    #pragma unroll
    for (int x = 0; x < kernel_size; x++)
    {
        int strip_x = threadIdx.x + (x - kernel_offset);
        bool x_in_bounds = strip_x >= 0 && strip_x < thread_count;

        #pragma unroll
        for (int y = 0; y < kernel_size; y++)
        {
            #pragma unroll
            for (int c = 0; c < channel_count; c++)
            {
                image_patch[x][y][c] = x_in_bounds ? s_image_strip[strip_x + y * thread_count + c * kernel_size * thread_count] : 0;
            }
        }
    }

    // ============================================================
    // Applying sobel kernel to image patch...

    float a[channel_count];
    float b[channel_count];

    #pragma unroll
    for (int c = 0; c < channel_count; c++)
    {
        a[c] = (image_patch[0][0][c] + 2 * image_patch[0][1][c] + image_patch[0][2][c]) / 4;
        b[c] = (image_patch[2][0][c] + 2 * image_patch[2][1][c] + image_patch[2][2][c]) / 4;
    }

    float x_grad = normed_euclidean(a, b);

    #pragma unroll
    for (int c = 0; c < channel_count; c++)
    {
        a[c] = (image_patch[0][0][c] + 2 * image_patch[1][0][c] + image_patch[2][0][c]) / 4;
        b[c] = (image_patch[0][2][c] + 2 * image_patch[1][2][c] + image_patch[2][2][c]) / 4;
    }

    float y_grad = normed_euclidean(a, b);
    
    float grad = sqrt(x_grad * x_grad + y_grad * y_grad);

    bool is_edge = grad > 0.040;

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
    constexpr uint warp_count = 8;
    constexpr uint warp_size = 32;
    constexpr uint kernel_size = 3;
    constexpr uint kernel_offset = (kernel_size - 1) / 2;
    constexpr uint channel_count = 3;

    constexpr uint patch_size = 16;
    
    uint warp_index = (threadIdx.x + 16 * threadIdx.y) >> 5;
    uint lane_index = (threadIdx.x + 16 * threadIdx.y) & 31;

    __shared__ uint8 s_image_patch[patch_size][patch_size];
    __shared__ uint s_reduction_buffer[warp_count * 7];
    
    // ============================================================
    // Get approximate point position...

    int point_index = blockIdx.x;

    uint point_x = g_edge_x[point_index];
    uint point_y = g_edge_y[point_index];

    if (point_x < 1 || point_x > image_width - 2)
        return;

    // ============================================================
    // Load patch centered around point position into shared memory...

    int patch_x = threadIdx.x;
    int patch_y = threadIdx.y;

    int image_x = point_x + (patch_x - patch_size / 2);
    int image_y = point_y + (patch_y - patch_size / 2);

    int patch_index = patch_x + patch_y * patch_size;
    int image_index = image_x + image_y * image_width;
    
    s_image_patch[patch_x][patch_y] = 0.2126 * g_image[image_index + 0 * image_width * image_height]
                                    + 0.7152 * g_image[image_index + 1 * image_width * image_height]
                                    + 0.0722 * g_image[image_index + 2 * image_width * image_height];
    
    __syncthreads();
    
    // ============================================================
    // Apply sobel filter...

    float x_grad = 0;
    float y_grad = 0;

    if (patch_x > 0 && patch_x < patch_size - 1 && patch_y > 0 && patch_y < patch_size - 1)
    {
        float left  = 0.25 * s_image_patch[patch_x - 1][patch_y - 1] + 0.5 * s_image_patch[patch_x - 1][patch_y] + 0.25 * s_image_patch[patch_x - 1][patch_y + 1];
        float right = 0.25 * s_image_patch[patch_x + 1][patch_y - 1] + 0.5 * s_image_patch[patch_x + 1][patch_y] + 0.25 * s_image_patch[patch_x + 1][patch_y + 1];
        x_grad = right - left;

        float top = 0.25 * s_image_patch[patch_x - 1][patch_y - 1] + 0.5 * s_image_patch[patch_x][patch_y - 1] + 0.25 * s_image_patch[patch_x + 1][patch_y - 1];
        float bot = 0.25 * s_image_patch[patch_x - 1][patch_y + 1] + 0.5 * s_image_patch[patch_x][patch_y + 1] + 0.25 * s_image_patch[patch_x + 1][patch_y + 1];
        y_grad = bot - top;
    }

    float edge_strength = sqrt(x_grad * x_grad + y_grad * y_grad);

    // ============================================================
    // Reduce values...
    // Reduce: sum_ix, sum_iy, sum_ixiy, sum_ixix, sum_gx, sum_gy

    bool is_edge = edge_strength > 3;

    int n = is_edge;
    int ix = image_x * is_edge;
    int iy = image_y * is_edge;
    int ixiy = image_x * image_y * is_edge;
    int ixix = image_x * image_x * is_edge;
    float gx = x_grad * is_edge;
    float gy = y_grad * is_edge;
    
    // warp reduction....
    #pragma unroll
    for (int offset = warp_size >> 1; offset > 0; offset >>= 1)
    {
        n    += __shfl_down_sync(0xffffffff, n    , offset);
        ix   += __shfl_down_sync(0xffffffff, ix   , offset);
        iy   += __shfl_down_sync(0xffffffff, iy   , offset);
        ixiy += __shfl_down_sync(0xffffffff, ixiy , offset);
        ixix += __shfl_down_sync(0xffffffff, ixix , offset);
        gx   += __shfl_down_sync(0xffffffff, gx   , offset);
        gy   += __shfl_down_sync(0xffffffff, gy   , offset);
    }

    if (lane_index == 0)
    {
        s_reduction_buffer[warp_index + 0 * warp_count] = n;
        s_reduction_buffer[warp_index + 1 * warp_count] = ix;
        s_reduction_buffer[warp_index + 2 * warp_count] = iy;
        s_reduction_buffer[warp_index + 3 * warp_count] = ixiy;
        s_reduction_buffer[warp_index + 4 * warp_count] = ixix;
        ((float*)s_reduction_buffer)[warp_index + 5 * warp_count] = gx;
        ((float*)s_reduction_buffer)[warp_index + 6 * warp_count] = gy;
    }

    __syncthreads();

    // block reduction....
    if (warp_index == 0 && lane_index < warp_count)
    {
        n    = s_reduction_buffer[lane_index + 0 * warp_count];
        ix   = s_reduction_buffer[lane_index + 1 * warp_count];
        iy   = s_reduction_buffer[lane_index + 2 * warp_count];
        ixiy = s_reduction_buffer[lane_index + 3 * warp_count];
        ixix = s_reduction_buffer[lane_index + 4 * warp_count];
        gx   = ((float*)s_reduction_buffer)[lane_index + 5 * warp_count];
        gy   = ((float*)s_reduction_buffer)[lane_index + 6 * warp_count];

        #pragma unroll
        for (int offset = warp_count >> 1 ; offset > 0; offset >>= 1)
        {
            n    += __shfl_down_sync(0xffffffff, n    , offset);
            ix   += __shfl_down_sync(0xffffffff, ix   , offset);
            iy   += __shfl_down_sync(0xffffffff, iy   , offset);
            ixiy += __shfl_down_sync(0xffffffff, ixiy , offset);
            ixix += __shfl_down_sync(0xffffffff, ixix , offset);
            gx   += __shfl_down_sync(0xffffffff, gx   , offset);
            gy   += __shfl_down_sync(0xffffffff, gy   , offset);
        }

        if (lane_index == 0)
        {
            float x_mean = (float)ix / n;
            float y_mean = (float)iy / n;

            float norm = sqrt(gx * gx + gy * gy);

            gx /= norm;
            gy /= norm;

            float ex = (image_width / 2) - x_mean;
            float ey = (image_height / 2) - y_mean;

            norm = sqrt(ex * ex + ey * ey);
            ex /= norm;
            ey /= norm;

            float dot = ex * gx + ey * gy;

            g_edge_x[point_index] = dot > 0.9 ? point_x : 0;
            // g_edge_x[point_index] = point_x;
            g_edge_y[point_index] = point_y;

            // g_edge_x[point_index] = dot > 0.95 ? x_mean : 0;
            // // g_edge_x[point_index] = x_mean;
            // g_edge_y[point_index] = y_mean;

            g_norm_x[point_index] = gx;
            g_norm_y[point_index] = gy;
        }
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

__device__ void square_indices(const int k, const int n, uint* i, uint* j)
{
    *i = n - 2 - int(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    *j = k + *i + 1 -  n * (n - 1) / 2 + (n - *i) * ((n - *i) - 1) / 2;
}

__global__ void check_triples(const uint* g_edge_x, const uint* g_edge_y, float* g_norm_x, float* g_norm_y, uint* g_edge_scores, const uint point_count, const uint image_height, const uint image_width)
{
    // HARDCODED SIZE!!!!!!!!!!!!!!!
    __shared__ uint s_edge_x[32];
    __shared__ uint s_edge_y[32];
    __shared__ float s_norm_x[32];
    __shared__ float s_norm_y[32];
    __shared__ uint s_scores[32];

    if (threadIdx.x < point_count)
    {
        s_edge_x[threadIdx.x] = g_edge_x[threadIdx.x];
        s_edge_y[threadIdx.x] = g_edge_y[threadIdx.x];
        s_norm_x[threadIdx.x] = g_norm_x[threadIdx.x];
        s_norm_y[threadIdx.x] = g_norm_y[threadIdx.x];
    }

    const uint warp_count = (blockDim.x >> 5) + 1; /// +1 ????????????????
    const uint warp_index = threadIdx.x >> 5;
    const uint lane_index = threadIdx.x & 31;

    uint a_index, b_index, c_index;
    a_index = blockIdx.x;
    square_indices(threadIdx.x, point_count, &b_index, &c_index);

    float ax = s_edge_x[a_index];
    float ay = s_edge_y[a_index];
    float anx = s_norm_x[a_index];
    float any = s_norm_y[a_index];
    
    float bx = s_edge_x[b_index];
    float by = s_edge_y[b_index];
    float bnx = s_norm_x[b_index];
    float bny = s_norm_y[b_index];
    
    float cx = s_edge_x[c_index];
    float cy = s_edge_y[c_index];
    float cnx = s_norm_x[c_index];
    float cny = s_norm_y[c_index];

    float x, y, r;
    bool valid = calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

    // Filter out bad circles
    uint score = valid;
    score &= ax > 1;
    score &= ax < image_width - 2;
    score &= bx > 1;
    score &= bx < image_width - 2;
    score &= cx > 1;
    score &= cx < image_width - 2;
    score &= abs(x - 0.5 * image_width) < MAX_CENTER_DIST_X * 0.5 * image_width;
    score &= abs(y - 0.5 * image_height) < MAX_CENTER_DIST_Y * 0.5 * image_height;
    score &= r > (MIN_RADIUS * image_width);
    score &= r < (MAX_RADIUS * image_width);

    if (score)
    {
        score = 0;

        for (int i=0; i < point_count; i++)
        {
            float py = s_edge_y[i];
            float px = s_edge_x[i];
            float pnx = s_norm_x[i];
            float pny = s_norm_y[i];

            float diff_x = (px - x);
            float diff_y = (py - y);

            float diff = diff_x * diff_x + diff_y * diff_y;
            uint d = abs(diff - r * r);

            if (d < 2000)
            {
                score += 2000 - d;
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
        s_scores[warp_index] = score;
    }

    // Syncing between warps
    __syncthreads();

    // Block reduction
    if (warp_index == 0 && lane_index < warp_count)
    {
        score = s_scores[lane_index];
        
        #pragma unroll
        for (int offset = warp_count / 2; offset > 0; offset /= 2)
        {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }

        // Outputting result
        if (lane_index == 0)
        {
            g_edge_scores[a_index] = score;
        }
    }
}

float distance_score(const uint height_samples, const uint i, const uint j)
{
    float x_diff = (i >= height_samples) != (j >= height_samples);
    float y_diff = abs(float(i % height_samples) - float(j % height_samples)) / height_samples;

    return sqrt((x_diff * x_diff + y_diff * y_diff) / 2);
}

void select_final_triple(const uint point_count, const uint* scores, int* indices)
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
#define warp_count 16

ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    // #########################################################
    // Some useful values...

    uint height_gap = image_height / m_height_samples;

    // #########################################################
    // Finding candididate points...
    // A thread block for each point

    dim3 find_points_grid(m_height_samples);
    dim3 find_points_block(warp_size * warp_count);
    find_points<warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, height_gap, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(16, 16);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, image_width, image_height, m_height_samples);
    
    // #########################################################
    // Evaluating candidate points...
    // A thread block for each point (left and right)
    // A thread per combination of the other two points in each triple.

    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, m_dev_scores, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results and freeing cuda memory...

    cudaMemcpy(m_hst_block, m_dev_block, m_buffer_size, cudaMemcpyDeviceToHost);

    // #########################################################
    // Choosing the final points and calculating circle...

    int indices[3] {0, 1, 2};
    select_final_triple(m_point_count, m_hst_scores, indices);

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
    dim3 find_points_block(warp_size * warp_count);
    find_points<warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, image_width, image_height, height_gap, m_height_samples);

    dim3 refine_points_grid(m_point_count);
    dim3 refine_points_block(16, 16);
    refine_points<<<refine_points_grid, refine_points_block>>>(image, m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, image_width, image_height, m_height_samples);

    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_edge_x, m_dev_edge_y, (float*)m_dev_norm_x, (float*)m_dev_norm_y, m_dev_scores, m_point_count, image_height, image_width);

    cudaMemcpy(m_hst_block, m_dev_block, m_buffer_size, cudaMemcpyDeviceToHost);

    int indices[3];
    select_final_triple(m_point_count, m_hst_scores, indices);

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
