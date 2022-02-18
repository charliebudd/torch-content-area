#include <cuda_runtime.h>
#include "content_area_inference.cuh"

__device__ uint8 grayscale(uint8 r, uint8 g, uint8 b)
{
    return 0.2989 * r + 0.5870 * g + 0.1140 * b;
}

template<int warp_count>
__global__ void find_points(uint8* g_image, uint* g_points, const uint image_width, const uint image_height, const uint height_gap, const uint point_count)
{
    __shared__ bool s_is_edge[warp_count];
    __shared__ uint s_indicies[warp_count];

    bool flip = blockIdx.x == 1;

    uint point_index = blockIdx.y;

    uint image_x = flip ? image_width - 1 - threadIdx.x : threadIdx.x;
    uint image_y = (point_index + 0.5) * height_gap;

    uint warp_index = threadIdx.x >> 5;
    uint lane_index = threadIdx.x & 31;

    int home = grayscale(
        g_image[image_x + image_y * image_width + 0 * image_width * image_height],
        g_image[image_x + image_y * image_width + 1 * image_width * image_height],
        g_image[image_x + image_y * image_width + 2 * image_width * image_height]
    );

    uint neighbour_offset = flip ? -1 : 1;

    int neighbour = grayscale(
        g_image[image_x + neighbour_offset + image_y * image_width + 0 * image_width * image_height],
        g_image[image_x + neighbour_offset + image_y * image_width + 1 * image_width * image_height],
        g_image[image_x + neighbour_offset + image_y * image_width + 2 * image_width * image_height]
    );

    bool is_edge = abs(home - neighbour) > 6;
    uint index = threadIdx.x;

    // Finding warp max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        bool other_is_edge = __shfl_down_sync(0xffffffff, is_edge, offset);
        uint other_index = __shfl_down_sync(0xffffffff, index, offset);

        if ((other_is_edge && other_index < index) || (other_is_edge && !is_edge))
        {
            is_edge = other_is_edge;
            index = other_index;
        }
    }

    // Writing warp max to shared memory
    if (lane_index == 0)
    {    
        s_is_edge[warp_index] = is_edge;
        s_indicies[warp_index] = index;
    }

    // Syncing between warps
    __syncthreads();

    // Finding block max
    if (warp_index == 0 && lane_index < warp_count)
    {
        is_edge = s_is_edge[lane_index];
        index = s_indicies[lane_index];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            bool other_is_edge = __shfl_down_sync(0xffffffff, is_edge, offset);
            uint other_index = __shfl_down_sync(0xffffffff, index, offset);

            if ((other_is_edge && other_index < index) || (other_is_edge && !is_edge))
            {
                is_edge = other_is_edge;
                index = other_index;
            }
        }

        // Outputting result
        if (lane_index == 0)
        {
            int point_offset = flip ? point_count : 0;
            g_points[point_index + point_offset] = flip ? image_width - index - 1 : index;
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

__device__ void square_indices(const uint k, const uint n, uint* i, uint* j)
{
    *i = n - 2 - int(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
    *j = k + *i + 1 -  n * (n - 1) / 2 + (n - *i) * ((n - *i) - 1) / 2;
}

__global__ void check_triples(const uint* g_points, uint* g_point_scores, const uint height_gap, const uint point_count, const uint image_height, const uint image_width)
{
    // HARDCODED SIZE!!!!!!!!!!!!!!!
    __shared__ uint s_points[32];
    __shared__ uint s_scores[32];

    if (threadIdx.x < point_count)
    {
        s_points[threadIdx.x] = g_points[threadIdx.x];
    }

    const uint warp_count = (blockDim.x >> 5) + 1; /// +1 ????????????????
    const uint warp_index = threadIdx.x >> 5;
    const uint lane_index = threadIdx.x & 31;

    uint a_index, b_index, c_index;
    a_index = blockIdx.x;
    square_indices(threadIdx.x, point_count, &b_index, &c_index);

    float ax = s_points[a_index];
    float bx = s_points[b_index];
    float cx = s_points[c_index];

    float ay = height_gap * (0.5  + (a_index % (point_count / 2))); 
    float by = height_gap * (0.5  + (b_index % (point_count / 2)));
    float cy = height_gap * (0.5  + (c_index % (point_count / 2)));

    float x, y, r;
    bool valid = calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

    // Filter out bad circles
    uint score = valid; 
    score &= abs(x - 0.5 * image_width) < 0.1 * image_width;
    score &= abs(y - 0.5 * image_height) < 0.1 * image_height;
    score &= r > (0.3 * image_width);
    score &= r < (0.6 * image_width);

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
            g_point_scores[a_index] = score;
        }
    }
}

float distance_score(const uint point_count, const uint i, const uint j)
{
    float x_diff = ((i > point_count) != (j > point_count)); 
    float y_diff = abs(float(i) - j) / point_count;

    return sqrt((x_diff * x_diff + y_diff * y_diff) / 2);
}

void select_final_triple(const uint point_count, const uint* scores, int* indices)
{
    float best_value = 0.0f;
    float best_score = 0.0f;

    for (int i = 0; i < point_count; i++)
    {
        float score = scores[i];


        if (score > best_score)
        {
            best_score = score;
        }
    }

    for (int i = 0; i < point_count; i++)
    {
        float score_i = scores[i];

        for (int j = i+1; j < point_count; j++)
        {
            float score_j = scores[j];

            for (int k = j+1; k < point_count; k++)
            {
                float score_k = scores[k];

                float dist_value = distance_score(point_count, i, j) + distance_score(point_count, i, k) + distance_score(point_count, j, k);
                float score_value = score_i * score_j * score_k / (3 * best_score * best_score * best_score);

                float value = dist_value * score_value;

                if (value > best_value)
                {
                    best_value = value;
                    
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

// #define PROFILE

#ifdef PROFILE
#include <torch/extension.h>
#endif

Area ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
{
    #ifdef PROFILE
    cudaEvent_t a, b, c, d, e;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventCreate(&c);
    cudaEventCreate(&d);
    cudaEventCreate(&e);
    #endif

    // #########################################################
    // Some useful values...

    uint height_gap = image_height / m_height_samples;

    // #########################################################
    // Finding candididate points...
    // A thread block for each point
    #ifdef PROFILE
    cudaEventRecord(a);
    #endif  

    dim3 find_points_grid(2, m_height_samples);
    dim3 find_points_block(warp_size * warp_count, 1);
    find_points<warp_count><<<find_points_grid, find_points_block>>>(image, m_dev_points, image_width, image_height, height_gap, m_height_samples);
    
    // #########################################################
    // Evaluating candidate points...
    // A thread block for each point (left and right)
    // A thread per combination of the other two points in each triple.
    #ifdef PROFILE
    cudaEventRecord(b);
    #endif  

    dim3 check_triples_grid(m_point_count);
    dim3 check_triples_block(triangle_size(m_point_count));
    check_triples<<<check_triples_grid, check_triples_block>>>(m_dev_points, m_dev_scores, height_gap, m_point_count, image_height, image_width);

    // #########################################################
    // Reading back results and freeing cuda memory...
    #ifdef PROFILE
    cudaEventRecord(c);
    #endif  

    cudaMemcpy(m_hst_block, m_dev_block, 2 * m_point_count * sizeof(uint), cudaMemcpyDeviceToHost);

    // #########################################################
    // Choosing the final points and calculating circle...
    #ifdef PROFILE
    cudaEventRecord(d);
    #endif  

    int indices[3];
    select_final_triple(m_point_count, m_hst_scores, indices);

    float ax = m_hst_points[indices[0]];
    float bx = m_hst_points[indices[1]];
    float cx = m_hst_points[indices[2]];

    float ay = int(((indices[0] % m_height_samples) + 0.5) * height_gap);
    float by = int(((indices[1] % m_height_samples) + 0.5) * height_gap);
    float cy = int(((indices[2] % m_height_samples) + 0.5) * height_gap);

    float x, y, r;
    calculate_circle(ax, ay, bx, by, cx, cy, &x, &y, &r);

    // #########################################################
    // Constructing final area to return...
    #ifdef PROFILE
    cudaEventRecord(e);

    cudaEventSynchronize(a);
    cudaEventSynchronize(b);
    cudaEventSynchronize(c);
    cudaEventSynchronize(d);
    cudaEventSynchronize(e);

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, a, g);
    py::print("TOTAL:", milliseconds);
    cudaEventElapsedTime(&milliseconds, a, b);
    py::print("find points:", milliseconds);
    cudaEventElapsedTime(&milliseconds, b, c);
    py::print("check triples:", milliseconds);
    cudaEventElapsedTime(&milliseconds, c, d);
    py::print("read back points:", milliseconds);
    cudaEventElapsedTime(&milliseconds, d, e);
    py::print("choose final points:", milliseconds);
    #endif  

    Area area;
    area.type = Area::Circle;
    area.circle.y = y;
    area.circle.x = x;
    area.circle.r = r;

    return area;
}
