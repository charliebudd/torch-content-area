#include "infer_area_cuda.cuh"

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

// =========================================================================
// Kernels...

__global__ void find_points_kernel(const uint8* g_image, uint* g_edge_x, uint* g_edge_y, float* g_edge_scores, const uint image_width, const uint image_height, const uint strip_count, const FeatureThresholds feature_thresholds)
{
    constexpr uint warp_size = 32;

    int thread_count = blockDim.x;
    int warp_count = 1 + (thread_count - 1) / warp_size;

    extern __shared__ uint s_shared_buffer[];
    float* s_image_strip = (float*)s_shared_buffer;
    uint* s_cross_warp_operation_buffer = s_shared_buffer + 3 * thread_count;
    float* s_cross_warp_operation_buffer_2 = (float*)(s_shared_buffer + 3 * thread_count + warp_count);

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

    float center_dir_x = (0.5f * image_width) - (float)image_x;
    float center_dir_y = (0.5f * image_height) - (float)strip_height;
    float center_dir_norm = sqrt(center_dir_x * center_dir_x + center_dir_y * center_dir_y);
 
    x_grad = flip ? -x_grad : x_grad;

    float dot = grad == 0 ? -1 : (center_dir_x * x_grad + center_dir_y * y_grad) / (center_dir_norm * grad);
    float angle = RAD2DEG * acos(dot);

    // ============================================================
    // Final scoring...

    float edge_score = tanh(grad / feature_thresholds.edge);
    float angle_score = 1.0f - tanh(angle / feature_thresholds.angle);
    float intensity_score = 1.0f - tanh(max_preceeding_intensity / feature_thresholds.intensity);

    float point_score = edge_score * angle_score * intensity_score;

    // ============================================================
    // Reduction to find the best edge...

    int best_edge_x = image_x;
    float best_edge_score = point_score;
    
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
            
            if (best_edge_x < DISCARD_BORDER || best_edge_x >= image_width - DISCARD_BORDER)
            {
                best_edge_score = 0.0f;
            }

            g_edge_x[point_index] = best_edge_x;
            g_edge_y[point_index] = strip_height;
            g_edge_scores[point_index] = best_edge_score;
        }
    }
}

// =========================================================================
// Main function...

void find_points(const uint8* image, const uint image_height, const uint image_width, const uint strip_count, const FeatureThresholds feature_thresholds, uint* points_x, uint* points_y, float* point_scores)
{
    int half_width = image_width / 2;
    int warps = 1 + (half_width - 1) / 32;
    int threads = warps * 32;

    dim3 find_points_grid(2, strip_count);
    dim3 find_points_block(threads);
    find_points_kernel<<<find_points_grid, find_points_block, (3 * threads + 2 * warps) * sizeof(int)>>>(image, points_x, points_y, point_scores, image_width, image_height, strip_count, feature_thresholds);
}
