#include <cuda_runtime.h>
#include "../common.hpp"

namespace cuda
{
    __global__ void find_best_edge(const float* g_score_strips_batch, float* g_edge_x_batch, float* g_edge_y_batch, float* g_edge_scores_batch, const int strip_width, const int image_height, const int strip_count, const int half_patch_size)
    {
        int thread_count = blockDim.x;
        int warp_count = 1 + (thread_count - 1) / 32;

        extern __shared__ float s_shared_buffer[];
        float* s_cross_warp_operation_buffer = s_shared_buffer;
        float* s_cross_warp_operation_buffer_2 = s_shared_buffer + warp_count;

        const float* g_score_strips = g_score_strips_batch + blockIdx.z * strip_count * strip_width;
        float* g_edge_x = g_edge_x_batch + blockIdx.z * 3 * 2 * strip_count;
        float* g_edge_y = g_edge_y_batch + blockIdx.z * 3 * 2 * strip_count;
        float* g_edge_scores = g_edge_scores_batch + blockIdx.z * 3 * 2 * strip_count;

        int warp_index = threadIdx.x >> 5;
        int lane_index = threadIdx.x & 31;

        bool flip = blockIdx.x == 1;

        // ============================================================
        // Load strip into shared memory...

        int image_x = flip ? strip_width - 1 - threadIdx.x : threadIdx.x;

        int strip_index = blockIdx.y;
        int strip_height = 1 + (image_height - 2) / (1.0f + exp(-(strip_index - strip_count / 2.0f + 0.5f)/(strip_count / 8.0f)));
        
        float point_score = g_score_strips[image_x + strip_index * strip_width];
    
        int best_edge_x = image_x;
        float best_edge_score = point_score;
        
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

            int next_power_two = pow(2, ceil(logf(warp_count)/logf(2)));

            #pragma unroll
            for (int offset = next_power_two >> 1 ; offset > 0; offset >>= 1)
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
                int point_index = flip ? strip_index : strip_index + strip_count;
                
                g_edge_x[point_index] = best_edge_x + half_patch_size;
                g_edge_y[point_index] = strip_height;
                g_edge_scores[point_index] = best_edge_score;
            }
        }
    }

    void find_points_from_strip_scores(const float* strips, const int batch_count, const int image_height, const int image_width, const int strip_count, const int model_patch_size, float* points_x, float* points_y, float* point_score)
    {
        int half_patch_size = (model_patch_size - 1) / 2;
        int strip_width = image_width - 2 * half_patch_size;

        int half_width = strip_width / 2;
        int warps = 1 + (half_width - 1) / 32;
        int threads = warps * 32;

        threads = threads > 1024 ? 1024 : threads;

        dim3 grid(2, strip_count, batch_count);
        dim3 block(threads);
        int shared_memory = 2 * warps * sizeof(int);
        find_best_edge<<<grid, block, shared_memory>>>(strips, points_x, points_y, point_score, strip_width, image_height, strip_count, half_patch_size);
    }
}
