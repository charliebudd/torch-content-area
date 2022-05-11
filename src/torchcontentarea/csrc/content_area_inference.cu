#include <cuda_runtime.h>
#include "content_area_inference.cuh"

ContentAreaInference::ContentAreaInference()
{
    m_height_samples = 8;
    m_point_count = 2 * m_height_samples;
    m_buffer_size = 5 * m_point_count * sizeof(uint);

    // cudaMallocHost(&m_hst_buffer, m_buffer_size);
    // m_hst_edge_x = (uint*)m_hst_buffer + 0 * m_point_count;
    // m_hst_edge_y = (uint*)m_hst_buffer + 1 * m_point_count;
    // m_hst_norm_x = (float*)m_hst_buffer + 2 * m_point_count;
    // m_hst_norm_y = (float*)m_hst_buffer + 3 * m_point_count;
    // m_hst_scores = (float*)m_hst_buffer + 4 * m_point_count;

    cudaMalloc(&m_dev_buffer, m_buffer_size);
    m_dev_edge_x = (uint*)m_dev_buffer + 0 * m_point_count;
    m_dev_edge_y = (uint*)m_dev_buffer + 1 * m_point_count;
    m_dev_norm_x = (float*)m_dev_buffer + 2 * m_point_count;
    m_dev_norm_y = (float*)m_dev_buffer + 3 * m_point_count;
    m_dev_scores = (float*)m_dev_buffer + 4 * m_point_count;
    

    int mapped_buffer_sixe = 3 * m_point_count * sizeof(uint);
    cudaMallocHost(&m_hst_mapped, mapped_buffer_sixe, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&m_dev_mapped, m_hst_mapped, 0);

    m_hst_edge_x = (uint* )m_hst_mapped + 0 * m_point_count;
    m_hst_edge_y = (uint* )m_hst_mapped + 1 * m_point_count;
    m_hst_scores = (float*)m_hst_mapped + 2 * m_point_count;
    
    m_dev_mapped_edge_x = (uint* )m_dev_mapped + 0 * m_point_count;
    m_dev_mapped_edge_y = (uint* )m_dev_mapped + 1 * m_point_count;
    m_dev_mapped_scores = (float*)m_dev_mapped + 2 * m_point_count;
}

ContentAreaInference::~ContentAreaInference()
{
    cudaFree(m_dev_buffer);
    // cudaFreeHost(m_hst_buffer);
    cudaFreeHost(m_hst_mapped);
}

// ContentArea ContentAreaInference::infer_area(uint8* image, const uint image_height, const uint image_width)
// {
//     ### Defined in "infer_area.cu" ###
// }

// void ContentAreaInference::draw_area(const ContentArea area, uint8* mask, const uint mask_height, const uint mask_width)
// {
//     ### Defined in "draw_area.cu" ###
// }

// void ContentAreaInference::crop_area(const ContentArea area, const uint8* src_image, uint8* dst_image, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, const InterpolationMode interpolation_mode)
// {
//     ### Defined in "crop_area.cu" ###
// }
