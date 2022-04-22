#include <cuda_runtime.h>
#include "content_area_inference.cuh"

ContentAreaInference::ContentAreaInference()
{
    m_height_samples = 8;
    m_point_count = 2 * m_height_samples;
    m_buffer_size = 5 * m_point_count * sizeof(uint);

    cudaMalloc(&m_dev_block, m_buffer_size);
    m_dev_edge_x = m_dev_block + 0 * m_point_count;
    m_dev_edge_y = m_dev_block + 1 * m_point_count;
    m_dev_norm_x = m_dev_block + 2 * m_point_count;
    m_dev_norm_y = m_dev_block + 3 * m_point_count;
    m_dev_scores = m_dev_block + 4 * m_point_count;
    
    cudaMallocHost((void**)&m_hst_block, m_buffer_size);
    m_hst_edge_x = m_hst_block + 0 * m_point_count;
    m_hst_edge_y = m_hst_block + 1 * m_point_count;
    m_hst_norm_x = m_hst_block + 2 * m_point_count;
    m_hst_norm_y = m_hst_block + 3 * m_point_count;
    m_hst_scores = m_hst_block + 4 * m_point_count;
}

ContentAreaInference::~ContentAreaInference()
{
    cudaFree(m_dev_block);
    cudaFreeHost(m_hst_block);
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
