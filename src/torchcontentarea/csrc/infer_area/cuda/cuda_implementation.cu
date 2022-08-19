#include <torch/extension.h>
#include "../infer_area.h"
#include "infer_area_cuda.cuh"

torch::Tensor InferAreaHandcrafted::cuda_implementation(torch::Tensor image, uint strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds)
{
    uint batch_count = image.size(0);
    uint image_channels = image.size(1);
    uint image_height = image.size(2);
    uint image_width = image.size(3);
    uint point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 4}, torch::device(image.device()).dtype(torch::kFloat32));

    void* temp_buffer;
    cudaMalloc(&temp_buffer, 3 * batch_count * point_count * sizeof(uint));
    uint*  points_x = (uint*) temp_buffer + 0 * batch_count * point_count; 
    uint*  points_y = (uint*)temp_buffer + 1 * batch_count * point_count; 
    float* points_s = (float*)temp_buffer + 2 * batch_count * point_count; 

    find_points(image.data_ptr<uint8>(), image_height, image_width, strip_count, points_x, points_y, points_s);
    fit_circle(points_x, points_y, points_s, point_count, image_height, image_width, result.data_ptr<float>());

    cudaFree(temp_buffer);

    return result;
}

torch::Tensor InferAreaLearned::cuda_implementation(torch::Tensor image, uint strip_count, torch::jit::Module model, uint model_patch_size, ConfidenceThresholds confidence_thresholds)
{
    uint batch_count = image.size(0);
    uint image_channels = image.size(1);
    uint image_height = image.size(2);
    uint image_width = image.size(3);
    uint point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 4}, torch::device(image.device()).dtype(torch::kFloat32));
    torch::Tensor strips = torch::empty({batch_count * strip_count, 5, model_patch_size, image_width}, torch::device(image.device()).dtype(torch::kFloat32));
    std::vector<torch::jit::IValue> model_input = {strips};

    //###################################################
    // make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
    // return strips;
    
    void* temp_buffer;
    cudaMalloc(&temp_buffer, 3 * batch_count * point_count * sizeof(uint));
    uint*  points_x = (uint*) temp_buffer + 0 * batch_count * point_count; 
    uint*  points_y = (uint*)temp_buffer + 1 * batch_count * point_count;
    float* points_s = (float*)temp_buffer + 2 * batch_count * point_count; 
    
    make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
    
    torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

    find_points_from_strip_scores(strip_scores.data_ptr<float>(), image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
    
    fit_circle(points_x, points_y, points_s, point_count, image_height, image_width, result.data_ptr<float>());

    cudaFree(temp_buffer);

    return result;
}
