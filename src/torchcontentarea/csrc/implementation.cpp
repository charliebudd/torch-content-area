#include <cuda_runtime.h>
#include "implementation.hpp"
#include "cpu_functions.hpp"
#include "cuda_functions.cuh"

#define IMAGE_DTYPE_ERROR_MSG std::string("Unsupported image dtype.")
#define IMAGE_NDIM_ERROR_MSG(d) std::string("Expected an image tensor with 4 dimensions but found . Is you Image in NCHW format?").insert(53, std::to_string(d))
#define IMAGE_CHANNEL_ERROR_MSG(c) std::string("Expected a grayscale or RGB image but found size  at position 1. Is you Image in NCHW format?").insert(49, std::to_string(c))

#define POINTS_NDIM_ERROR_MSG(d) std::string("Expected a point tensor with 3 dimensions but found .").insert(52, std::to_string(d))
#define POINTS_CHANNEL_ERROR_MSG(d) std::string("Expected a point tensor with 3 channels but found .").insert(50, std::to_string(d))

ImageFormat check_image_tensor(torch::Tensor &image)
{
    image = image.contiguous();

    if (image.ndimension() != 4)
    {
        throw std::runtime_error(IMAGE_NDIM_ERROR_MSG(image.ndimension()));
    }

    if (image.size(1) != 1 && image.size(1) != 3)
    {
        throw std::runtime_error(IMAGE_CHANNEL_ERROR_MSG(image.size(1)));
    }

    bool is_rgb = image.size(1) == 3;
    switch (torch::typeMetaToScalarType(image.dtype()))
    {
        case (torch::kFloat): return is_rgb ? rgb_float : gray_float;
        case (torch::kDouble): return is_rgb ? rgb_double : gray_double;
        case (torch::kByte): return is_rgb ? rgb_uint8 : gray_uint8;
        case (torch::kInt): return is_rgb ? rgb_int : gray_int;
        default: throw std::runtime_error(IMAGE_DTYPE_ERROR_MSG);
    }
}

void check_points(torch::Tensor &points)
{
    points = points.contiguous();

    if (points.ndimension() != 3)
    {
        throw std::runtime_error(POINTS_NDIM_ERROR_MSG(points.ndimension()));
    }

    if (points.size(1) != 3 )
    {
        throw std::runtime_error(POINTS_CHANNEL_ERROR_MSG(points.size(1)));
    }
}

torch::Tensor estimate_area_handcrafted(torch::Tensor image, int strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds)
{
    ImageFormat image_format = check_image_tensor(image);

    int batch_count = image.size(0);
    int image_height = image.size(2);
    int image_width = image.size(3);
    int point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 4}, torch::device(image.device()).dtype(torch::kFloat32));

    if (image.device().is_cpu())
    {
        void* temp_buffer = malloc(3 * batch_count * point_count * sizeof(int));
        int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count;
        int*  points_y = (int*) temp_buffer + 1 * batch_count * point_count;
        float* points_s = (float*)temp_buffer + 2 * batch_count * point_count;

        cpu::find_points(image.data_ptr<uint8>(), image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
       
        cpu::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());
        
        free(temp_buffer);
    }
    else
    {
        void* temp_buffer;
        cudaMalloc(&temp_buffer, 3 * batch_count * point_count * sizeof(int));
        int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count; 
        int*  points_y = (int*) temp_buffer + 1 * batch_count * point_count; 
        float* points_s = (float*)temp_buffer + 2 * batch_count * point_count;

        cuda::find_points(image.data_ptr<uint8>(), image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
       
        cuda::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());

        cudaFree(temp_buffer);
    }

    return result;
}

torch::Tensor estimate_area_learned(torch::Tensor image, int strip_count, torch::jit::Module model, int model_patch_size, ConfidenceThresholds confidence_thresholds)
{
    ImageFormat image_format = check_image_tensor(image);

    int batch_count = image.size(0);
    int image_height = image.size(2);
    int image_width = image.size(3);
    int point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 4}, torch::device(image.device()).dtype(torch::kFloat32));
    torch::Tensor strips = torch::empty({batch_count * strip_count, 5, model_patch_size, image_width}, torch::device(image.device()).dtype(torch::kFloat32));
    std::vector<torch::jit::IValue> model_input = {strips};

    
    if (image.device().is_cpu())
    {
        void* temp_buffer = malloc(3 * batch_count * point_count * sizeof(int));
        int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count;
        int*  points_y = (int*) temp_buffer + 1 * batch_count * point_count;
        float* points_s = (float*)temp_buffer + 2 * batch_count * point_count;
        
        cpu::make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());

        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cpu::find_points_from_strip_scores(strip_scores.data_ptr<float>(), image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
        
        cpu::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());

        free(temp_buffer);
    }
    else
    { 
        void* temp_buffer;
        cudaMalloc(&temp_buffer, 3 * batch_count * point_count * sizeof(int));
        int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count; 
        int*  points_y = (int*)temp_buffer + 1 * batch_count * point_count;
        float* points_s = (float*)temp_buffer + 2 * batch_count * point_count; 
        
        cuda::make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cuda::find_points_from_strip_scores(strip_scores.data_ptr<float>(), image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
        
        cuda::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());

        cudaFree(temp_buffer);
    }

    return result;
}

torch::Tensor get_points_handcrafted(torch::Tensor image, int strip_count, FeatureThresholds feature_thresholds)
{
    ImageFormat image_format = check_image_tensor(image);

    int batch_count = image.size(0);
    int image_height = image.size(2);
    int image_width = image.size(3);
    int point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 3, point_count}, torch::device(image.device()).dtype(torch::kFloat32));

    void* temp_buffer = (void*)result.data_ptr<float>();
    int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count;
    int*  points_y = (int*)temp_buffer + 1 * batch_count * point_count;
    float* points_s = (float*)temp_buffer + 2 * batch_count * point_count;
    
    if (image.device().is_cpu())
    {
        cpu::find_points(image.data_ptr<uint8>(), image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
    }
    else
    {
        cuda::find_points(image.data_ptr<uint8>(), image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
    }

    return result;
}

torch::Tensor get_points_learned(torch::Tensor image, int strip_count, torch::jit::Module model, int model_patch_size)
{
    ImageFormat image_format = check_image_tensor(image);

    int batch_count = image.size(0);
    int image_height = image.size(2);
    int image_width = image.size(3);
    int point_count = 2 * strip_count;

    torch::Tensor result = torch::empty({batch_count, 3, point_count}, torch::device(image.device()).dtype(torch::kFloat32));
    torch::Tensor strips = torch::empty({batch_count * strip_count, 5, model_patch_size, image_width}, torch::device(image.device()).dtype(torch::kFloat32));
    std::vector<torch::jit::IValue> model_input = {strips};

    void* temp_buffer = (void*)result.data_ptr<float>();
    int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count; 
    int*  points_y = (int*)temp_buffer + 1 * batch_count * point_count;
    float* points_s = (float*)temp_buffer + 2 * batch_count * point_count; 
    
    if (image.device().is_cpu())
    {
        cpu::make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cpu::find_points_from_strip_scores(strip_scores.data_ptr<float>(), image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
    }
    else
    {
        cuda::make_strips(image.data_ptr<uint8>(), image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cuda::find_points_from_strip_scores(strip_scores.data_ptr<float>(), image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
    }

    return result;
}

torch::Tensor fit_area(torch::Tensor points, py::tuple image_size, ConfidenceThresholds confidence_thresholds)
{
    check_points(points);

    int batch_count = points.size(0);
    int image_height = image_size[0].cast<int>();
    int image_width = image_size[1].cast<int>();
    int point_count = points.size(2);

    torch::Tensor result = torch::empty({batch_count, 4}, torch::device(points.device()).dtype(torch::kFloat32));
    
    void* temp_buffer = (void*)points.data_ptr<float>();
    int*  points_x = (int*) temp_buffer + 0 * batch_count * point_count; 
    int*  points_y = (int*)temp_buffer + 1 * batch_count * point_count;
    float* points_s = (float*)temp_buffer + 2 * batch_count * point_count; 

    if (points.device().is_cpu())
    {
        cpu::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());
    }
    else
    {
        cuda::fit_circle(points_x, points_y, points_s, point_count, confidence_thresholds, image_height, image_width, result.data_ptr<float>());
    }

    return result;
}
