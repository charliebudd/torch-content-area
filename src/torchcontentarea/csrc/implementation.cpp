#include <cuda_runtime.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include "implementation.hpp"
#include "cpu_functions.hpp"
#include "cuda_functions.cuh"

#define IMAGE_DTYPE_ERROR_MSG(t) std::string("Unsupported image dtype .").insert(24, torch::utils::getDtypeNames(t).second)
#define IMAGE_NDIM_ERROR_MSG(d) std::string("Expected an image tensor with 3 or 4 dimensions but found .").insert(58, std::to_string(d))
#define IMAGE_CHANNEL_ERROR_MSG(c) std::string("Expected a grayscale or RGB image but found size  at position 1.").insert(49, std::to_string(c))
#define POINTS_NDIM_ERROR_MSG(d) std::string("Expected a point tensor with 2 or 3 dimensions but found .").insert(52, std::to_string(d))
#define POINTS_CHANNEL_ERROR_MSG(d) std::string("Expected a point tensor with 3 channels but found .").insert(50, std::to_string(d))

void check_image_tensor(torch::Tensor &image)
{
    image = image.contiguous();

    if (image.dim() != 3 && image.dim() != 4)
    {
        throw std::runtime_error(IMAGE_NDIM_ERROR_MSG(image.dim()));
    }

    if (image.size(-3) != 1 && image.size(-3) != 3)
    {
        throw std::runtime_error(IMAGE_CHANNEL_ERROR_MSG(image.size(1)));
    }

    switch (torch::typeMetaToScalarType(image.dtype()))
    {
        case (torch::kFloat): break;
        case (torch::kDouble): break;
        case (torch::kByte): break;
        case (torch::kInt): break;
        case (torch::kLong): break;
        default: throw std::runtime_error(IMAGE_DTYPE_ERROR_MSG(torch::typeMetaToScalarType(image.dtype())));
    }
}

void check_points(torch::Tensor &points)
{
    points = points.contiguous();

    if (points.dim() != 2 && points.dim() != 3)
    {
        throw std::runtime_error(POINTS_NDIM_ERROR_MSG(points.dim()));
    }

    if (points.size(-2) != 3)
    {
        throw std::runtime_error(POINTS_CHANNEL_ERROR_MSG(points.size(1)));
    }
}

Image get_image_data(torch::Tensor image)
{
    bool is_rgb = image.size(-3) == 3;
    switch (torch::typeMetaToScalarType(image.dtype()))
    {
        case (torch::kFloat):  return Image(is_rgb ? ImageFormat::rgb_float  : ImageFormat::gray_float,  (void*)image.data_ptr<float   >());
        case (torch::kDouble): return Image(is_rgb ? ImageFormat::rgb_double : ImageFormat::gray_double, (void*)image.data_ptr<double  >());
        case (torch::kByte):   return Image(is_rgb ? ImageFormat::rgb_uint8  : ImageFormat::gray_uint8,  (void*)image.data_ptr<uint8   >());
        case (torch::kInt):    return Image(is_rgb ? ImageFormat::rgb_int    : ImageFormat::gray_int,    (void*)image.data_ptr<int     >());
        case (torch::kLong):   return Image(is_rgb ? ImageFormat::rgb_long   : ImageFormat::gray_long,   (void*)image.data_ptr<long int>());
        default: throw std::runtime_error(IMAGE_DTYPE_ERROR_MSG(torch::typeMetaToScalarType(image.dtype())));
    }
}

std::tuple<torch::Tensor, FitCircleStatus> estimate_area_handcrafted(torch::Tensor image, int strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds)
{
    check_image_tensor(image);

    Image image_data = get_image_data(image);

    bool batched = image.dim() == 4;

    int batch_count = batched ? image.size(0) : 1;
    int channel_count = image.size(-3);
    int image_height = image.size(-2);
    int image_width = image.size(-1);
    int point_count = 2 * strip_count;

    torch::TensorOptions options = torch::device(image.device()).dtype(torch::kFloat32);
    auto result = batched ?
        std::make_tuple(torch::empty({batch_count, 4}, options), FitCircleStatus::invalid) : 
        std::make_tuple(torch::empty({4}, options), FitCircleStatus::invalid);

    if (image.device().is_cpu())
    {
        float* temp_buffer = (float*)malloc(3 * batch_count * point_count * sizeof(float));
        float* points_x = temp_buffer + 0 * point_count;
        float* points_y = temp_buffer + 1 * point_count;
        float* points_s = temp_buffer + 2 * point_count;

        cpu::find_points(image_data, batch_count, channel_count, image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
       
        cpu::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>(), &std::get<FitCircleStatus>(result));
        
        free(temp_buffer);
    }
    else
    {
        float* temp_buffer;
        cudaMalloc((void**)&temp_buffer, 3 * batch_count * point_count * sizeof(float));
        float* points_x = temp_buffer + 0 * point_count;
        float* points_y = temp_buffer + 1 * point_count;
        float* points_s = temp_buffer + 2 * point_count;

        cuda::find_points(image_data, batch_count, channel_count, image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
       
        cuda::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>());

        cudaFree(temp_buffer);
    }

    return result;
}

std::tuple<torch::Tensor, FitCircleStatus> estimate_area_learned(torch::Tensor image, int strip_count, torch::jit::Module model, int model_patch_size, ConfidenceThresholds confidence_thresholds)
{
    check_image_tensor(image);
    
    Image image_data = get_image_data(image);

    bool batched = image.dim() == 4;

    int batch_count = batched ? image.size(0) : 1;
    int channel_count = image.size(-3);
    int image_height = image.size(-2);
    int image_width = image.size(-1);
    int point_count = 2 * strip_count;

    torch::TensorOptions options = torch::device(image.device()).dtype(torch::kFloat32);
    auto result = batched ? 
        std::make_tuple(torch::empty({batch_count, 4}, options), FitCircleStatus::invalid) : 
        std::make_tuple(torch::empty({4}, options), FitCircleStatus::invalid);


    torch::Tensor strips = torch::empty({batch_count * strip_count, 5, model_patch_size, image_width}, options);
    std::vector<torch::jit::IValue> model_input = {strips};

    if (image.device().is_cpu())
    {
        float* temp_buffer = (float*)malloc(3 * batch_count * point_count * sizeof(float));
        float* points_x = temp_buffer + 0 * point_count;
        float* points_y = temp_buffer + 1 * point_count;
        float* points_s = temp_buffer + 2 * point_count;
        
        cpu::make_strips(image_data, batch_count, channel_count, image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());

        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cpu::find_points_from_strip_scores(strip_scores.data_ptr<float>(), batch_count, image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
        
        cpu::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>(), &std::get<FitCircleStatus>(result));

        free(temp_buffer);
    }
    else
    {
        float* temp_buffer;
        cudaMalloc((void**)&temp_buffer, 3 * batch_count * point_count * sizeof(float));
        float* points_x = temp_buffer + 0 * point_count;
        float* points_y = temp_buffer + 1 * point_count;
        float* points_s = temp_buffer + 2 * point_count;
        
        cuda::make_strips(image_data, batch_count, channel_count, image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cuda::find_points_from_strip_scores(strip_scores.data_ptr<float>(), batch_count, image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
        
        cuda::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>());

        cudaFree(temp_buffer);
    }

    return result;
}

torch::Tensor get_points_handcrafted(torch::Tensor image, int strip_count, FeatureThresholds feature_thresholds)
{
    check_image_tensor(image);

    Image image_data = get_image_data(image);
    
    bool batched = image.dim() == 4;

    int batch_count = batched ? image.size(0) : 1;
    int channel_count = image.size(-3);
    int image_height = image.size(-2);
    int image_width = image.size(-1);
    int point_count = 2 * strip_count;

    torch::TensorOptions options = torch::device(image.device()).dtype(torch::kFloat32);
    torch::Tensor result = batched ? torch::empty({batch_count, 3, point_count}, options) : torch::empty({3, point_count}, options);

    float* temp_buffer = result.data_ptr<float>();
    float* points_x = temp_buffer + 0 * point_count; 
    float* points_y = temp_buffer + 1 * point_count;
    float* points_s = temp_buffer + 2 * point_count; 

    if (image.device().is_cpu())
    {
        cpu::find_points(image_data, batch_count, channel_count, image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
    }
    else
    {
        cuda::find_points(image_data, batch_count, channel_count, image_height, image_width, strip_count, feature_thresholds, points_x, points_y, points_s);
    }

    return result;
}

torch::Tensor get_points_learned(torch::Tensor image, int strip_count, torch::jit::Module model, int model_patch_size)
{
    check_image_tensor(image);

    Image image_data = get_image_data(image);

    bool batched = image.dim() == 4;

    int batch_count = batched ? image.size(0) : 1;
    int channel_count = image.size(-3);
    int image_height = image.size(-2);
    int image_width = image.size(-1);
    int point_count = 2 * strip_count;

    torch::TensorOptions options = torch::device(image.device()).dtype(torch::kFloat32);
    torch::Tensor result = batched ? torch::empty({batch_count, 3, point_count}, options) : torch::empty({3, point_count}, options);
    
    torch::Tensor strips = torch::empty({batch_count * strip_count, 5, model_patch_size, image_width}, options);
    std::vector<torch::jit::IValue> model_input = {strips};

    float* temp_buffer = result.data_ptr<float>();
    float* points_x = temp_buffer + 0 * point_count; 
    float* points_y = temp_buffer + 1 * point_count;
    float* points_s = temp_buffer + 2 * point_count; 
    
    if (image.device().is_cpu())
    {
        cpu::make_strips(image_data, batch_count, channel_count, image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cpu::find_points_from_strip_scores(strip_scores.data_ptr<float>(), batch_count, image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
    }
    else
    {
        cuda::make_strips(image_data, batch_count, channel_count, image_height, image_width, strip_count, model_patch_size, strips.data_ptr<float>());
        
        torch::Tensor strip_scores = torch::sigmoid(model.forward(model_input).toTensor());

        cuda::find_points_from_strip_scores(strip_scores.data_ptr<float>(), batch_count, image_height, image_width, strip_count, model_patch_size, points_x, points_y, points_s);
    }

    return result;
}

std::tuple<torch::Tensor, FitCircleStatus> fit_area(torch::Tensor points, py::tuple image_size, ConfidenceThresholds confidence_thresholds)
{
    check_points(points);

    bool batched = points.dim() == 3;

    int batch_count = batched ? points.size(0) : 1;
    int image_height = image_size[0].cast<int>();
    int image_width = image_size[1].cast<int>();
    int point_count = points.size(-1);

    torch::TensorOptions options = torch::device(points.device()).dtype(torch::kFloat32);
    auto result = batched ? 
        std::make_tuple(torch::empty({batch_count, 4}, options), FitCircleStatus::invalid) : 
        std::make_tuple(torch::empty({4}, options), FitCircleStatus::invalid);

    float* temp_buffer = points.data_ptr<float>();
    float* points_x = temp_buffer + 0 * point_count; 
    float* points_y = temp_buffer + 1 * point_count;
    float* points_s = temp_buffer + 2 * point_count; 

    if (points.device().is_cpu())
    {
        cpu::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>(), &std::get<FitCircleStatus>(result));
    }
    else
    {
        cuda::fit_circle(points_x, points_y, points_s, batch_count, point_count, confidence_thresholds, image_height, image_width, std::get<torch::Tensor>(result).data_ptr<float>());
    }

    return result;
}
