#include "infer_area.h"

#define NDIM_ERROR_MSG(d) std::string("Expected an image tensor with 4 dimensions but found . Is you Image in NCHW format?").insert(53, std::to_string(d))
#define CHANNEL_ERROR_MSG(c) std::string("Expected a grayscale or RGB image but found size  at position 1. Is you Image in NCHW format?").insert(49, std::to_string(c))

void check_image_tensor(torch::Tensor image)
{
    if (image.ndimension() != 4)
    {
        throw std::runtime_error(NDIM_ERROR_MSG(image.ndimension()));
    }

    if (image.size(1) != 1 && image.size(1) != 3)
    {
        throw std::runtime_error(CHANNEL_ERROR_MSG(image.size(1)));
    }
}
