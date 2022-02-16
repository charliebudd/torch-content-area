#include <torch/extension.h>

#include "torch_content_area.h"

py::tuple circle_to_tuple(Area area)
{
    return py::make_tuple(
        py::cast(area.circle.x),
        py::cast(area.circle.y),
        py::cast(area.circle.r)
    );
};

py::tuple rectangle_to_tuple(Area area)
{
    return py::make_tuple(
        py::cast(area.rectangle.x),
        py::cast(area.rectangle.y),
        py::cast(area.rectangle.w),
        py::cast(area.rectangle.h)
    );
};

py::tuple area_to_tuple(Area area)
{
    switch(area.type)
    {
        case(Area::Circle): return py::make_tuple("Circle", circle_to_tuple(area)); break;
        case(Area::Rectangle): return py::make_tuple("Rectangle", rectangle_to_tuple(area)); break;
        case(Area::None): return py::make_tuple("None"); break;
        default: return py::none(); break;
    }
}

py::tuple get_area(torch::Tensor image) 
{
    uint height = image.size(1);
    uint width = image.size(2);

    Area area = infer_area_cuda(image.data_ptr<uint8>(), height, width, 8);

    return area_to_tuple(area);
}

void get_area_mask(torch::Tensor image, torch::Tensor mask) 
{
    uint height = image.size(1);
    uint width = image.size(2);

    Area area = infer_area_cuda(image.data_ptr<uint8>(), height, width, 8);
    draw_area_cuda(area, mask.data_ptr<uint8>(), height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("get_area", &get_area, "Infers the conent area for a given endoscopic image");
    m.def("get_area_mask", &get_area_mask, "Infers the conent area for a given endoscopic image and returns a binary mask");
}
