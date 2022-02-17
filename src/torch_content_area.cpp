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

Area tuple_to_area(py::tuple tuple)
{
    std::string area_type = tuple[0].cast<std::string>();
    py::tuple area_info = tuple[1];

    if (area_type == "Circle")
    {
        Area area;
        area.type = Area::Circle;
        area.circle.x = area_info[0].cast<uint>();
        area.circle.y = area_info[1].cast<uint>();
        area.circle.r = area_info[2].cast<uint>();
        return area;
    }
    else if (area_type == "Rectangle")
    {
        Area area;
        area.type = Area::Rectangle;
        area.rectangle.x = area_info[0].cast<uint>();
        area.rectangle.y = area_info[1].cast<uint>();
        area.rectangle.w = area_info[2].cast<uint>();
        area.rectangle.h = area_info[3].cast<uint>();
        return area;
    }
    else
    {
        Area area;
        area.type = Area::None;
        return area;
    }
}

py::tuple infer_area(torch::Tensor image) 
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    Area area = infer_area_cuda(image.data_ptr<uint8>(), height, width, 8);

    return area_to_tuple(area);
}

torch::Tensor draw_mask(torch::Tensor image, py::tuple area_tuple) 
{
    uint height = image.size(1);
    uint width = image.size(2);

    Area area = tuple_to_area(area_tuple);

    torch::Tensor mask = torch::empty_like(image[0]);
    draw_area_cuda(area, mask.data_ptr<uint8>(), height, width);

    return mask;
}


torch::Tensor infer_mask(torch::Tensor image)
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    torch::Tensor mask = torch::empty_like(image[0]);
    Area area = infer_area_cuda(image.data_ptr<uint8>(), height, width, 8);
    draw_area_cuda(area, mask.data_ptr<uint8>(), height, width);

    return mask;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("infer_area", &infer_area, "Infers the conent area for a given endoscopic image");
    m.def("draw_mask", &draw_mask, "Draws the provided content area");
    m.def("infer_mask", &infer_mask, "Infers the conent area for a given endoscopic image and returns a binary mask");
}
