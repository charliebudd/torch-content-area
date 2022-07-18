#include <torch/extension.h>
#include "content_area_inference.cuh"

ContentArea area_from_pyobject(py::object object)
{
    if (object == py::none())
    {
        return ContentArea();
    }
    else
    {
        std::vector<uint> values = object.cast<std::vector<uint>>();
        return ContentArea(values[0], values[1], values[2]);
    }
}

py::object pyobject_from_area(ContentArea area)
{
    if (area.type == ContentAreaType::None)
    {
        return py::none();
    }
    else
    {
        return py::make_tuple(area.x, area.y, area.r);
    }
}

py::object infer_area_wrapper(ContentAreaInference &self, torch::Tensor image) 
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    ContentArea area = self.infer_area(image.data_ptr<uint8>(), height, width);

    return pyobject_from_area(area);
}


py::object infer_area_hybrid_wrapper(ContentAreaInference &self, torch::Tensor image, torch::Tensor edges) 
{
    edges = edges.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);
    uint strip_count = edges.size(0);

    ContentArea area = self.infer_area_hybrid(edges.data_ptr<float>(), height, width, strip_count);

    return pyobject_from_area(area);
}

torch::Tensor draw_area_wrapper(ContentAreaInference &self, torch::Tensor image, py::object area_tuple) 
{
    uint height = image.size(1);
    uint width = image.size(2);

    ContentArea area = area_from_pyobject(area_tuple);

    torch::Tensor mask = torch::empty_like(image[0]);

    self.draw_area(area, mask.data_ptr<uint8>(), height, width);

    return mask;
}

torch::Tensor crop_area_wrapper(ContentAreaInference &self, torch::Tensor src_image, py::object area_tuple, std::vector<int> size, const int interpolation_mode)
{
    uint src_height = src_image.size(1);
    uint src_width = src_image.size(2);

    uint dst_height = size[0];
    uint dst_width = size[1];

    ContentArea area = area_from_pyobject(area_tuple);

    torch::Tensor dst_image = torch::empty({3, dst_height, dst_width}, src_image.options());

    src_image = src_image.contiguous();
    dst_image = dst_image.contiguous();

    self.crop_area(area, src_image.data_ptr<uint8>(), dst_image.data_ptr<uint8>(), src_width, src_height, dst_width, dst_height, (InterpolationMode)interpolation_mode);

    return dst_image;
}

torch::Tensor infer_mask_wrapper(ContentAreaInference &self, torch::Tensor image)
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    torch::Tensor mask = torch::empty_like(image[0]);

    ContentArea area = self.infer_area(image.data_ptr<uint8>(), height, width);

    self.draw_area(area, mask.data_ptr<uint8>(), height, width);

    return mask;
}

std::vector<std::vector<float>> get_debug_wrapper(ContentAreaInference &self)
{
    return self.get_debug();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    pybind11::class_<ContentAreaInference>(m, "ContentAreaInference")
        .def(py::init())
        .def("_ContentAreaInference__infer_area", &infer_area_wrapper)
        .def("_ContentAreaInference__infer_area_hybrid", &infer_area_hybrid_wrapper)
        .def("_ContentAreaInference__draw_mask", &draw_area_wrapper)
        .def("_ContentAreaInference__crop_area", &crop_area_wrapper)
        .def("_ContentAreaInference__infer_mask", &infer_mask_wrapper)
        .def("_ContentAreaInference__get_debug", &get_debug_wrapper);
}
