#include <torch/extension.h>
#include "content_area_inference.cuh"

#ifdef PROFILE
#include <cuda_runtime.h>
#endif

ContentArea infer_area_wrapper(ContentAreaInference &self, torch::Tensor image) 
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    ContentArea area = self.infer_area(image.data_ptr<uint8>(), height, width);

    return area;
}

torch::Tensor draw_area_wrapper(ContentAreaInference &self, torch::Tensor image, ContentArea area) 
{
    uint height = image.size(1);
    uint width = image.size(2);

    torch::Tensor mask = torch::empty_like(image[0]);
    self.draw_area(area, mask.data_ptr<uint8>(), height, width);

    return mask;
}

torch::Tensor infer_mask_wrapper(ContentAreaInference &self, torch::Tensor image)
{
    #ifdef PROFILE
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    #endif

    #ifdef PROFILE
    cudaEventRecord(a);
    #endif

    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    torch::Tensor mask = torch::empty_like(image[0]);
    ContentArea area = self.infer_area(image.data_ptr<uint8>(), height, width);
    self.draw_area(area, mask.data_ptr<uint8>(), height, width);

    #ifdef PROFILE
    cudaEventRecord(b);
    #endif

    #ifdef PROFILE
    cudaEventSynchronize(a);
    cudaEventSynchronize(b);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, a, b);
    ADD_SAMPLE("infer mask wrapper", milliseconds);
    #endif

    return mask;
}

std::vector<std::vector<int>> get_points_wrapper(ContentAreaInference &self, torch::Tensor image)
{
    image = image.contiguous();

    uint height = image.size(1);
    uint width = image.size(2);

    return self.get_points(image.data_ptr<uint8>(), height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    pybind11::class_<ContentArea>(m, "ContentArea")
        .def(py::init<>())
        .def(py::init<uint, uint, uint>())
        .def_property("_ContentArea__type", [](const ContentArea &a) { return static_cast<int>(a.type); }, [](ContentArea &a, int v) { a.type = static_cast<ContentAreaType>(v); })
        .def_readwrite("_ContentArea__x", &ContentArea::x)
        .def_readwrite("_ContentArea__y", &ContentArea::y)
        .def_readwrite("_ContentArea__r", &ContentArea::r);

    pybind11::class_<ContentAreaInference>(m, "ContentAreaInference")
        .def(py::init())
        .def("_ContentAreaInference__infer_area", &infer_area_wrapper)
        .def("_ContentAreaInference__draw_mask", &draw_area_wrapper)
        .def("_ContentAreaInference__infer_mask", &infer_mask_wrapper)
        .def("_ContentAreaInference__get_points", &get_points_wrapper);

    #ifdef PROFILE
    m.def("get_times",
        []() {
            return GET_TIMES();
        }
    );
    #endif
}
