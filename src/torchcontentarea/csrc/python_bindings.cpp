#include <torch/extension.h>
#include "implementation.hpp"

template<>
struct py::detail::type_caster<FeatureThresholds> 
{
    PYBIND11_TYPE_CASTER(FeatureThresholds, py::detail::_("FeatureThresholds"));

    bool load(handle src, bool) 
    {
        if (!src | src.is_none() | !py::isinstance<py::tuple>(src)) 
            return false;

        py::tuple args = reinterpret_borrow<tuple>(src);
        if (len(args) != 3)
            return false;

        value.edge = args[0].cast<float>();
        value.angle = args[1].cast<float>();
        value.intensity = args[2].cast<float>();
        return true;
    }
};

template<>
struct py::detail::type_caster<ConfidenceThresholds> 
{
    PYBIND11_TYPE_CASTER(ConfidenceThresholds, py::detail::_("ConfidenceThresholds"));

    bool load(handle src, bool convert) 
    {
        if (!src | src.is_none() | !py::isinstance<py::tuple>(src)) 
            return false;

        py::tuple args = reinterpret_borrow<tuple>(src);
        if (len(args) != 2)
            return false;

        value.edge = args[0].cast<float>();
        value.circle = args[1].cast<float>();
        return true;
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    m.def("estimate_area_handcrafted", &estimate_area_handcrafted);
    m.def("estimate_area_learned", &estimate_area_learned);
    m.def("get_points_handcrafted", &get_points_handcrafted);
    m.def("get_points_learned", &get_points_learned);
    m.def("fit_circle", &fit_circle);
}
