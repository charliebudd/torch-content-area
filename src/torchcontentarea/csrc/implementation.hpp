#pragma once
#include <tuple>
#include <torch/extension.h>
#include "common.hpp"

std::tuple<torch::Tensor, FitCircleStatus> estimate_area_handcrafted(torch::Tensor image, int strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds);
std::tuple<torch::Tensor, FitCircleStatus> estimate_area_learned(torch::Tensor image, int strip_count, torch::jit::Module model, int model_patch_size, ConfidenceThresholds confidence_thresholds);

torch::Tensor get_points_handcrafted(torch::Tensor points, int strip_count, FeatureThresholds feature_thresholds);
torch::Tensor get_points_learned(torch::Tensor points, int strip_count, torch::jit::Module model, int model_patch_size);

std::tuple<torch::Tensor, FitCircleStatus> fit_area(torch::Tensor points, py::tuple image_size, ConfidenceThresholds confidence_thresholds);
