#pragma once
#include "../torch_functor.h"

struct FeatureThresholds
{
    float edge;
    float angle;
    float intensity;
};

struct ConfidenceThresholds
{
    float edge;
    float circle;
};

class InferAreaHandcrafted : public TorchFunctor<InferAreaHandcrafted, torch::Tensor, torch::Tensor, uint, FeatureThresholds, ConfidenceThresholds>
{
public:
    static torch::Tensor cpu_implementation(torch::Tensor image, uint strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds);
    static torch::Tensor cuda_implementation(torch::Tensor image, uint strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds);
};

class InferAreaLearned : public TorchFunctor<InferAreaLearned, torch::Tensor, torch::Tensor, uint, torch::jit::Module, uint, ConfidenceThresholds>
{
public:
    static torch::Tensor cpu_implementation(torch::Tensor image,  uint strip_count, torch::jit::Module model, uint model_patch_size, ConfidenceThresholds confidence_thresholds);
    static torch::Tensor cuda_implementation(torch::Tensor image, uint strip_count, torch::jit::Module model, uint model_patch_size, ConfidenceThresholds confidence_thresholds);
};
