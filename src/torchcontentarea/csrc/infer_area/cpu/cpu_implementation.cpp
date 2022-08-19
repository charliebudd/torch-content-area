#include "../infer_area.h"

torch::Tensor InferAreaHandcrafted::cpu_implementation(torch::Tensor image, uint strip_count, FeatureThresholds feature_thresholds, ConfidenceThresholds confidence_thresholds)
{
    throw std::logic_error("CPU implementation currently unavailable!");
}


torch::Tensor InferAreaLearned::cpu_implementation(torch::Tensor image, uint strip_count, torch::jit::Module model, uint model_patch_size, ConfidenceThresholds confidence_thresholds)
{
    throw std::logic_error("CPU implementation currently unavailable!");
}
