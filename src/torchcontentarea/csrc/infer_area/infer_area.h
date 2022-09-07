#pragma once
#include "../torch_functor.h"

#define MAX_POINT_COUNT 32
#define DISCARD_BORDER 3
#define DEG2RAD 0.01745329251f
#define RAD2DEG (1.0f / DEG2RAD)
#define MAX_CENTER_DIST 0.2 // * image width
#define MIN_RADIUS 0.2 // * image width
#define MAX_RADIUS 0.8 // * image width
#define RANSAC_ATTEMPTS 32
#define RANSAC_ITERATIONS 3
#define RANSAC_INLIER_THRESHOLD 3

typedef unsigned int uint;
typedef unsigned char uint8;

void check_image_tensor(torch::Tensor image);

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
