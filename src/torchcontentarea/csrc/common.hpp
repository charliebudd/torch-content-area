#pragma once
#include <torch/extension.h>

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

typedef unsigned char uint8;

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
