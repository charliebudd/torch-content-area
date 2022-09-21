#pragma once

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

enum ImageFormat {
    rgb_float,
    rgb_double,
    rgb_uint8,
    rgb_int,
    gray_float,
    gray_double,
    gray_uint8,  
    gray_int,
};

typedef unsigned char uint8;

#define DISPATCH_IMAGE_FORMAT(FORMAT, FUNCTION, IMAGE, ...)                                     \
    switch(FORMAT)                                                                              \
    {                                                                                           \
        case(rgb_float):   FUNCTION<float,  3>(IMAGE.data_ptr<float >(), __VA_ARGS__); break;   \
        case(rgb_double):  FUNCTION<double, 3>(IMAGE.data_ptr<double>(), __VA_ARGS__); break;   \
        case(rgb_uint8):   FUNCTION<uint8,  3>(IMAGE.data_ptr<uint8 >(), __VA_ARGS__); break;   \
        case(rgb_int):     FUNCTION<int,    3>(IMAGE.data_ptr<int   >(), __VA_ARGS__); break;   \
        case(gray_float):  FUNCTION<float,  1>(IMAGE.data_ptr<float >(), __VA_ARGS__); break;   \
        case(gray_double): FUNCTION<double, 1>(IMAGE.data_ptr<double>(), __VA_ARGS__); break;   \
        case(gray_uint8):  FUNCTION<uint8,  1>(IMAGE.data_ptr<uint8 >(), __VA_ARGS__); break;   \
        case(gray_int):    FUNCTION<int,    1>(IMAGE.data_ptr<int   >(), __VA_ARGS__); break;   \
    }

#define INSTANTIATE_IMAGE_FORMAT(FUNCTION, ...)  \
    template void FUNCTION<float,  3>(const float*  image, __VA_ARGS__);     \
    template void FUNCTION<double, 3>(const double* image, __VA_ARGS__);     \
    template void FUNCTION<uint8,  3>(const uint8*  image, __VA_ARGS__);     \
    template void FUNCTION<int,    3>(const int*    image, __VA_ARGS__);     \
    template void FUNCTION<float,  1>(const float*  image, __VA_ARGS__);     \
    template void FUNCTION<double, 1>(const double* image, __VA_ARGS__);     \
    template void FUNCTION<uint8,  1>(const uint8*  image, __VA_ARGS__);     \
    template void FUNCTION<int,    1>(const int*    image, __VA_ARGS__);     
    