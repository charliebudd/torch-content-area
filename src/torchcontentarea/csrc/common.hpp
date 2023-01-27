#pragma once
#include <type_traits>

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

enum ImageFormat 
{
    rgb_float,
    rgb_double,
    rgb_uint8,
    rgb_int,
    rgb_long,
    gray_float,
    gray_double,
    gray_uint8,  
    gray_int,
    gray_long,
};

enum FitCircleStatus
{
    success,
    no_points,
    invalid,
};

struct Image
{
    Image(ImageFormat format, const void* data) : format(format), data(data) {}

    ImageFormat format;
    const void* data;
};

#define ARG(...) __VA_ARGS__
#define KERNEL_DISPATCH_IMAGE_FORMAT(FUNCTION, DISPATCH_ARGS, IMAGE, ...)                                               \
    switch(IMAGE.format)                                                                                                \
    {                                                                                                                   \
        case(rgb_float):   FUNCTION<3, float   ><<<DISPATCH_ARGS>>>((const float*   )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_double):  FUNCTION<3, double  ><<<DISPATCH_ARGS>>>((const double*  )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_uint8):   FUNCTION<3, uint8   ><<<DISPATCH_ARGS>>>((const uint8*   )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_int):     FUNCTION<3, int     ><<<DISPATCH_ARGS>>>((const int*     )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_long):    FUNCTION<3, long int><<<DISPATCH_ARGS>>>((const long int*)IMAGE.data, __VA_ARGS__); break;   \
        case(gray_float):  FUNCTION<1, float   ><<<DISPATCH_ARGS>>>((const float*   )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_double): FUNCTION<1, double  ><<<DISPATCH_ARGS>>>((const double*  )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_uint8):  FUNCTION<1, uint8   ><<<DISPATCH_ARGS>>>((const uint8*   )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_int):    FUNCTION<1, int     ><<<DISPATCH_ARGS>>>((const int*     )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_long):   FUNCTION<1, long int><<<DISPATCH_ARGS>>>((const long int*)IMAGE.data, __VA_ARGS__); break;   \
    }

#define FUNCTION_CALL_IMAGE_FORMAT(FUNCTION, IMAGE, ...)                                             \
    switch(IMAGE.format)                                                                             \
    {                                                                                                \
        case(rgb_float):   FUNCTION<3, float   >((const float*   )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_double):  FUNCTION<3, double  >((const double*  )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_uint8):   FUNCTION<3, uint8   >((const uint8*   )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_int):     FUNCTION<3, int     >((const int*     )IMAGE.data, __VA_ARGS__); break;   \
        case(rgb_long):    FUNCTION<3, long int>((const long int*)IMAGE.data, __VA_ARGS__); break;   \
        case(gray_float):  FUNCTION<1, float   >((const float*   )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_double): FUNCTION<1, double  >((const double*  )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_uint8):  FUNCTION<1, uint8   >((const uint8*   )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_int):    FUNCTION<1, int     >((const int*     )IMAGE.data, __VA_ARGS__); break;   \
        case(gray_long):   FUNCTION<1, long int>((const long int*)IMAGE.data, __VA_ARGS__); break;   \
    }
