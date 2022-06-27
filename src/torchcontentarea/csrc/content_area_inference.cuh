#pragma once

#include "profiling.h"

typedef unsigned int uint;
typedef unsigned char uint8;

enum ContentAreaType{None=0, Circle=1};

enum InterpolationMode{Nearest=0, Bilinear=1};

struct ContentArea
{   
    ContentArea(): type(ContentAreaType::None) {};
    ContentArea(uint x, uint y, uint r): type(ContentAreaType::Circle), x(x), y(y), r(r) {};
    
    ContentAreaType type;
    uint x, y, r;
};

class ContentAreaInference
{
public:
    ContentAreaInference();
    ~ContentAreaInference();

    ContentArea infer_area(uint8* image, const uint image_height, const uint image_width);
    void draw_area(const ContentArea area, uint8* mask, const uint mask_height, const uint mask_width);
    void crop_area(const ContentArea area, const uint8* src_image, uint8* dst_image, const uint src_width, const uint src_height, const uint dst_width, const uint dst_height, const InterpolationMode interpolation_mode);

    std::vector<std::vector<float>> get_debug(uint8* image, const uint image_height, const uint image_width);

private:
    uint m_height_samples;
    uint m_point_count;
    uint m_buffer_size;

    void  *m_hst_buffer;
    uint  *m_hst_edge_x;
    uint  *m_hst_edge_y;
    float *m_hst_edge_scores;
    float *m_hst_circle;
    
    void  *m_dev_buffer;
    uint  *m_dev_edge_x;
    uint  *m_dev_edge_y;
    float *m_dev_edge_scores;
    float *m_dev_circle;
};
