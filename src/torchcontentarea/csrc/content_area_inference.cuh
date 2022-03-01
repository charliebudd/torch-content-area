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
    std::vector<std::vector<int>> get_points(uint8* image, const uint image_height, const uint image_width);

private:
    uint m_point_count, m_height_samples;
    uint *m_hst_block, *m_hst_points, *m_hst_scores;
    uint *m_dev_block, *m_dev_points, *m_dev_scores;
};
