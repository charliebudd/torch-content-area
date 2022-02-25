#pragma once

#include "profiling.h"

typedef unsigned int uint;
typedef unsigned char uint8;


struct Area
{
    enum{None, Rectangle, Circle} type;

    union
    {
        struct { uint y, x, w, h; } rectangle;
        struct { uint y, x, r; } circle;
    };
};

class ContentAreaInference
{
public:
    ContentAreaInference();
    ~ContentAreaInference();

    Area infer_area(uint8* image, const uint image_height, const uint image_width);
    void draw_area(Area area, uint8* mask, const uint mask_height, const uint mask_width);
    std::vector<std::vector<int>> get_points(uint8* image, const uint image_height, const uint image_width);

private:
    uint m_point_count, m_height_samples;
    uint *m_hst_block, *m_hst_points, *m_hst_scores;
    uint *m_dev_block, *m_dev_points, *m_dev_scores;
};
