typedef unsigned int uint;
typedef unsigned char uint8;

struct Area
{
    enum{None, Rectangle, Circle} type;

    union
    {
        struct
        {
            //Rectangle Area
            uint y, x, w, h;
        } rectangle;
        struct
        {
            //Circle Area
            uint y, x, r;
        } circle;
    };
};

Area infer_area_cuda(uint8* image, const uint image_height, const uint image_width, const uint point_count);
void draw_area_cuda(Area area, uint8* mask,  const uint mask_height, const uint mask_width);
