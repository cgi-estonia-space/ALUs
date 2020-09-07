#pragma once

#include <climits>

#include "shapes.h"

namespace alus {
namespace shapeutils {

inline __device__ __host__ Rectangle GetIntersection(const Rectangle &rectangle_1, const Rectangle rectangle_2) {
    int target_x = rectangle_1.x;
    int target_y = rectangle_1.y;
    long target_width = target_x + rectangle_1.width;
    long target_height = target_y + rectangle_1.height;
    long rx2 = rectangle_2.x + rectangle_2.width;
    long ry2 = rectangle_2.y + rectangle_2.height;
    if (target_x < rectangle_2.x) {
        target_x = rectangle_2.x;
    }
    if (target_y < rectangle_2.y) {
        target_y = rectangle_2.y;
    }
    if (target_width > rx2) {
        target_width = rx2;
    }
    if (target_height > ry2) {
        target_height = ry2;
    }
    target_width -= target_x;
    target_height -= target_y;
    if (target_width < INT_MIN) {
        target_width = INT_MIN;
    }
    if (target_height < INT_MIN) {
        target_height = INT_MIN;
    }
    return {target_x, target_y, static_cast<int>(target_width), static_cast<int>(target_height)};
}
}  // namespace shapeutils
}  // namespace alus