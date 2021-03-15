#pragma once

#include <memory>

namespace alus {
namespace snapengine {
namespace custom {

struct Rectangle {
    Rectangle() = default;
    Rectangle(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}
    explicit Rectangle(const std::shared_ptr<Rectangle>& rectangle) {
        x = rectangle->x;
        y = rectangle->y;
        width = rectangle->width;
        height = rectangle->height;
    }
    int x, y, width, height;
};

}  // namespace custom
}  // namespace snapengine
}  // namespace alus
