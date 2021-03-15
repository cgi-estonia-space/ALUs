#pragma once

namespace alus {
namespace snapengine {
namespace custom {
struct Dimension {
    Dimension(int width, int height) : width(width), height(height) {}
    int width;
    int height;
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus
