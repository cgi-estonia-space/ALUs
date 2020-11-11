#include "snap-core/util/math/math_utils.h"

namespace alus {
namespace snapengine {

//float MathUtils::Interpolate2D(float wi, float wj, float x00, float x10, float x01, float x11) {
//    return x00 + wi * (x10 - x00) + wj * (x01 - x00) + wi * wj * (x11 + x00 - x01 - x10);
//}

double MathUtils::Interpolate2D(double wi, double wj, double x00, double x10, double x01, double x11) {
    return x00 + wi * (x10 - x00) + wj * (x01 - x00) + wi * wj * (x11 + x00 - x01 - x10);
}
int MathUtils::FloorAndCrop(double x, int min, int max) {
        int rx = FloorInt(x);
        return Crop(rx, min, max);
}

}  // namespace snapengine
}  // namespace alus