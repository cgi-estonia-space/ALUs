/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.MathUtils.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "snap-core/core/util/math/math_utils.h"

#include <cmath>
#include <cstddef>  //for std::size_t, currently my clion not agree, but might be needed
#include <limits>
#include <vector>

#include "custom/dimension.h"
#include "custom/rectangle.h"

namespace alus {
namespace snapengine {

double MathUtils::LOG10 = log(10.0);
// float MathUtils::Interpolate2D(float wi, float wj, float x00, float x10, float x01, float x11) {
//    return x00 + wi * (x10 - x00) + wj * (x01 - x00) + wi * wj * (x11 + x00 - x01 - x10);
//}

double MathUtils::Interpolate2D(double wi, double wj, double x00, double x10, double x01, double x11) {
    return x00 + wi * (x10 - x00) + wj * (x01 - x00) + wi * wj * (x11 + x00 - x01 - x10);
}
int MathUtils::FloorAndCrop(double x, int min, int max) {
    int rx = FloorInt(x);
    return Crop(rx, min, max);
}

std::shared_ptr<custom::Dimension> MathUtils::FitDimension(int n, double a, double b) {
    if (n == 0) {
        return std::make_shared<custom::Dimension>(0, 0);
    }
    double wd = sqrt(n * a / b);
    double hd = n / wd;
    int w1 = static_cast<int>(floor(wd));
    int h1 = static_cast<int>(floor(hd));
    int w2;
    int h2;
    if (w1 > 0) {
        w2 = w1 + 1;
    } else {
        w2 = w1 = 1;
    }
    if (h1 > 0) {
        h2 = h1 + 1;
    } else {
        h2 = h1 = 1;
    }
    std::vector<double> d(4);
    d.at(0) = std::abs(b * w1 - a * h1);
    d.at(1) = std::abs(b * w1 - a * h2);
    d.at(2) = std::abs(b * w2 - a * h1);
    d.at(3) = std::abs(b * w2 - a * h2);
    int index = -1;
    double d_min = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < d.size(); i++) {
        if (d.at(i) < d_min) {
            d_min = d.at(i);
            index = i;
        }
    }
    if (index == 0) {
        return std::make_shared<custom::Dimension>(w1, h1);
    }
    if (index == 1) {
        return std::make_shared<custom::Dimension>(w1, h2);
    }
    if (index == 2) {
        return std::make_shared<custom::Dimension>(w2, h1);
    }
    return std::make_shared<custom::Dimension>(w2, h2);
}

std::vector<std::shared_ptr<custom::Rectangle>> MathUtils::SubdivideRectangle(int width, int height, int num_tiles_x,
                                                                              int num_tiles_y, int extra_border) {
    std::vector<std::shared_ptr<custom::Rectangle>> rectangles(num_tiles_x * num_tiles_y);
    int k = 0;
    float w = static_cast<float>(width) / num_tiles_x;
    float h = static_cast<float>(height) / num_tiles_y;
    for (int j = 0; j < num_tiles_y; j++) {
        int y1 = static_cast<int>(std::floor((j + 0) * h));
        int y2 = static_cast<int>(std::floor((j + 1) * h)) - 1;
        if (y2 < y1) {
            y2 = y1;
        }
        y1 -= extra_border;
        y2 += extra_border;
        if (y1 < 0) {
            y1 = 0;
        }
        if (y2 > height - 1) {
            y2 = height - 1;
        }
        for (int i = 0; i < num_tiles_x; i++) {
            int x1 = static_cast<int>(std::floor((i + 0) * w));
            int x2 = static_cast<int>(std::floor((i + 1) * w)) - 1;
            if (x2 < x1) {
                x2 = x1;
            }
            x1 -= extra_border;
            x2 += extra_border;
            if (x1 < 0) {
                x1 = 0;
            }
            if (x2 > width - 1) {
                x2 = width - 1;
            }

            rectangles.at(k) = std::make_shared<custom::Rectangle>(x1, y1, (x2 - x1) + 1, (y2 - y1) + 1);
            k++;
        }
    }
    return rectangles;
}

bool MathUtils::EqualValues(double x1, double x2, double eps) {
    return std::abs(x1 - x2) <= eps;
    ;
}

}  // namespace snapengine
}  // namespace alus