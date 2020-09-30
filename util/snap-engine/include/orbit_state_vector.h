/**
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

#pragma once

#include "product_data.h"

namespace alus {
namespace snapengine {

struct OrbitStateVector {
    alus::snapengine::old::Utc time_{};
    double timeMjd_{};
    double xPos_{};
    double yPos_{};
    double zPos_{};
    double xVel_{};
    double yVel_{};
    double zVel_{};

    OrbitStateVector() = default;
    OrbitStateVector(
        alus::snapengine::old::Utc time, double xPos, double yPos, double zPos, double xVel, double yVel, double zVel);
};

}  // namespace snapengine
}  // namespace alus