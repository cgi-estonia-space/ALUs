/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.subset.AbstractSubsetRegion.java
 * ported for native code.
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
#include "snap-core/subset/abstract_subset_region.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {

AbstractSubsetRegion::AbstractSubsetRegion(int border_pixels) {
    if (border_pixels < 0) {
        throw std::invalid_argument("The border pixels " + std::to_string(border_pixels) + " is negative.");
    }
    border_pixels_ = border_pixels;
}

void AbstractSubsetRegion::ValidateDefaultSize(int default_product_width, int default_product_height,
                                               std::string_view exception_message_prefix) {
    if (default_product_width < 1) {
        throw std::invalid_argument(std::string(exception_message_prefix) + " width " +
                                    std::to_string(default_product_width) + " must be greater or equal than 1.");
    }
    if (default_product_height < 1) {
        throw std::invalid_argument(std::string(exception_message_prefix) + " height " +
                                    std::to_string(default_product_height) + " must be greater or equal than 1.");
    }
}

}  // namespace snapengine
}  // namespace alus
