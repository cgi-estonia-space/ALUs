/**
 * This file is a duplicate of a SNAP's Constant.java
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
#pragma once

#include <cstddef>
#include <string>

namespace alus {
namespace utils {
namespace constants {
constexpr int INVALID_INDEX{-1};

constexpr float THERMAL_NOISE_TRG_FLOOR_VALUE{0.01234567890000F};

// GDAL driver constants
constexpr char GDAL_MEM_DRIVER[]{"MEM"};
constexpr char GDAL_GTIFF_DRIVER[]{"GTiff"};
constexpr char GDAL_GTIFF_FILE_EXTENSION[]{".tif"};
constexpr int GDAL_DEFAULT_RASTER_BAND{1};
}  // namespace constants
}  // namespace utils
}  // namespace alus