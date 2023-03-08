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

#include <cstddef>

#include "raster_properties.h"

namespace alus::rowresample {

inline double GetRatio(size_t in_line_len, size_t out_line_len) {
    return static_cast<double>(in_line_len) / static_cast<double>(out_line_len);
}

void FillLineFrom(float* in_line, size_t in_size, float* out_line, size_t out_size);

void ProcessAndTransferHost(const float* input, RasterDimension input_dimensions, float* output,
                            RasterDimension output_dimension);

void Process(const float* input, RasterDimension input_dimension, float* output, RasterDimension output_dimension);

}  // namespace alus::rowresample