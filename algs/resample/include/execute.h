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

#include <optional>
#include <string>
#include <vector>

#include "cuda_device_init.h"
#include "raster_properties.h"
#include "resample_method.h"

namespace alus::resample {

class Execute final {
public:
    struct Parameters {
        std::vector<std::string> inputs;
        std::string output_path;
        std::optional<size_t> resample_dimension_band;
        RasterDimension resample_dimension;
        RasterDimension tile_dimension;
        size_t pixel_overlap;
        Method resample_method;
        std::optional<std::string> output_format;
        std::optional<std::string> crs;
        std::vector<size_t> excluded_bands;
    };

    Execute() = delete;
    explicit Execute(Parameters params);

    Execute(const Execute& other) = delete;
    Execute& operator=(const Execute& other) = delete;

    void Run(alus::cuda::CudaInit& cuda_init, size_t gpu_mem_percentage);

    ~Execute();

private:
    Parameters params_;
};
}  // namespace alus::resample