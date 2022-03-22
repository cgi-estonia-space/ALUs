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

#include <future>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cuda_device_init.h"
#include "raster_properties.h"
#include "resample_method.h"
#include "sentinel2_dataset.h"

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
    void WaitForCudaInit(const alus::cuda::CudaInit& cuda_init);
    [[nodiscard]] bool CanDeviceFit(size_t bytes, size_t percentage_available) const;
    void TryCertifyResampleDimensions(RasterDimension dim);
    void ReviseOutputFactories();
    // band_index starting from 1.
    [[nodiscard]] bool DoesRequireResampling(size_t band_index, RasterDimension band_dim) const;
    [[nodiscard]] bool IsBandInExcludeList(size_t band_index) const;
    [[nodiscard]] bool DoesBandNeedResampling(alus::RasterDimension band_dim) const;
    // per Domain -> [key, value]
    void AddMetadata(std::vector<std::pair<std::string, std::pair<std::string, std::string>>>& from_ds) const;

    Parameters params_;
    bool cuda_init_done_{false};
    const cuda::CudaDevice* gpu_device_;
    bool resample_dim_certified_{false};
    RasterDimension resample_dim_{};
    std::list<std::future<void>> output_factories_{};
};
}  // namespace alus::resample