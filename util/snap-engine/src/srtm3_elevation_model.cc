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
#include "srtm3_elevation_model.h"

#include "cuda_util.hpp"
#include "earth_gravitational_model96_computation.h"
#include "srtm3_elevation_model_constants.h"
#include "geo_utils.h"

namespace alus {
namespace snapengine {

Srtm3ElevationModel::Srtm3ElevationModel(std::vector<std::string> file_names) : file_names_(std::move(file_names)) {
    device_srtm3_tiles_count_ = 0;
}

Srtm3ElevationModel::~Srtm3ElevationModel() { this->DeviceFree(); }

// use a preconfigured emg96 instance with the tiles already loaded onto the gpu.
void Srtm3ElevationModel::ReadSrtmTiles(EarthGravitationalModel96* egm_96) {
    for (auto&& dem_file : file_names_) {
        // TODO: Priority needed for keeping results as close as possible to SNAP.
        auto& ds =
            srtms_.emplace_back(dem_file, GeoTransformSourcePriority::WORLDFILE_PAM_INTERNAL_TABFILE_NONE);
        ds.LoadRasterBand(1);
        const auto* geo_transform = ds.GetTransform();

        Srtm3FormatComputation srtm_data{};
        srtm_data.m00 = static_cast<float>(geo_transform[transform::TRANSFORM_PIXEL_X_SIZE_INDEX]);
        srtm_data.m10 = static_cast<float>(geo_transform[transform::TRANSFORM_ROTATION_1]);
        srtm_data.m01 = static_cast<float>(geo_transform[transform::TRANSFORM_ROTATION_2]);
        srtm_data.m11 = static_cast<float>(geo_transform[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX]);
        // TODO: Rounding in order to keep end results as close as possible to SNAP.
        srtm_data.m02 = std::round(static_cast<float>(geo_transform[transform::TRANSFORM_LON_ORIGIN_INDEX]));
        // TODO: Rounding in order to keep end results as close as possible to SNAP.
        srtm_data.m12 = std::round(static_cast<float>(geo_transform[transform::TRANSFORM_LAT_ORIGIN_INDEX]));

        srtm_data.no_data_value = srtm3elevationmodel::NO_DATA_VALUE;
        srtm_data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
        srtm_data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
        srtm_data.egm = const_cast<float*>(egm_96->GetDeviceValues());

        this->srtm_format_info_.push_back(srtm_data);
    }
}

// keep it for accuracy tests. Not sure if first 4 values(by column) are calcuated or read.
// not sure if the last 2 values get their decimals ripped of or not.
// once backgeocoding is completed and SRTM3 tile processing is automated and without placeholders, remove this.
std::vector<Srtm3FormatComputation> Srtm3ElevationModel::SrtmPlaceholderData(EarthGravitationalModel96* egm_96) {
    std::vector<Srtm3FormatComputation> result;

    Srtm3FormatComputation srtm41_01Data;
    srtm41_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m01 = 0;
    srtm41_01Data.m02 = 20;
    srtm41_01Data.m10 = 0;
    srtm41_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m12 = 60;
    srtm41_01Data.no_data_value = -32768.0;
    srtm41_01Data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
    srtm41_01Data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
    srtm41_01Data.egm = const_cast<float*>(egm_96->GetDeviceValues());

    Srtm3FormatComputation srtm42_01Data;
    srtm42_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m01 = 0;
    srtm42_01Data.m02 = 25;
    srtm42_01Data.m10 = 0;
    srtm42_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m12 = 60;
    srtm42_01Data.no_data_value = -32768.0;
    srtm42_01Data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
    srtm42_01Data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
    srtm42_01Data.egm = const_cast<float*>(egm_96->GetDeviceValues());

    result.push_back(srtm41_01Data);
    result.push_back(srtm42_01Data);
    return result;
}

void Srtm3ElevationModel::HostToDevice() {
    std::vector<PointerHolder> temp_tiles;
    const auto nr_of_tiles = srtms_.size();
    temp_tiles.resize(nr_of_tiles);
    device_formated_srtm_buffers_.resize(nr_of_tiles);
    srtm_format_info_.resize(nr_of_tiles);
    constexpr dim3 block_size(20, 20);

    for (size_t i = 0; i < nr_of_tiles; i++) {
        const auto x_size = this->srtms_.at(i).GetXSize();
        const auto y_size = this->srtms_.at(i).GetYSize();
        const auto dem_size_bytes = x_size * y_size * sizeof(float);
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_formated_srtm_buffers_.at(i), dem_size_bytes));
        float* temp_buffer;
        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_buffer, dem_size_bytes));
        CHECK_CUDA_ERR(cudaMemcpy(temp_buffer, this->srtms_.at(i).GetHostDataBuffer().data(), dem_size_bytes,
                                  cudaMemcpyHostToDevice));
        this->srtm_format_info_.at(i).x_size = x_size;
        this->srtm_format_info_.at(i).y_size = y_size;
        const dim3 grid_size(cuda::GetGridDim(block_size.x, x_size), cuda::GetGridDim(block_size.y, y_size));

        CHECK_CUDA_ERR(LaunchDemFormatter(grid_size, block_size, this->device_formated_srtm_buffers_.at(i), temp_buffer,
                                          this->srtm_format_info_.at(i)));
        temp_tiles.at(i).pointer = this->device_formated_srtm_buffers_.at(i);
        temp_tiles.at(i).x = x_size;
        temp_tiles.at(i).y = y_size;
        temp_tiles.at(i).z = 1;
        cudaFree(temp_buffer);
    }
    CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_formated_srtm_buffers_info_, nr_of_tiles * sizeof(PointerHolder)));
    CHECK_CUDA_ERR(cudaMemcpy(this->device_formated_srtm_buffers_info_, temp_tiles.data(),
                              nr_of_tiles * sizeof(PointerHolder), cudaMemcpyHostToDevice));
    device_srtm3_tiles_count_ = nr_of_tiles;
}

void Srtm3ElevationModel::DeviceToHost() { CHECK_CUDA_ERR(cudaErrorNotYetImplemented); }

void Srtm3ElevationModel::DeviceFree() {
    REPORT_WHEN_CUDA_ERR(cudaFree(this->device_formated_srtm_buffers_info_));
    device_formated_srtm_buffers_info_ = nullptr;
    for (auto&& buf : device_formated_srtm_buffers_) {
        REPORT_WHEN_CUDA_ERR(cudaFree(buf));
    }
    this->device_formated_srtm_buffers_.clear();
}

}  // namespace snapengine
}  // namespace alus
