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

#include "copdem_cog_30m.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "alus_log.h"
#include "cuda_util.h"
#include "dem_egm96.h"

namespace {
constexpr size_t RASTER_DEG_RES_X{1};
constexpr size_t RASTER_DEG_RES_Y{1};
constexpr size_t MAX_LON_COVERAGE{180};
constexpr size_t MAX_LAT_COVERAGE{90};
constexpr size_t RASTER_X_TILE_COUNT{360};
constexpr size_t RASTER_Y_TILE_COUNT{180};
constexpr size_t TILE_HEIGHT_PIXELS{3600};
constexpr double NO_DATA_VALUE{0.0};
constexpr std::array<size_t, 7> ALLOWED_WIDTHS{TILE_HEIGHT_PIXELS, 2400, 1800, 1200, 720, 360};
constexpr size_t TIMEOUT_SEC_PER_TILE{10};

int DoubleForCompare(double value, size_t digits) { return value * std::pow(10, digits); }

}  // namespace

namespace alus::dem {

CopDemCog30m::CopDemCog30m(std::vector<std::string> filenames) : filenames_(std::move(filenames)) {
    if (filenames_.size() == 0) {
        throw std::runtime_error("No sense to prepare DEM files without any filenames given");
    }
}

CopDemCog30m::CopDemCog30m(std::vector<std::string> filenames,
                           std::shared_ptr<snapengine::EarthGravitationalModel96> egm96)
    : CopDemCog30m(std::move(filenames)) {
    egm96_ = egm96;
}

void CopDemCog30m::VerifyProperties(const Property& prop, const Dataset<float>& ds, std::string_view filename) {
    std::string exception_message_header = "Given file '" + std::string(filename) + "'";
    if (prop.tile_pixel_count_y != TILE_HEIGHT_PIXELS || ds.GetRasterSizeY() != TILE_HEIGHT_PIXELS) {
        std::runtime_error(exception_message_header + " height '" + std::to_string(prop.tile_pixel_count_y) +
                           "' is not COPDEM 30m COG height(" + std::to_string(TILE_HEIGHT_PIXELS) + ")");
    }
    if (!std::any_of(ALLOWED_WIDTHS.cbegin(), ALLOWED_WIDTHS.cend(),
                     [&prop](size_t v) { return v == prop.tile_pixel_count_x; })) {
        std::runtime_error(exception_message_header + " width '" + std::to_string(prop.tile_pixel_count_x) +
                           "' is not COPDEM 30m COG width");
    }
    if (prop.tile_pixel_count_x != static_cast<size_t>(ds.GetRasterSizeX())) {
        throw std::logic_error("COPDEM property width does not equal real raster one.");
    }
    constexpr size_t dig_comp{12};
    if (DoubleForCompare(prop.tile_pixel_count_inverted_x, dig_comp) !=
            DoubleForCompare(prop.tile_pixel_size_deg_x, dig_comp) ||
        DoubleForCompare(prop.tile_pixel_count_inverted_x, dig_comp) !=
            DoubleForCompare(ds.GetPixelSizeLon(), dig_comp)) {
        throw std::runtime_error(exception_message_header +
                                 " pixel 'X' size does not equal to inverted value of pixel count");
    }
    if (DoubleForCompare(prop.tile_pixel_count_inverted_y, dig_comp) !=
            DoubleForCompare(prop.tile_pixel_size_deg_y, dig_comp) ||
        DoubleForCompare(prop.tile_pixel_count_inverted_y, dig_comp) !=
            DoubleForCompare(std::abs(ds.GetPixelSizeLat()), dig_comp)) {
        throw std::runtime_error(exception_message_header +
                                 " pixel 'Y' size does not equal to inverted value of pixel count");
    }
}

void CopDemCog30m::LoadTilesImpl() {
    for (auto&& dem_file : filenames_) {
        auto& ds = datasets_.emplace_back(dem_file);
        ds.LoadRasterBand(1);

        Property prop;
        prop.tile_pixel_count_x = ds.GetRasterSizeX();
        prop.tile_pixel_count_y = ds.GetRasterSizeY();
        prop.tile_pixel_count_inverted_x = 1.0 / prop.tile_pixel_count_x;
        prop.tile_pixel_count_inverted_y = 1.0 / prop.tile_pixel_count_y;
        prop.grid_tile_count_x = RASTER_X_TILE_COUNT;
        prop.grid_tile_count_y = RASTER_Y_TILE_COUNT;
        prop.grid_total_width_pixels = prop.tile_pixel_count_x * RASTER_X_TILE_COUNT;
        prop.grid_total_height_pixels = TILE_HEIGHT_PIXELS * RASTER_Y_TILE_COUNT;
        prop.no_data_value = NO_DATA_VALUE;
        prop.tile_pixel_size_deg_x = ds.GetPixelSizeLon();
        prop.tile_pixel_size_deg_y = std::abs(ds.GetPixelSizeLat());
        prop.tile_pixel_size_deg_inverted_x = 1 / prop.tile_pixel_size_deg_x;
        prop.tile_pixel_size_deg_inverted_y = 1 / prop.tile_pixel_size_deg_y;
        prop.grid_max_lat = 90.0;
        prop.grid_max_lon = 180.0;
        prop.tile_lat_origin = ds.GetOriginLat();
        prop.tile_lat_extent = prop.tile_lat_origin + (ds.GetRasterSizeY() * ds.GetPixelSizeLat());
        prop.tile_lon_origin = ds.GetOriginLon();
        prop.tile_lon_extent = prop.tile_lon_origin + (ds.GetRasterSizeX() * ds.GetPixelSizeLon());
        host_dem_properties_.push_back(prop);

        VerifyProperties(prop, ds, dem_file);
    }
}

void CopDemCog30m::TransferToDeviceImpl() {
    std::vector<PointerHolder> temp_tiles;
    const auto nr_of_tiles = datasets_.size();
    temp_tiles.resize(nr_of_tiles);
    device_formated_buffers_.resize(nr_of_tiles);
    constexpr dim3 device_block_size(20, 20);
    for (size_t i = 0; i < nr_of_tiles; i++) {
        const auto& tile_prop = host_dem_properties_.at(i);
        const auto x_size = tile_prop.tile_pixel_count_x;
        const auto y_size = tile_prop.tile_pixel_count_y;
        const auto dem_size_bytes = x_size * y_size * sizeof(float);
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_formated_buffers_.at(i), dem_size_bytes));

        if (egm96_) {
            float* device_dem_values;
            CHECK_CUDA_ERR(cudaMalloc((void**)&device_dem_values, dem_size_bytes));
            CHECK_CUDA_ERR(cudaMemcpy(device_dem_values, datasets_.at(i).GetHostDataBuffer().data(), dem_size_bytes,
                                      cudaMemcpyHostToDevice));
            double gt[transform::GEOTRANSFORM_ARRAY_LENGTH];
            datasets_.at(i).GetGdalDataset()->GetGeoTransform(gt);
            EgmFormatProperties prop{};
            prop.tile_size_x = x_size;
            prop.tile_size_y = y_size;
            prop.m00 = gt[transform::TRANSFORM_PIXEL_X_SIZE_INDEX];
            prop.m10 = gt[transform::TRANSFORM_ROTATION_1];
            prop.m01 = gt[transform::TRANSFORM_ROTATION_2];
            prop.m11 = gt[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX];
            prop.m02 = gt[transform::TRANSFORM_LON_ORIGIN_INDEX];
            prop.m12 = gt[transform::TRANSFORM_LAT_ORIGIN_INDEX];
            prop.no_data_value = tile_prop.no_data_value;
            prop.device_egm_array = egm96_->GetDeviceValues();
            const dim3 grid_size(cuda::GetGridDim(device_block_size.x, x_size),
                                 cuda::GetGridDim(device_block_size.y, y_size));
            ConditionWithEgm96(grid_size, device_block_size, device_formated_buffers_.at(i), device_dem_values, prop);
            CHECK_CUDA_ERR(cudaFree(device_dem_values));
        } else {
            CHECK_CUDA_ERR(cudaMemcpy(device_formated_buffers_.at(i), datasets_.at(i).GetHostDataBuffer().data(),
                                      dem_size_bytes, cudaMemcpyHostToDevice));
        }

        // When converting to integer C++ rules cast down positive float numbers and towards zero for negative
        // float numbers. Since values are slightly below whole, for example 34.999567 a whole number 35 is desired
        // for index calculations. Without std::round() first the result would be 34.
        const auto lon = static_cast<int>(std::round(host_dem_properties_.at(i).tile_lon_origin));
        const auto lat = static_cast<int>(std::round(host_dem_properties_.at(i).tile_lat_origin - 1.0));
        // ID according to the bottom left point. E.g W01 + S90 = 0 or E179 + N89 = 359 * 1000 + 179.
        temp_tiles.at(i).id = ComputeId(lon, lat);
        temp_tiles.at(i).x = x_size;
        temp_tiles.at(i).y = y_size;
        temp_tiles.at(i).z = 1;
        temp_tiles.at(i).pointer = device_formated_buffers_.at(i);
        LOGI << "COPDEM COG 30m tile ID " << temp_tiles.at(i).id << " loaded to GPU";
    }
    CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_formated_buffers_table_, nr_of_tiles * sizeof(PointerHolder)));
    CHECK_CUDA_ERR(cudaMemcpy(this->device_formated_buffers_table_, temp_tiles.data(),
                              nr_of_tiles * sizeof(PointerHolder), cudaMemcpyHostToDevice));

    const auto dem_property_count = host_dem_properties_.size();
    CHECK_CUDA_ERR(cudaMalloc(&device_dem_properties_, sizeof(dem::Property) * dem_property_count));
    for (size_t i = 0; i < dem_property_count; i++) {
        CHECK_CUDA_ERR(cudaMemcpy(device_dem_properties_ + i, host_dem_properties_.data() + i, sizeof(dem::Property),
                                  cudaMemcpyHostToDevice));
    }
}

void CopDemCog30m::LoadTiles() {
    if (load_tiles_future_.valid()) {
        throw std::runtime_error("Tile loading have already been commenced. Invalid state for load");
    }

    load_tiles_future_ = std::async([this]() { this->LoadTilesImpl(); });
}

size_t CopDemCog30m::GetTileCount() {
    WaitTransferDeviceAndCheckErrors();

    return device_formated_buffers_.size();
}

const PointerHolder* CopDemCog30m::GetBuffers() {
    WaitTransferDeviceAndCheckErrors();

    return device_formated_buffers_table_;
}

const Property* CopDemCog30m::GetProperties() {
    WaitTransferDeviceAndCheckErrors();

    return device_dem_properties_;
}

const std::vector<Property>& CopDemCog30m::GetPropertiesValue() {
    WaitLoadTilesAndCheckErrors();

    return host_dem_properties_;
}

void CopDemCog30m::TransferToDevice() {
    WaitLoadTilesAndCheckErrors();

    if (host_dem_properties_.size() == 0) {
        throw std::runtime_error("A call to 'LoadTiles()' is needed first in order to transfer DEM files");
    }

    transfer_to_device_future_ = std::async([this]() { this->TransferToDeviceImpl(); });
}

void CopDemCog30m::ReleaseFromDevice() {
    LOGI << "Unloading COPDEM COG 30m tiles";
    for (auto* buf : device_formated_buffers_) {
        REPORT_WHEN_CUDA_ERR(cudaFree(buf));
    }
    device_formated_buffers_.clear();

    if (device_formated_buffers_table_ != nullptr) {
        REPORT_WHEN_CUDA_ERR(cudaFree(device_formated_buffers_table_));
        device_formated_buffers_table_ = nullptr;
    }

    if (device_dem_properties_ != nullptr) {
        REPORT_WHEN_CUDA_ERR(cudaFree(device_dem_properties_));
        device_dem_properties_ = nullptr;
    }
}

CopDemCog30m::~CopDemCog30m() {
    // Just in case left hanging, do not deal with it if still running.
    if (load_tiles_future_.valid()) {
        load_tiles_future_.wait_for(std::chrono::seconds(0));
    }

    if (transfer_to_device_future_.valid()) {
        transfer_to_device_future_.wait_for(std::chrono::seconds(0));
    }

    ReleaseFromDevice();
}

void CopDemCog30m::WaitFutureAndCheckErrorsDefault(std::future<void>& f, size_t tile_count, std::string_view ex_msg) {
    if (f.valid()) {
        // For each tile 10 seconds is given. If it takes longer, regard this as a stupid slow system.
        const auto status = f.wait_for(std::chrono::seconds(tile_count * TIMEOUT_SEC_PER_TILE));
        if (status == std::future_status::timeout) {
            throw std::runtime_error(std::string(ex_msg));
        }
        f.get();
    }
}

void CopDemCog30m::WaitLoadTilesAndCheckErrors() {
    WaitFutureAndCheckErrorsDefault(load_tiles_future_, filenames_.size(),
                                    "Timeout has reached when waiting for DEM tiles loading to finish.");
}

void CopDemCog30m::WaitTransferDeviceAndCheckErrors() {
    if (GetPropertiesValue().size() == 0) {
        std::runtime_error("A call to 'LoadTiles()' is required first");
    }

    WaitFutureAndCheckErrorsDefault(transfer_to_device_future_, host_dem_properties_.size(),
                                    "Timeout has reached when waiting for DEM tiles transfer to finish");

    if (device_formated_buffers_.size() == 0) {
        std::runtime_error("A call to 'TransferToDevice()' is required first");
    }
}

int CopDemCog30m::ComputeId(double lon_origin, double lat_origin) {
    return ((MAX_LON_COVERAGE + lon_origin) / RASTER_DEG_RES_X) * 1000 +
           ((MAX_LAT_COVERAGE - lat_origin) / RASTER_DEG_RES_Y);
}

}  // namespace alus::dem