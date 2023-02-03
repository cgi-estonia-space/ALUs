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

namespace {
constexpr size_t RASTER_WIDTH_DEG{1};
constexpr size_t RASTER_HEIGHT_DEG{1};
constexpr size_t MAX_LON_COVERAGE{180};
constexpr size_t MAX_LAT_COVERAGE{90};
constexpr size_t RASTER_X_TILE_COUNT{360};
constexpr size_t RASTER_Y_TILE_COUNT{180};
constexpr size_t TILE_HEIGHT_PIXELS{3600};
constexpr double PIXELS_PER_TILE_HEIGHT_INVERTED{1 / static_cast<double>(TILE_HEIGHT_PIXELS)};
constexpr double PIXEL_SIZE_Y_DEGREES{RASTER_HEIGHT_DEG / static_cast<double>(TILE_HEIGHT_PIXELS)};
constexpr double PIXEL_SIZE_Y_DEGREES_INVERTED{1 / PIXEL_SIZE_Y_DEGREES};
constexpr double NO_DATA_VALUE{0.0};
constexpr std::array<size_t, 7> ALLOWED_WIDTHS{TILE_HEIGHT_PIXELS, 2400, 1800, 1200, 720, 360};

int DoubleForCompare(double value, size_t digits) { return value * std::pow(10, digits); }

}  // namespace

namespace alus::dem {

CopDemCog30m::CopDemCog30m(std::vector<std::string> filenames) : filenames_(std::move(filenames)) {
    if (filenames_.size() == 0) {
        throw std::runtime_error("No sense to prepare DEM files without any filenames given");
    }
}

void CopDemCog30m::VerifyProperties(const Property& prop, const Dataset<float>& ds, std::string_view filename) {
    std::string exception_message_header = "Given file '" + std::string(filename) + "'";
    if (prop.pixels_per_tile_y_axis != TILE_HEIGHT_PIXELS || ds.GetRasterSizeY() != TILE_HEIGHT_PIXELS) {
        std::runtime_error(exception_message_header + " height '" + std::to_string(prop.pixels_per_tile_y_axis) +
                           "' is not COPDEM 30m COG height(" + std::to_string(TILE_HEIGHT_PIXELS) + ")");
    }
    if (!std::any_of(ALLOWED_WIDTHS.cbegin(), ALLOWED_WIDTHS.cend(),
                     [&prop](size_t v) { return v == prop.pixels_per_tile_x_axis; })) {
        std::runtime_error(exception_message_header + " width '" + std::to_string(prop.pixels_per_tile_x_axis) +
                           "' is not COPDEM 30m COG width");
    }
    if (prop.pixels_per_tile_x_axis != static_cast<size_t>(ds.GetRasterSizeX())) {
        throw std::logic_error("COPDEM property width does not equal real raster one.");
    }
    constexpr size_t dig_comp{12};
    if (DoubleForCompare(prop.pixels_per_tile_inverted_x_axis, dig_comp) !=
            DoubleForCompare(prop.pixel_size_degrees_x_axis, dig_comp) ||
        DoubleForCompare(prop.pixels_per_tile_inverted_x_axis, dig_comp) !=
            DoubleForCompare(ds.GetPixelSizeLon(), dig_comp)) {
        throw std::runtime_error(exception_message_header +
                                 " pixel 'X' size does not equal to inverted value of pixel count");
    }
    if (DoubleForCompare(prop.pixels_per_tile_inverted_y_axis, dig_comp) !=
            DoubleForCompare(prop.pixel_size_degrees_y_axis, dig_comp) ||
        DoubleForCompare(prop.pixels_per_tile_inverted_y_axis, dig_comp) !=
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
        prop.pixels_per_tile_x_axis = ds.GetRasterSizeX();
        prop.pixels_per_tile_y_axis = ds.GetRasterSizeY();
        prop.pixels_per_tile_inverted_x_axis = 1.0 / prop.pixels_per_tile_x_axis;
        prop.pixels_per_tile_inverted_y_axis = 1.0 / prop.pixels_per_tile_y_axis;
        prop.tiles_x_axis = RASTER_X_TILE_COUNT;
        prop.tiles_y_axis = RASTER_Y_TILE_COUNT;
        prop.raster_width = prop.pixels_per_tile_x_axis * RASTER_X_TILE_COUNT;
        prop.raster_height = TILE_HEIGHT_PIXELS * RASTER_Y_TILE_COUNT;
        prop.no_data_value = NO_DATA_VALUE;
        prop.pixel_size_degrees_x_axis = ds.GetPixelSizeLon();
        prop.pixel_size_degrees_y_axis = std::abs(ds.GetPixelSizeLat());
        prop.pixel_size_degrees_inverted_x_axis = 1 / prop.pixel_size_degrees_x_axis;
        prop.pixel_size_degrees_inverted_y_axis = 1 / prop.pixel_size_degrees_y_axis;
        prop.lat_coverage = 90.0;
        prop.lon_coverage = 180.0;
        prop.lat_origin = ds.GetOriginLat();
        prop.lat_extent = prop.lat_origin + (ds.GetRasterSizeY() * ds.GetPixelSizeLat());
        prop.lon_origin = ds.GetOriginLon();
        prop.lon_extent = prop.lon_origin + (ds.GetRasterSizeX() * ds.GetPixelSizeLon());
        host_dem_properties_.push_back(prop);

        LOGI << "Loaded " << dem_file;
        VerifyProperties(prop, ds, dem_file);
    }
}

void CopDemCog30m::TransferToDeviceImpl() {
    std::vector<PointerHolder> temp_tiles;
    const auto nr_of_tiles = datasets_.size();
    temp_tiles.resize(nr_of_tiles);
    device_formated_buffers_.resize(nr_of_tiles);
    //    constexpr dim3 BLOCK_SIZE(20, 20);

    for (size_t i = 0; i < nr_of_tiles; i++) {
        const auto x_size = host_dem_properties_.at(i).pixels_per_tile_x_axis;
        const auto y_size = host_dem_properties_.at(i).pixels_per_tile_y_axis;
        const auto dem_size_bytes = x_size * y_size * sizeof(float);
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_formated_buffers_.at(i), dem_size_bytes));
        //        float* temp_buffer;
        //        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_buffer, dem_size_bytes));
        CHECK_CUDA_ERR(cudaMemcpy(device_formated_buffers_.at(i), datasets_.at(i).GetHostDataBuffer().data(),
                                  dem_size_bytes, cudaMemcpyHostToDevice));
        //        this->srtm_format_info_.at(i).x_size = x_size;
        //        this->srtm_format_info_.at(i).y_size = y_size;
        //        const dim3 grid_size(cuda::GetGridDim(BLOCK_SIZE.x, x_size),
        //                             cuda::GetGridDim(static_cast<int>(BLOCK_SIZE.y), y_size));
        //
        //        CHECK_CUDA_ERR(LaunchDemFormatter(grid_size, BLOCK_SIZE, this->device_formated_srtm_buffers_.at(i),
        //                                          temp_buffer, this->srtm_format_info_.at(i)));
        //        temp_tiles.at(i).pointer = this->device_formated_srtm_buffers_.at(i);
        // When converting to integer C++ rules cast down positive float numbers and towards zero for negative
        // float numbers. Since values are slightly below whole, for example 34.999567 a whole number 35 is desired
        // for index calculations. Without std::round() first the result would be 34.
        const auto lon = static_cast<int>(std::round(host_dem_properties_.at(i).lon_origin));
        const auto lat = static_cast<int>(std::round(host_dem_properties_.at(i).lat_origin - 1.0));
        // ID according to the bottom left point. E.g W01 + S90 = 0 or E179 + N89 = 359 * 1000 + 179.
        temp_tiles.at(i).id = ComputeId(lon, lat);
        temp_tiles.at(i).x = x_size;
        temp_tiles.at(i).y = y_size;
        temp_tiles.at(i).z = 1;
        temp_tiles.at(i).pointer = device_formated_buffers_.at(i);
        LOGI << "Loading COPDEM COG 30m tile ID " << temp_tiles.at(i).id << " to GPU";
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

    TransferToDeviceImpl();
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
    LOGI << "DESTRUCTOR";
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
        const auto status = f.wait_for(std::chrono::seconds(tile_count * 10));
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

// Calculates according to bottom left point.
int CopDemCog30m::ComputeId(double lon_origin, double lat_origin) {
    return ((lon_origin + MAX_LON_COVERAGE) / RASTER_WIDTH_DEG) * 1000 +
           ((lat_origin + MAX_LAT_COVERAGE) / RASTER_HEIGHT_DEG);
}

}  // namespace alus::dem