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

#include <chrono>
#include <future>
#include <stdexcept>

#include "alus_log.h"

namespace {
constexpr size_t RASTER_WIDTH_DEG{1};
constexpr size_t RASTER_HEIGHT_DEG{1};
constexpr size_t RASTER_X_TILE_COUNT{360};
constexpr size_t RASTER_Y_TILE_COUNT{180};
constexpr size_t TILE_HEIGHT_PIXELS{3600};
constexpr double PIXELS_PER_TILE_HEIGHT_INVERTED{1 / static_cast<double>(TILE_HEIGHT_PIXELS)};
constexpr double PIXEL_SIZE_Y_DEGREES{RASTER_HEIGHT_DEG / static_cast<double>(TILE_HEIGHT_PIXELS)};
constexpr double PIXEL_SIZE_Y_DEGREES_INVERTED{1 / PIXEL_SIZE_Y_DEGREES};
constexpr double NO_DATA_VALUE{0.0};
}

namespace alus::dem {

CopDemCog30m::CopDemCog30m(std::vector<std::string> filenames) : filenames_(std::move(filenames)) {}

void CopDemCog30m::LoadTilesThread() {
    for (auto&& dem_file : filenames_) {
        auto& ds = datasets_.emplace_back(dem_file);
        ds.LoadRasterBand(1);

        Property prop;
        prop.pixels_per_tile_inverted_x_axis = 1 / static_cast<double>(ds.GetRasterSizeX());
        prop.pixels_per_tile_inverted_y_axis = PIXELS_PER_TILE_HEIGHT_INVERTED;
        prop.pixels_per_tile_x_axis = ds.GetRasterSizeX();
        prop.pixels_per_tile_y_axis = TILE_HEIGHT_PIXELS;
        prop.tiles_x_axis = RASTER_X_TILE_COUNT;
        prop.tiles_y_axis = RASTER_Y_TILE_COUNT;
        prop.raster_width = ds.GetRasterSizeX() * RASTER_X_TILE_COUNT;
        prop.raster_height = TILE_HEIGHT_PIXELS * RASTER_Y_TILE_COUNT;
        prop.no_data_value = NO_DATA_VALUE;
        prop.pixel_size_degrees_x_axis = ds.GetPixelSizeLon();
        prop.pixel_size_degrees_y_axis = PIXEL_SIZE_Y_DEGREES;
        prop.pixel_size_degrees_inverted_x_axis = 1 / static_cast<double>(ds.GetRasterSizeX());
        prop.pixel_size_degrees_inverted_y_axis = PIXEL_SIZE_Y_DEGREES_INVERTED;
        prop.lat_coverage = 90.0;
        prop.lon_coverage = 180.0;
        prop.lat_origin = ds.GetOriginLat();
        prop.lat_extent = prop.lat_origin + (ds.GetRasterSizeY() * ds.GetPixelSizeLat());
        prop.lon_origin = ds.GetOriginLon();
        prop.lon_extent = prop.lon_origin + (ds.GetRasterSizeX() * ds.GetPixelSizeLon());
        host_dem_properties_.push_back(prop);

        LOGI << "Loaded " << dem_file;
    }
}

void CopDemCog30m::LoadTiles() {
    if (load_tiles_future_.valid()) {
        throw std::runtime_error("Tile loading have already been commenced. Invalid state for load");
    }

    load_tiles_future_ = std::async([this]() { this->LoadTilesThread(); });
}

size_t CopDemCog30m::GetTileCount() { return {}; }

const PointerHolder* CopDemCog30m::GetBuffers() { return {}; }

const Property* CopDemCog30m::GetProperties() { return {}; }

const std::vector<Property>& CopDemCog30m::GetPropertiesValue() {
    WaitLoadTilesAndCheckErrors();

    return host_dem_properties_;
}

void CopDemCog30m::TransferToDevice() {}

void CopDemCog30m::ReleaseFromDevice() {}

CopDemCog30m::~CopDemCog30m() {
    LOGI << "DESTRUCTOR";
    // Just in case left hanging, do not deal with it if still running.
    if (load_tiles_future_.valid()) {
        load_tiles_future_.wait_for(std::chrono::seconds(0));
    }

    ReleaseFromDevice();
}

void CopDemCog30m::WaitLoadTilesAndCheckErrors() {
    if (load_tiles_future_.valid()) {
        // For each tile 10 seconds is given. If it takes longer, regard this as a stupid slow system.
        const auto status = load_tiles_future_.wait_for(std::chrono::seconds(filenames_.size() * 10));
        if (status == std::future_status::timeout) {
            throw std::runtime_error("Timeout has reached when waiting for DEM tile loading to finish.");
        }
        load_tiles_future_.get();
    }
}

}  // namespace alus::dem