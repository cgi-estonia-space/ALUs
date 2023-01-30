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

#include <cmath>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "alus_log.h"
#include "cuda_util.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96_computation.h"
#include "srtm3_elevation_model_constants.h"

namespace alus::snapengine {

Srtm3ElevationModel::Srtm3ElevationModel(std::vector<std::string> file_names) : file_names_(std::move(file_names)) {
    device_srtm3_tiles_count_ = 0;
}

Srtm3ElevationModel::~Srtm3ElevationModel() {
    if (copy_thread_.joinable()) {
        copy_thread_.join();
    }
    if (init_thread_.joinable()) {
        init_thread_.join();
    }

    this->ReleaseFromDevice();
}

void Srtm3ElevationModel::ReadSrtmTilesThread() {
    try {
        for (auto&& dem_file : file_names_) {
            // TODO: Priority needed for keeping results as close as possible to SNAP.  // NOLINT is this todo or just a
            // simple comment
            auto& ds = srtms_.emplace_back(dem_file, GeoTransformSourcePriority::WORLDFILE_PAM_INTERNAL_TABFILE_NONE);
            ds.LoadRasterBand(1);
            const auto* geo_transform = ds.GetTransform();

            Srtm3FormatComputation srtm_data{};
            srtm_data.m00 = static_cast<float>(geo_transform[transform::TRANSFORM_PIXEL_X_SIZE_INDEX]);
            srtm_data.m10 = static_cast<float>(geo_transform[transform::TRANSFORM_ROTATION_1]);
            srtm_data.m01 = static_cast<float>(geo_transform[transform::TRANSFORM_ROTATION_2]);
            srtm_data.m11 = static_cast<float>(geo_transform[transform::TRANSFORM_PIXEL_Y_SIZE_INDEX]);
            srtm_data.m02 = static_cast<float>(geo_transform[transform::TRANSFORM_LON_ORIGIN_INDEX]);
            srtm_data.m12 = static_cast<float>(geo_transform[transform::TRANSFORM_LAT_ORIGIN_INDEX]);

            srtm_data.no_data_value = srtm3elevationmodel::NO_DATA_VALUE;
            srtm_data.max_lats = alus::snapengine::earthgravitationalmodel96computation::MAX_LATS;
            srtm_data.max_lons = alus::snapengine::earthgravitationalmodel96computation::MAX_LONS;
            srtm_data.egm = const_cast<float*>(egm_96_->GetDeviceValues());

            this->srtm_format_info_.push_back(srtm_data);

            dem::Property prop;
            prop.pixels_per_tile_inverted_x_axis = srtm3elevationmodel::NUM_PIXELS_PER_TILE_INVERTED;
            prop.pixels_per_tile_inverted_y_axis = srtm3elevationmodel::NUM_PIXELS_PER_TILE_INVERTED;
            prop.pixels_per_tile_x_axis = srtm3elevationmodel::NUM_PIXELS_PER_TILE;
            prop.pixels_per_tile_y_axis = srtm3elevationmodel::NUM_PIXELS_PER_TILE;
            prop.tiles_x_axis = srtm3elevationmodel::NUM_X_TILES;
            prop.tiles_y_axis = srtm3elevationmodel::NUM_Y_TILES;
            prop.raster_width = srtm3elevationmodel::RASTER_WIDTH;
            prop.raster_height = srtm3elevationmodel::RASTER_HEIGHT;
            prop.no_data_value = srtm3elevationmodel::NO_DATA_VALUE;
            prop.pixel_size_degrees_x_axis = srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
            prop.pixel_size_degrees_y_axis = srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
            prop.pixel_size_degrees_inverted_x_axis =
                srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;
            prop.pixel_size_degrees_inverted_y_axis =
                srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE_INVERTED;
            prop.lat_coverage = srtm3elevationmodel::MAX_LAT_COVERAGE;
            prop.lon_coverage = srtm3elevationmodel::MAX_LON_COVERAGE;
            prop.lat_origin = srtm_data.m12;
            prop.lat_extent = prop.lat_origin + (ds.GetRasterSizeY() * ds.GetPixelSizeLat());
            prop.lon_origin = srtm_data.m02;
            prop.lon_extent = prop.lon_origin + (ds.GetRasterSizeX() * ds.GetPixelSizeLon());
            dem_property_host_.push_back(prop);
        }

    } catch (const std::exception&) {
        elevation_exception_ = std::current_exception();
    }
    std::unique_lock init_lock(init_mutex_);
    is_inited_ = true;
    init_var_.notify_all();
}

// use a preconfigured emg96 instance with the tiles already loaded onto the gpu.
void Srtm3ElevationModel::ReadSrtmTiles(std::shared_ptr<EarthGravitationalModel96>& egm_96) {
    if (!init_thread_.joinable()) {
        egm_96_ = egm_96;
        init_thread_ = std::thread(&Srtm3ElevationModel::ReadSrtmTilesThread, this);
    }
}

void Srtm3ElevationModel::HostToDeviceThread() {
    std::unique_lock copy_lock(init_mutex_);
    init_var_.wait(copy_lock, [this]() { return is_inited_; });

    if (elevation_exception_ == nullptr) {
        try {
            std::vector<PointerHolder> temp_tiles;
            const auto nr_of_tiles = srtms_.size();
            temp_tiles.resize(nr_of_tiles);
            device_formated_srtm_buffers_.resize(nr_of_tiles);
            constexpr dim3 BLOCK_SIZE(20, 20);

            for (size_t i = 0; i < nr_of_tiles; i++) {
                const auto x_size = this->srtms_.at(i).GetRasterSizeX();
                const auto y_size = this->srtms_.at(i).GetRasterSizeY();
                const auto dem_size_bytes = x_size * y_size * sizeof(float);
                CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_formated_srtm_buffers_.at(i), dem_size_bytes));
                float* temp_buffer;
                CHECK_CUDA_ERR(cudaMalloc((void**)&temp_buffer, dem_size_bytes));
                CHECK_CUDA_ERR(cudaMemcpy(temp_buffer, this->srtms_.at(i).GetHostDataBuffer().data(), dem_size_bytes,
                                          cudaMemcpyHostToDevice));
                this->srtm_format_info_.at(i).x_size = x_size;
                this->srtm_format_info_.at(i).y_size = y_size;
                const dim3 grid_size(cuda::GetGridDim(BLOCK_SIZE.x, x_size),
                                     cuda::GetGridDim(static_cast<int>(BLOCK_SIZE.y), y_size));

                CHECK_CUDA_ERR(LaunchDemFormatter(grid_size, BLOCK_SIZE, this->device_formated_srtm_buffers_.at(i),
                                                  temp_buffer, this->srtm_format_info_.at(i)));
                temp_tiles.at(i).pointer = this->device_formated_srtm_buffers_.at(i);
                // When converting to integer C++ rules cast down positive float numbers and towards zero for negative
                // float numbers. Since ESRI Worldfile is read first for coordinates, values are slightly below whole,
                // for example 34.999567 a whole number 35 is desired for index calculations. Without std::round() first
                // the result would be 34.
                const auto lon = static_cast<int>(std::round(srtm_format_info_.at(i).m02));
                const auto lat = static_cast<int>(std::round(srtm_format_info_.at(i).m12));
                // ID field in PointerHolder is used for identifying SRTM3 tile indexes. File data in srtm_42_01.tif
                // results in ID with 4201. Yes, it is hacky. SRTM3 files cover earth at full longitude (from -180 to
                // 180) and partially latitude wise (from -60 to 60). Index 0 for longitude starts at -180 degrees
                // (incrementing towards 180). Index 0 for latitude starts at 60 degrees (incrementing towards -60).
                // Therefore SRTM3 file covering longitude 180 and latitude -60 index is 7224.
                temp_tiles.at(i).id =
                    // NOLINTNEXTLINE
                    (((srtm3elevationmodel::MAX_LON_COVERAGE + lon) / srtm3elevationmodel::DEGREE_RES) + 1) * 100 +
                    (((srtm3elevationmodel::MAX_LAT_COVERAGE - lat) / srtm3elevationmodel::DEGREE_RES) + 1);
                LOGI << "Loading SRTM3 tile ID " << temp_tiles.at(i).id << " to GPU";
                temp_tiles.at(i).x = x_size;
                temp_tiles.at(i).y = y_size;
                temp_tiles.at(i).z = 1;
                cudaFree(temp_buffer);
            }
            CHECK_CUDA_ERR(
                cudaMalloc((void**)&this->device_formated_srtm_buffers_info_, nr_of_tiles * sizeof(PointerHolder)));
            CHECK_CUDA_ERR(cudaMemcpy(this->device_formated_srtm_buffers_info_, temp_tiles.data(),
                                      nr_of_tiles * sizeof(PointerHolder), cudaMemcpyHostToDevice));
            device_srtm3_tiles_count_ = nr_of_tiles;

            const auto dem_property_count = dem_property_host_.size();
            CHECK_CUDA_ERR(cudaMalloc(&dem_property_, sizeof(dem::Property) * dem_property_count));
            for (size_t i = 0; i < dem_property_count; i++) {
                CHECK_CUDA_ERR(cudaMemcpy(dem_property_ + i, dem_property_host_.data() + i, sizeof(dem::Property),
                                          cudaMemcpyHostToDevice));
            }
        } catch (const std::exception&) {
            elevation_exception_ = std::current_exception();
        }
    }

    std::unique_lock info_lock(info_mutex_);
    std::unique_lock buffer_lock(buffer_mutex_);
    is_on_device_ = true;
    copy_var_.notify_all();
}

void Srtm3ElevationModel::TransferToDevice() {
    if (!copy_thread_.joinable()) {
        copy_thread_ = std::thread(&Srtm3ElevationModel::HostToDeviceThread, this);
    }
}

void Srtm3ElevationModel::ReleaseFromDevice() {
    if (device_formated_srtm_buffers_info_ != nullptr) {
        LOGI << "Unloading SRTM3 tiles from GPU";
        REPORT_WHEN_CUDA_ERR(cudaFree(this->device_formated_srtm_buffers_info_));
        device_formated_srtm_buffers_info_ = nullptr;
    }

    for (auto&& buf : device_formated_srtm_buffers_) {
        REPORT_WHEN_CUDA_ERR(cudaFree(buf));
    }
    this->device_formated_srtm_buffers_.clear();

    if (dem_property_ != nullptr) {
        REPORT_WHEN_CUDA_ERR(cudaFree(dem_property_));
        dem_property_ = nullptr;
    }
}

PointerHolder* Srtm3ElevationModel::GetBuffers() {
    std::unique_lock info_lock(info_mutex_);
    copy_var_.wait(info_lock, [this]() { return is_on_device_; });

    if (elevation_exception_ != nullptr) {
        std::rethrow_exception(elevation_exception_);
    }
    return device_formated_srtm_buffers_info_;
}
size_t Srtm3ElevationModel::GetTileCount() {
    std::unique_lock buffer_lock(buffer_mutex_);
    copy_var_.wait(buffer_lock, [this]() { return is_on_device_; });

    if (elevation_exception_ != nullptr) {
        std::rethrow_exception(elevation_exception_);
    }
    return device_srtm3_tiles_count_;
}

const dem::Property* Srtm3ElevationModel::GetProperties() {
    std::unique_lock info_lock(info_mutex_);
    copy_var_.wait(info_lock, [this]() { return is_on_device_; });

    if (elevation_exception_ != nullptr) {
        std::rethrow_exception(elevation_exception_);
    }
    return dem_property_;
}

const std::vector<dem::Property>& Srtm3ElevationModel::GetPropertiesValue() {
    std::unique_lock info_lock(info_mutex_);
    copy_var_.wait(info_lock, [this]() { return is_on_device_; });

    if (elevation_exception_ != nullptr) {
        std::rethrow_exception(elevation_exception_);
    }
    return dem_property_host_;
}

}  // namespace alus::snapengine
