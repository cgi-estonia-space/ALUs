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
#include "terrain_correction.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "alus_log.h"
#include "crs_geocoding.h"
#include "cuda_util.h"
#include "gdal_util.h"
#include "general_constants.h"
#include "raster_properties.h"
#include "shapes_util.h"
#include "snap-engine-utilities/engine-utilities//eo/constants.h"
#include "tc_tile.h"
#include "terrain_correction_constants.h"
#include "terrain_correction_kernel.h"
#include "tie_point_geocoding.h"
#include "tile_queue.h"

#define TC_DEBUG_DEVICE_ARRAYS 0
#define TC_DEBUG_SRGR 0

namespace {
void InitThreadContext(alus::terraincorrection::PerThreadData* ctx, size_t max_tile_size, bool use_pinned_memory) {
    ctx->use_pinned_memory = use_pinned_memory;
    // tradeoff between source target size ratio, too large and we spend too much time allocating pinned memory
    // having it too low means stopping streams from overlapping
    constexpr size_t src_ratio = 3;
    ctx->source_buffer_size = max_tile_size * src_ratio;

    ctx->h_source_rectangle_result.Allocate(use_pinned_memory, 1);
    ctx->h_target_tile.Allocate(use_pinned_memory, max_tile_size);
    ctx->h_source_tile.Allocate(use_pinned_memory, max_tile_size * src_ratio);
    ctx->h_resampling_tile.Allocate(use_pinned_memory, 1);

    size_t gpu_mem_sz = 0;
    gpu_mem_sz += max_tile_size * sizeof(double);                 // azimuth index
    gpu_mem_sz += max_tile_size * sizeof(double);                 // range index
    gpu_mem_sz += max_tile_size * sizeof(float);                  // target tile buffer
    gpu_mem_sz += sizeof(*ctx->h_source_rectangle_result.Get());  // get src rectangle result
    gpu_mem_sz += sizeof(*ctx->h_resampling_tile.Get());          // resampling tile
    gpu_mem_sz += gpu_mem_sz / 20;                                // alignment paddings
    ctx->device_memory_arena.ReserveMemory(gpu_mem_sz);

    ctx->d_source_buffer.Resize(max_tile_size * src_ratio);

    CHECK_CUDA_ERR(cudaStreamCreate(&ctx->stream));
}
}  // namespace

void FreeThreadContext(alus::terraincorrection::PerThreadData* ctx) {
    // all other elements free by destructors
    cudaStreamDestroy(ctx->stream);
}

namespace alus::terraincorrection {

enum class DataSinkOption {
    GDAL_DATASET,
    MEMORY_BUFFER
};

struct TerrainCorrection::SharedThreadData {
    // read-write access, must explictly synchronize between threads
    ThreadSafeTileQueue<TcTileIndexPair> tile_queue;
    GDALDataset* input_dataset = nullptr;
    std::mutex gdal_read_mutex;
    DataSinkOption sink_option;
    GDALDataset* output_dataset = nullptr;
    std::mutex gdal_write_mutex;
    float no_data_value;
    int output_buffer_stride;
    std::shared_ptr<float[]> output_buffer{};
    std::exception_ptr exception_ptr = nullptr;
    std::mutex exception_mutex;

    // read only access, no synchronization needed
    const TerrainCorrection* terrain_correction = nullptr;
    GeoTransformParameters target_geo_transform = {};
    int diff_lat = 0;
    size_t max_tile_size = 0;
    bool use_pinned_memory = false;
    bool db_values = false;
};

void FillGetPositionMetadata(GetPositionMetadata& get_position_metadata, const ComputationMetadata& comp_metadata,
                             double line_time_interval_in_days) {
    get_position_metadata.first_line_utc = comp_metadata.first_line_time_mjd;
    get_position_metadata.line_time_interval = line_time_interval_in_days;
    get_position_metadata.wavelength = snapengine::eo::constants::LIGHT_SPEED /
                                       (comp_metadata.radar_frequency * snapengine::eo::constants::ONE_MILLION);
    get_position_metadata.range_spacing = comp_metadata.range_spacing;
    get_position_metadata.near_edge_slant_range = comp_metadata.slant_range_to_first_pixel;
}

void TerrainCorrection::DebugDeviceArrays() const {
    auto d_pos = d_get_position_metadata_.sensor_position;
    auto d_vel = d_get_position_metadata_.sensor_velocity;
    auto d_osv = d_get_position_metadata_.orbit_state_vectors;
    auto d_osv_lut = d_get_position_metadata_.orbit_state_vector_lut;
    std::vector<snapengine::PosVector> h_positions(d_pos.size);
    std::vector<snapengine::PosVector> h_velocities(d_vel.size);
    std::vector<snapengine::OrbitStateVectorComputation> h_osv(d_osv.size);
    std::vector<double> h_osv_lut(d_osv_lut.size);

    cuda::CopyArrayD2H(h_positions.data(), d_pos.array, h_positions.size());
    cuda::CopyArrayD2H(h_velocities.data(), d_vel.array, h_velocities.size());
    cuda::CopyArrayD2H(h_osv.data(), d_osv.array, h_osv.size());
    cuda::CopyArrayD2H(h_osv_lut.data(), d_osv_lut.array, h_osv_lut.size());
    // arrays on host, can be viewed in debugger
    LOGD << "sensor position size = " << h_positions.size() << " velocity = " << h_velocities.size();
    LOGD << "osv size = " << h_osv.size() << " osv lut size = " << h_osv_lut.size();

    std::stringstream debug_str_buffer;
    //    for (const auto& pos : h_positions) {
    //        debug_str_buffer << std::setprecision(16) << pos.x << " " << pos.y << " " << pos.z << std::endl;
    //    }
    //    LOGD << "Positions:" << std::endl << debug_str_buffer.str();
    //
    //    for (const auto& pos : h_velocities) {
    //        debug_str_buffer << std::setprecision(16) << pos.x << " " << pos.y << " " << pos.z << std::endl;
    //    }
    //    LOGD << "Velocities:" << std::endl << debug_str_buffer.str();

    size_t index{0};
    for (const auto& o : h_osv) {
        debug_str_buffer << "Orbit vector " << ++index << std::endl
                         << std::setprecision(16) << o.timeMjd_ << std::endl
                         << std::setprecision(16) << o.xPos_ << std::endl
                         << std::setprecision(16) << o.yPos_ << std::endl
                         << std::setprecision(16) << o.zPos_ << std::endl
                         << std::setprecision(16) << o.xVel_ << std::endl
                         << std::setprecision(16) << o.yVel_ << std::endl
                         << std::setprecision(16) << o.zVel_ << std::endl;
    }
    LOGD << std::endl << debug_str_buffer.str();
}

void TerrainCorrection::DebugSrgrEntries() const {
    std::stringstream debug_str;
    for (size_t i{0}; i < metadata_.srgr_coefficients.size(); i++) {
        debug_str << "Srgr entry " << i + 1 << std::endl
                  << std::setprecision(16) << metadata_.srgr_coefficients.at(i).time_mjd << std::endl
                  << std::setprecision(16) << metadata_.srgr_coefficients.at(i).ground_range_origin << std::endl;
        for (size_t j{0}; j < metadata_.srgr_coefficients.at(i).coefficient.size(); j++) {
            debug_str << std::setprecision(16) << metadata_.srgr_coefficients.at(i).coefficient.at(j) << std::endl;
        }
    }
    LOGD << debug_str.str();
}

void TerrainCorrection::CreateHostMetadata(double line_time_interval_in_days) {
    h_orbit_state_vectors_.clear();
    for (auto&& o : metadata_.orbit_state_vectors2) {
        h_orbit_state_vectors_.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }
    d_get_position_metadata_.first_line_utc = metadata_.first_line_time->GetMjd();
    d_get_position_metadata_.line_time_interval = line_time_interval_in_days;
    d_get_position_metadata_.wavelength =
        snapengine::eo::constants::LIGHT_SPEED / (metadata_.radar_frequency * snapengine::eo::constants::ONE_MILLION);
    d_get_position_metadata_.range_spacing = metadata_.range_spacing;
    d_get_position_metadata_.near_edge_slant_range = metadata_.slant_range_to_first_pixel;
}

void TerrainCorrection::CreateGetPositionDeviceArrays(int y_size, double line_time_interval_in_days) {
    // orbit state vectors -> GPU
    auto& kernel_orbits = d_get_position_metadata_.orbit_state_vectors;
    kernel_orbits.size = h_orbit_state_vectors_.size();
    CHECK_CUDA_ERR(cudaMalloc(&kernel_orbits.array, kernel_orbits.ByteSize()));
    cuda_arrays_to_clean_.push_back(kernel_orbits.array);
    cuda::CopyArrayH2D(kernel_orbits.array, h_orbit_state_vectors_.data(), h_orbit_state_vectors_.size());

    // get position optimization lookup table -> GPU
    std::vector<double> h_osv_lookup = CalculateOrbitStateVectorLUT(h_orbit_state_vectors_);
    auto& k_osv_lookup = d_get_position_metadata_.orbit_state_vector_lut;
    k_osv_lookup.size = h_osv_lookup.size();
    CHECK_CUDA_ERR(cudaMalloc(&k_osv_lookup.array, k_osv_lookup.ByteSize()));
    cuda_arrays_to_clean_.push_back(k_osv_lookup.array);
    cuda::CopyArrayH2D(k_osv_lookup.array, h_osv_lookup.data(), h_osv_lookup.size());

    // allocate position and velocity arrays
    auto& k_sensor_pos = d_get_position_metadata_.sensor_position;
    auto& k_sensor_vel = d_get_position_metadata_.sensor_velocity;
    k_sensor_pos.size = k_sensor_vel.size = y_size;

    CHECK_CUDA_ERR(cudaMalloc(&k_sensor_pos.array, k_sensor_pos.ByteSize()));
    cuda_arrays_to_clean_.push_back(k_sensor_pos.array);
    CHECK_CUDA_ERR(cudaMalloc(&k_sensor_vel.array, k_sensor_vel.ByteSize()));
    cuda_arrays_to_clean_.push_back(k_sensor_vel.array);

    // calculate positions and velocities for each line

    CalculateVelocitiesAndPositions(y_size, metadata_.first_line_time->GetMjd(), line_time_interval_in_days,
                                    d_get_position_metadata_.orbit_state_vectors,
                                    d_get_position_metadata_.sensor_velocity, d_get_position_metadata_.sensor_position);
}

TerrainCorrection::TerrainCorrection(GDALDataset* input_dataset, const RangeDopplerTerrainMetadata& metadata,
                                     std::shared_ptr<snapengine::TiePointGrid> lat_tie_point_grid,
                                     std::shared_ptr<snapengine::TiePointGrid> lon_tie_point_grid,
                                     const PointerHolder* dem_tiles, size_t dem_tiles_length,
                                     const dem::Property* dem_property, const dem::Type dem_type,
                                     const std::vector<dem::Property>& dem_property_value, int selected_band_id,
                                     bool use_average_scene_height)
    : input_ds_{input_dataset},
      metadata_{metadata},
      d_dem_tiles_(dem_tiles),
      d_dem_tiles_length_(dem_tiles_length),
      dem_property_{dem_property},
      dem_type_{dem_type},
      dem_property_value_{dem_property_value},
      selected_band_id_(selected_band_id),
      lat_tie_point_grid_{lat_tie_point_grid},
      lon_tie_point_grid_{lon_tie_point_grid},
      use_average_scene_height_{use_average_scene_height} {}

void TerrainCorrection::ExecuteTerrainCorrection(std::string_view output_file_name, size_t tile_width,
                                                 size_t tile_height, bool output_db_values) {
    // Calculate target dimensions
    auto const ds_y_size{
        static_cast<size_t>(input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetYSize())};

    snapengine::geocoding::TiePointGeocoding source_geocoding(lat_tie_point_grid_, lon_tie_point_grid_);
    snapengine::old::Product target_product = CreateTargetProduct(&source_geocoding, output_file_name);
    auto const target_x_size{target_product.dataset_->GetRasterBand(1)->GetXSize()};
    auto const target_y_size{target_product.dataset_->GetRasterBand(1)->GetYSize()};

    int diff_lat = static_cast<int>(std::abs(target_product.geocoding_->GetPixelCoordinates(0, 0).lat -
                                             target_product.geocoding_->GetPixelCoordinates(0, target_y_size - 1).lat));

    // Populate GeoTransformParameters
    double target_geo_transform_array[6];
    target_product.dataset_->GetGeoTransform(target_geo_transform_array);
    GeoTransformParameters const target_geo_transform{GeoTransformConstruct::BuildFromGdal(target_geo_transform_array)};

    target_product.dataset_->GetRasterBand(1)->SetDescription(metadata_.band_info.front().band_name.c_str());

    AddMetadataTo(target_product.dataset_.get(), metadata_output_);

    const auto line_time_interval_in_days{(metadata_.last_line_time->GetMjd() - metadata_.first_line_time->GetMjd()) /
                                          static_cast<double>(ds_y_size - 1)};

    CreateHostMetadata(line_time_interval_in_days);
    d_get_position_metadata_.source_image_width =
        input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetXSize();
    CreateGetPositionDeviceArrays(static_cast<int>(ds_y_size), line_time_interval_in_days);

    // Target tile
    snapengine::resampling::Tile target_image{
        0, 0, static_cast<size_t>(target_x_size), static_cast<size_t>(target_y_size), false, false, nullptr};
    std::vector<TcTileIndexPair> tiles =
        CalculateTiles(target_image, {0, 0, target_x_size, target_y_size}, tile_width, tile_height);

    // If bound by GPU, then more than 2 give almost no gains
    // If bound by gdal, then locking for both input and output means more than 3 should give almost no benefit
    constexpr size_t thread_limit = 3;

    const size_t n_threads = (tiles.size() / thread_limit) > 0 ? thread_limit : 1;

    /*
     * Pinned memory allows streams to overlap kernel execution with H2D & D2H transfers,
     * however allocating pinned memory takes time
     */
    const bool use_pinned_memory = (tiles.size() / n_threads) > 5;
    LOGD << "TC tiles = " << tiles.size() << " threads = " << n_threads
         << " transfer mode = " << (use_pinned_memory ? "pinned" : "paged");

    // setup data for use by threads, these must outlive threads themselves
    SharedThreadData shared_data = {};
    shared_data.terrain_correction = this;
    shared_data.sink_option = DataSinkOption::GDAL_DATASET;
    shared_data.output_dataset = target_product.dataset_.get();
    const auto band_info = metadata_.band_info.front();
    shared_data.no_data_value = band_info.no_data_value_used && band_info.no_data_value.has_value()
                                    ? band_info.no_data_value.value()
                                    : input_ds_->GetRasterBand(selected_band_id_)->GetNoDataValue();
    shared_data.input_dataset = input_ds_;

    shared_data.target_geo_transform = target_geo_transform;
    shared_data.diff_lat = diff_lat;
    shared_data.use_pinned_memory = use_pinned_memory;
    shared_data.max_tile_size = tile_height * tile_width;
    shared_data.tile_queue.InsertData(std::move(tiles));
    shared_data.db_values = output_db_values;

    CreateSrgrCoefficientsOnDevice();

#if TC_DEBUG_DEVICE_ARRAYS
    DebugDeviceArrays();
#endif

#if TC_DEBUG_SRGR
    DebugSrgrEntries();
#endif

    std::vector<PerThreadData> context_vec(n_threads);
    std::vector<std::thread> threads_vec;

    for (size_t i = 0; i < n_threads; i++) {
        threads_vec.emplace_back(TileLoop, &shared_data, &context_vec.at(i));
    }

    for (auto& thread : threads_vec) {
        thread.join();
    }
    for (auto& ctx : context_vec) {
        FreeThreadContext(&ctx);
    }

    if (shared_data.exception_ptr != nullptr) {
        std::rethrow_exception(shared_data.exception_ptr);
    }
}

SimpleDataset<float> TerrainCorrection::ExecuteTerrainCorrection(size_t tile_width, size_t tile_height,
                                                                 bool output_db_values) {
    // Calculate target dimensions
    auto const ds_y_size{
        static_cast<size_t>(input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetYSize())};

    snapengine::geocoding::TiePointGeocoding source_geocoding(lat_tie_point_grid_, lon_tie_point_grid_);
    int diff_lat;
    auto target_dataset = CreateSimpleTargetProduct(&source_geocoding, diff_lat);
    auto const target_x_size{target_dataset.width};
    auto const target_y_size{target_dataset.height};

    GeoTransformParameters const target_geo_transform{
        GeoTransformConstruct::BuildFromGdal(target_dataset.geo_transform)};

    const auto line_time_interval_in_days{(metadata_.last_line_time->GetMjd() - metadata_.first_line_time->GetMjd()) /
                                          static_cast<double>(ds_y_size - 1)};

    CreateHostMetadata(line_time_interval_in_days);
    d_get_position_metadata_.source_image_width =
        input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetXSize();
    CreateGetPositionDeviceArrays(static_cast<int>(ds_y_size), line_time_interval_in_days);

    // Target tile
    snapengine::resampling::Tile target_image{
        0, 0, static_cast<size_t>(target_x_size), static_cast<size_t>(target_y_size), false, false, nullptr};
    std::vector<TcTileIndexPair> tiles =
        CalculateTiles(target_image, {0, 0, target_x_size, target_y_size}, tile_width, tile_height);

    // If bound by GPU, then more than 2 give almost no gains
    // If bound by gdal, then locking for both input and output means more than 3 should give almost no benefit
    constexpr size_t thread_limit = 3;

    const size_t n_threads = (tiles.size() / thread_limit) > 0 ? thread_limit : 1;

    /*
     * Pinned memory allows streams to overlap kernel execution with H2D & D2H transfers,
     * however allocating pinned memory takes time
     */
    const bool use_pinned_memory = (tiles.size() / n_threads) > 5;
    LOGD << "TC tiles = " << tiles.size() << " threads = " << n_threads
         << " transfer mode = " << (use_pinned_memory ? "pinned" : "paged");

    // setup data for use by threads, these must outlive threads themselves
    SharedThreadData shared_data = {};
    shared_data.terrain_correction = this;
    shared_data.sink_option = DataSinkOption::MEMORY_BUFFER;
    shared_data.output_buffer_stride = target_x_size;
    shared_data.output_buffer = target_dataset.buffer;
    shared_data.no_data_value = target_dataset.no_data;
    shared_data.input_dataset = input_ds_;

    shared_data.target_geo_transform = target_geo_transform;
    shared_data.diff_lat = diff_lat;
    shared_data.use_pinned_memory = use_pinned_memory;
    shared_data.max_tile_size = tile_height * tile_width;
    shared_data.tile_queue.InsertData(std::move(tiles));
    shared_data.db_values = output_db_values;

    CreateSrgrCoefficientsOnDevice();

#if TC_DEBUG_DEVICE_ARRAYS
    DebugDeviceArrays();
#endif

#if TC_DEBUG_SRGR
    DebugSrgrEntries();
#endif

    std::vector<PerThreadData> context_vec(n_threads);
    std::vector<std::thread> threads_vec;

    for (size_t i = 0; i < n_threads; i++) {
        threads_vec.emplace_back(TileLoop, &shared_data, &context_vec.at(i));
    }

    for (auto& thread : threads_vec) {
        thread.join();
    }
    for (auto& ctx : context_vec) {
        FreeThreadContext(&ctx);
    }

    if (shared_data.exception_ptr != nullptr) {
        std::rethrow_exception(shared_data.exception_ptr);
    }

    return target_dataset;
}

std::vector<double> TerrainCorrection::ComputeImageBoundary(const snapengine::geocoding::Geocoding* geocoding,
                                                            int source_width, int source_height) {
    std::vector<double> image_boundary{90, -90, 180, -180};  // lat_min, lat_max, lon_min, lon_max

    Coordinates geo_pos_first_near = geocoding->GetPixelCoordinates(0.5, 0.5);
    Coordinates geo_pos_first_far = geocoding->GetPixelCoordinates(source_width - 0.5, 0.5);
    Coordinates geo_pos_last_near = geocoding->GetPixelCoordinates(0.5, source_height - 0.5);
    Coordinates geo_pos_last_far = geocoding->GetPixelCoordinates(source_width - 0.5, source_height - 0.5);

    double lats[]{geo_pos_first_near.lat, geo_pos_first_far.lat, geo_pos_last_near.lat, geo_pos_last_far.lat};
    double lons[]{geo_pos_first_near.lon, geo_pos_first_far.lon, geo_pos_last_near.lon, geo_pos_last_far.lon};

    for (auto lat : lats) {
        if (lat < image_boundary[0]) {
            image_boundary[0] = lat;
        }
        if (lat > image_boundary[1]) {
            image_boundary[1] = lat;
        }
    }

    for (auto lon : lons) {
        if (lon < image_boundary[2]) {
            image_boundary[2] = lon;
        }
        if (lon > image_boundary[3]) {
            image_boundary[3] = lon;
        }
    }

    if (image_boundary[3] - image_boundary[2] >= 180) {
        image_boundary[2] = 360.0;
        image_boundary[3] = 0.0;
        for (auto lon : lons) {
            if (lon < 0) {
                lon += 360.0;
            }
            if (lon < image_boundary[2]) {
                image_boundary[2] = lon;
            }
            if (lon > image_boundary[3]) {
                image_boundary[3] = lon;
            }
        }
    }

    return image_boundary;
}

// TODO: this is a very good place for optimisation in case of small tiles (420x416 tiles causes 238 iterations)
// (SNAPGPU-216)
std::vector<TcTileIndexPair> TerrainCorrection::CalculateTiles(const snapengine::resampling::Tile& base_image,
                                                               Rectangle dest_bounds, int tile_width,
                                                               int tile_height) const {
    std::vector<TcTileIndexPair> output_tiles;
    int x_count = base_image.width / tile_width + 1;
    int y_count = base_image.height / tile_height + 1;

    for (int x_index = 0; x_index < x_count; x_index++) {
        for (int y_index = 0; y_index < y_count; y_index++) {
            Rectangle rectangle{base_image.x_0 + x_index * tile_width, base_image.y_0 + y_index * tile_height,
                                tile_width, tile_height};
            if (rectangle.x > dest_bounds.width || rectangle.y > dest_bounds.height) {
                continue;
            }
            Rectangle intersection = shapeutils::GetIntersection(rectangle, dest_bounds);
            if (intersection.width == 0 || intersection.height == 0) {
                continue;
            }
            TcTileIndexPair tile{0,
                                 0,
                                 0,
                                 0,
                                 intersection.x,
                                 intersection.y,
                                 static_cast<size_t>(intersection.width),
                                 static_cast<size_t>(intersection.height)};
            output_tiles.push_back(tile);
        }
    }

    return output_tiles;
}

snapengine::old::Product TerrainCorrection::CreateTargetProduct(
    const snapengine::geocoding::Geocoding* source_geocoding, const std::string_view output_filename) {
    double pixel_spacing_in_meter = metadata_.azimuth_spacing;
    double pixel_spacing_in_degree =
        pixel_spacing_in_meter / snapengine::eo::constants::SEMI_MAJOR_AXIS * snapengine::eo::constants::RTOD;

    OGRSpatialReference target_crs;
    // EPSG:4326 is a WGS84 code
    target_crs.importFromEPSG(4326);

    std::vector<double> image_boundary = ComputeImageBoundary(
        source_geocoding, input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetXSize(),
        input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetYSize());

    double pixel_size_x = pixel_spacing_in_degree;
    double pixel_size_y = pixel_spacing_in_degree;

    // Calculates a rectangle coordinates (x0, y0, width, height) for new product based on source image diagonals
    std::vector<double> target_bounds{image_boundary[3] < image_boundary[2] ? image_boundary[3] : image_boundary[2],
                                      image_boundary[1] < image_boundary[0] ? image_boundary[1] : image_boundary[0],
                                      std::abs(image_boundary[3] - image_boundary[2]),
                                      std::abs(image_boundary[1] - image_boundary[0])};

    CHECK_OGR_ERROR(target_crs.Validate());

    GDALDriver* output_driver = GetGdalGeoTiffDriver();

    CHECK_GDAL_PTR(output_driver);

    GDALDataset* output_dataset;
    int a{static_cast<int>(std::floor((target_bounds[2]) / pixel_size_x))};
    int b{static_cast<int>(std::floor((target_bounds[3]) / pixel_size_y))};

    std::string x_tile_sz = FindOptimalTileSize(a);
    std::string y_tile_sz = FindOptimalTileSize(b);

    LOGD << "TC output dimensions = (" << a << ", " << b << ") block size = (" << x_tile_sz << ", " << y_tile_sz << ")";

    // TODO Optimization, should tiff tile size determine calculation tile size or vice versa?
    char** output_driver_options = nullptr;
    output_driver_options = CSLSetNameValue(output_driver_options, "TILED", "YES");
    output_driver_options = CSLSetNameValue(output_driver_options, "BLOCKXSIZE", x_tile_sz.c_str());
    output_driver_options = CSLSetNameValue(output_driver_options, "BLOCKYSIZE", y_tile_sz.c_str());

    auto csl_destroy = [](char** options) { CSLDestroy(options); };
    std::unique_ptr<char*, decltype(csl_destroy)> driver_opt_guard(output_driver_options, csl_destroy);
    output_dataset = output_driver->Create(output_filename.data(), a, b, 1, GDT_Float32, output_driver_options);

    CHECK_GDAL_PTR(output_dataset);

    output_.first = output_filename;
    output_.second.reset(output_dataset, [](auto dataset) { GDALClose(dataset); });

    double output_geo_transform[6] = {target_bounds[0], pixel_size_x, 0, target_bounds[1] + target_bounds[3], 0,
                                      -pixel_size_y};

    // Apply shear and translate
    output_geo_transform[0] = output_geo_transform[0] + output_geo_transform[1] * (-0.5);
    output_geo_transform[3] = output_geo_transform[3] + output_geo_transform[5] * (-0.5);

    output_.second->SetGeoTransform(output_geo_transform);
    char* projection_wkt = nullptr;
    auto cpl_free = [](char* csl_data) { CPLFree(csl_data); };
    target_crs.exportToWkt(&projection_wkt);
    std::unique_ptr<char, decltype(cpl_free)> projection_guard(projection_wkt, cpl_free);
    output_.second->SetProjection(projection_wkt);
    auto band_info = metadata_.band_info.at(0);
    output_.second->GetRasterBand(selected_band_id_)
        ->SetNoDataValue(band_info.no_data_value_used && band_info.no_data_value.has_value()
                             ? band_info.no_data_value.value()
                             : input_ds_->GetRasterBand(selected_band_id_)->GetNoDataValue());

    GeoTransformParameters geo_transform_parameters{output_geo_transform[0], output_geo_transform[3],
                                                    output_geo_transform[1], output_geo_transform[5]};

    snapengine::old::Product target_product{std::unique_ptr<snapengine::geocoding::CrsGeocoding>{}, output_.second,
                                            "TIFF"};
    target_product.geocoding_ = std::make_unique<snapengine::geocoding::CrsGeocoding>(geo_transform_parameters);

    return target_product;
}

SimpleDataset<float> TerrainCorrection::CreateSimpleTargetProduct(const snapengine::geocoding::Geocoding* geocoding,
                                                                  int& diff_lat) {
    SimpleDataset<float> simple_target_dataset;
    double pixel_spacing_in_meter = metadata_.azimuth_spacing;
    double pixel_spacing_in_degree =
        pixel_spacing_in_meter / snapengine::eo::constants::SEMI_MAJOR_AXIS * snapengine::eo::constants::RTOD;

    OGRSpatialReference target_crs;
    // EPSG:4326 is target_ds_width WGS84 code
    target_crs.importFromEPSG(4326);

    std::vector<double> image_boundary =
        ComputeImageBoundary(geocoding, input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetXSize(),
                             input_ds_->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)->GetYSize());

    double pixel_size_x = pixel_spacing_in_degree;
    double pixel_size_y = pixel_spacing_in_degree;

    // Calculates target_ds_width rectangle coordinates (x0, y0, width, height) for new product based on source image
    // diagonals
    std::vector<double> target_bounds{image_boundary[3] < image_boundary[2] ? image_boundary[3] : image_boundary[2],
                                      image_boundary[1] < image_boundary[0] ? image_boundary[1] : image_boundary[0],
                                      std::abs(image_boundary[3] - image_boundary[2]),
                                      std::abs(image_boundary[1] - image_boundary[0])};

    CHECK_OGR_ERROR(target_crs.Validate());

    simple_target_dataset.width = static_cast<int>(std::floor((target_bounds[2]) / pixel_size_x));
    simple_target_dataset.height = static_cast<int>(std::floor((target_bounds[3]) / pixel_size_y));

    double output_geo_transform[6] = {target_bounds[0], pixel_size_x, 0, target_bounds[1] + target_bounds[3], 0,
                                      -pixel_size_y};
    // Apply shear and translate
    output_geo_transform[0] = output_geo_transform[0] + output_geo_transform[1] * (-0.5);
    output_geo_transform[3] = output_geo_transform[3] + output_geo_transform[5] * (-0.5);
    std::memcpy(simple_target_dataset.geo_transform, output_geo_transform, 6 * sizeof(double));

    {
        char* projection_wkt = nullptr;
        auto cpl_free = [](char* csl_data) { CPLFree(csl_data); };
        target_crs.exportToWkt(&projection_wkt);
        std::unique_ptr<char, decltype(cpl_free)> projection_guard(projection_wkt, cpl_free);
        simple_target_dataset.projection_wkt = std::string(projection_wkt);
    }
    auto band_info = metadata_.band_info.at(0);
    simple_target_dataset.no_data = band_info.no_data_value_used && band_info.no_data_value.has_value()
                                        ? band_info.no_data_value.value()
                                        : input_ds_->GetRasterBand(selected_band_id_)->GetNoDataValue();

    GeoTransformParameters geo_transform_parameters{output_geo_transform[0], output_geo_transform[3],
                                                    output_geo_transform[1], output_geo_transform[5]};

    const auto geocoding_stub = snapengine::geocoding::CrsGeocoding(geo_transform_parameters);
    diff_lat = static_cast<int>(std::abs(geocoding_stub.GetPixelCoordinates(0, 0).lat -
                                         geocoding_stub.GetPixelCoordinates(0, simple_target_dataset.height - 1).lat));
    simple_target_dataset.buffer =
        std::shared_ptr<float[]>(new float[simple_target_dataset.width * simple_target_dataset.height]);

    return simple_target_dataset;
}

TerrainCorrection::~TerrainCorrection() { FreeCudaArrays(); }
void TerrainCorrection::FreeCudaArrays() {
    for (auto&& a : cuda_arrays_to_clean_) {
        cudaFree(a);
    }
    cuda_arrays_to_clean_.clear();
}
std::pair<std::string, std::shared_ptr<GDALDataset>> TerrainCorrection::GetOutputDataset() const { return output_; }

void SetTileSourceCoordinates(TcTileIndexPair& tile_coordinates, const Rectangle& source_rectangle) {
    tile_coordinates.source_height = source_rectangle.height;
    tile_coordinates.source_width = source_rectangle.width;
    tile_coordinates.source_x_0 = source_rectangle.x;
    tile_coordinates.source_y_0 = source_rectangle.y;
}

void TerrainCorrection::CalculateTile(TcTileIndexPair tile_coordinates, SharedThreadData* shared, PerThreadData* ctx) {
    auto* band = shared->input_dataset->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND);

    GetSourceRectangleKernelArgs src_args = {};
    src_args.get_position_metadata = shared->terrain_correction->d_get_position_metadata_;
    src_args.use_avg_scene_height = shared->terrain_correction->use_average_scene_height_;
    src_args.avg_scene_height = shared->terrain_correction->metadata_.avg_scene_height;
    src_args.dem_property = shared->terrain_correction->dem_property_;
    src_args.dem_type = shared->terrain_correction->dem_type_;
    if (!shared->terrain_correction->use_average_scene_height_) {
        src_args.dem_no_data_value = shared->terrain_correction->dem_property_value_.front().no_data_value;
    }
    src_args.source_image_width = band->GetXSize();
    src_args.source_image_height = band->GetYSize();
    src_args.diff_lat = shared->diff_lat;
    src_args.target_geo_transform = shared->target_geo_transform;
    src_args.dem_tiles = {shared->terrain_correction->d_dem_tiles_, shared->terrain_correction->d_dem_tiles_length_};
    src_args.d_srgr_coefficients = shared->terrain_correction->d_srgr_coefficients_;

    const size_t target_sz = tile_coordinates.target_height * tile_coordinates.target_width;

    src_args.d_azimuth_index = ctx->device_memory_arena.AllocArray<double>(target_sz);
    src_args.d_range_index = ctx->device_memory_arena.AllocArray<double>(target_sz);

    snapengine::resampling::ResamplingRaster resampling_raster{0, 0, INVALID_SUB_SWATH_INDEX, {}, nullptr, false};
    resampling_raster.source_rectangle = GetSourceRectangle(tile_coordinates, src_args, ctx);

    const auto& source_rectangle = resampling_raster.source_rectangle;

    const size_t source_rect_sz = source_rectangle.width * source_rectangle.height;
    if (source_rect_sz > 0) {
        // valid source rectangle found, actually calculate target tile data
        SetTileSourceCoordinates(tile_coordinates, resampling_raster.source_rectangle);

        // check if we can use preallocated buffers, if not we have to do cudaMalloc and use non-pinned memory
        // these will hinder the operation of cuda streams
        if (source_rect_sz > ctx->d_source_buffer.GetElemCount()) {
            ctx->d_source_buffer.Resize(source_rect_sz);
        }

        std::unique_ptr<float[]> source_buffer;
        float* h_source_buffer = ctx->h_source_tile.Get();
        if (source_rect_sz > ctx->source_buffer_size) {
            // source rectangle is too big, use backup paged memory
            source_buffer = std::unique_ptr<float[]>(new float[source_rect_sz]);
            h_source_buffer = source_buffer.get();
            if (ctx->use_pinned_memory) {
                LOGV << "target/sz ratio = " << static_cast<double>(source_rect_sz) / static_cast<double>(target_sz);
                LOGV << "target rect = [" << tile_coordinates.target_x_0 << ", " << tile_coordinates.target_y_0 << ", "
                     << tile_coordinates.target_width << ", " << tile_coordinates.target_height << "]";
            }
        }

        *ctx->h_resampling_tile.Get() = {static_cast<int>(tile_coordinates.source_x_0),
                                         static_cast<int>(tile_coordinates.source_y_0),
                                         tile_coordinates.source_width,
                                         tile_coordinates.source_height,
                                         false,
                                         false,
                                         nullptr};

        ctx->h_resampling_tile.Get()->data_buffer = ctx->d_source_buffer.Get();
        resampling_raster.source_tile_i = ctx->device_memory_arena.Alloc<snapengine::resampling::Tile>();

        cuda::CopyAsyncH2D(resampling_raster.source_tile_i, ctx->h_resampling_tile.Get(), ctx->stream);

        {
            std::unique_lock lock(shared->gdal_read_mutex);
            CHECK_GDAL_ERROR(shared->input_dataset->GetRasterBand(gdal::constants::GDAL_DEFAULT_RASTER_BAND)
                                 ->RasterIO(GF_Read, source_rectangle.x, source_rectangle.y, source_rectangle.width,
                                            source_rectangle.height, h_source_buffer, source_rectangle.width,
                                            source_rectangle.height, GDALDataType::GDT_Float32, 0, 0));
        }

        cuda::CopyArrayAsyncH2D(ctx->d_source_buffer.Get(), h_source_buffer, source_rect_sz, ctx->stream);

        cuda::KernelArray<float> target_array = {};
        target_array.array = ctx->device_memory_arena.AllocArray<float>(target_sz);
        target_array.size = target_sz;

        TerrainCorrectionKernelArgs tc_args = {};
        tc_args.source_image_width = src_args.source_image_width;
        tc_args.source_image_height = src_args.source_image_width;
        tc_args.target_no_data_value = shared->no_data_value;
        tc_args.db_values = shared->db_values;
        tc_args.d_azimuth_index = src_args.d_azimuth_index;
        tc_args.d_range_index = src_args.d_range_index;
        tc_args.resampling_raster = resampling_raster;
        tc_args.d_target = target_array;
        CHECK_CUDA_ERR(LaunchTerrainCorrectionKernel(tile_coordinates, tc_args, ctx->h_target_tile.Get(), ctx->stream));

        if (shared->sink_option == DataSinkOption::GDAL_DATASET) {
            std::unique_lock lock(shared->gdal_write_mutex);
            CHECK_GDAL_ERROR(shared->output_dataset->GetRasterBand(shared->terrain_correction->selected_band_id_)
                                 ->RasterIO(GF_Write, tile_coordinates.target_x_0, tile_coordinates.target_y_0,
                                            tile_coordinates.target_width, tile_coordinates.target_height,
                                            ctx->h_target_tile.Get(), tile_coordinates.target_width,
                                            tile_coordinates.target_height, GDT_Float32, 0, 0));
        } else if (shared->sink_option == DataSinkOption::MEMORY_BUFFER) {
            // Strided write.
            for (size_t i = 0; i < tile_coordinates.target_height; i++) {
                memcpy(shared->output_buffer.get() +
                       ((tile_coordinates.target_y_0 + i) * shared->output_buffer_stride + tile_coordinates.target_x_0),
                       ctx->h_target_tile.Get() + (i * tile_coordinates.target_width),
                       tile_coordinates.target_width * sizeof(float));
            }
        }
    }

    ctx->device_memory_arena.Reset();
}

void TerrainCorrection::TileLoop(SharedThreadData* shared, PerThreadData* ctx) {
    try {
        InitThreadContext(ctx, shared->max_tile_size, shared->use_pinned_memory);
        while (true) {
            {
                std::unique_lock l(shared->exception_mutex);
                if (shared->exception_ptr != nullptr) {
                    // another thread has failed, no point going further
                    break;
                }
            }
            TcTileIndexPair tile = {};
            if (!shared->tile_queue.PopFront(tile)) {
                break;
            }
            CalculateTile(tile, shared, ctx);
        }
    } catch (const std::exception& e) {
        std::unique_lock l(shared->exception_mutex);
        if (shared->exception_ptr == nullptr) {
            shared->exception_ptr = std::current_exception();
        } else {
            LOGE << e.what();
        }
    }
}

void TerrainCorrection::CreateSrgrCoefficientsOnDevice() {
    const auto srgr_coefficients_count = metadata_.srgr_coefficients.size();
    if (srgr_coefficients_count == 0) {
        return;
    }

    CHECK_CUDA_ERR(cudaMalloc(&d_srgr_coefficients_.array, srgr_coefficients_count * sizeof(SrgrCoefficientsDevice)));
    d_srgr_coefficients_.size = srgr_coefficients_count;
    for (size_t i{0}; i < srgr_coefficients_count; i++) {
        const auto& h_srgr = metadata_.srgr_coefficients.at(i);
        auto* d_srgr_entry = &d_srgr_coefficients_.array[i];
        const auto coefficient_count = h_srgr.coefficient.size();

        SrgrCoefficientsDevice h_tmp = {};
        h_tmp.ground_range_origin = h_srgr.ground_range_origin;
        h_tmp.time_mjd = h_srgr.time_mjd;
        h_tmp.coefficients.size = coefficient_count;

        CHECK_CUDA_ERR(cudaMalloc(&h_tmp.coefficients.array, coefficient_count * sizeof(double)));
        CHECK_CUDA_ERR(cudaMemcpy(h_tmp.coefficients.array, h_srgr.coefficient.data(),
                                  coefficient_count * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMemcpy(d_srgr_entry, &h_tmp, sizeof(SrgrCoefficientsDevice), cudaMemcpyHostToDevice));

        cuda_arrays_to_clean_.push_back(h_tmp.coefficients.array);
    }
    cuda_arrays_to_clean_.push_back(d_srgr_coefficients_.array);

#if TC_DEBUG_SRGR
    for (size_t i{0}; i < srgr_coefficients_count; i++) {
        SrgrCoefficientsDevice h_tmp = {};
        CHECK_CUDA_ERR(
            cudaMemcpy(&h_tmp, &d_srgr_coefficients_.array[i], sizeof(SrgrCoefficientsDevice), cudaMemcpyDeviceToHost));

        auto& host_prop = metadata_.srgr_coefficients.at(i);
        if (h_tmp.time_mjd != metadata_.srgr_coefficients.at(i).time_mjd) {
            std::cout << "MJD origin: " << host_prop.time_mjd << " on device:" << h_tmp.time_mjd << std::endl;
        }
        if (h_tmp.ground_range_origin != host_prop.ground_range_origin) {
            std::cout << "Ground range origin: " << host_prop.ground_range_origin
                      << " on device: " << h_tmp.ground_range_origin << std::endl;
        }

        std::vector<double> coeff(host_prop.coefficient.size());
        CHECK_CUDA_ERR(cudaMemcpy(coeff.data(), h_tmp.coefficients.array, host_prop.coefficient.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        for (size_t j{0}; j < host_prop.coefficient.size(); j++) {
            if (coeff.at(j) != host_prop.coefficient.at(j)) {
                std::cout << "Coeff origin: " << host_prop.coefficient.at(j) << " on device: " << coeff.at(j)
                          << std::endl;
            }
        }
    }
#endif
}
}  // namespace alus::terraincorrection
