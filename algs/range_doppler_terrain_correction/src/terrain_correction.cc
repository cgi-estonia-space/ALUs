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

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "crs_geocoding.h"
#include "cuda_util.hpp"
#include "dem.hpp"
#include "gdal_util.h"
#include "general_constants.h"
#include "product_old.h"
#include "raster_properties.hpp"
#include "shapes_util.h"
#include "srtm3_elevation_model_constants.h"
#include "tc_tile.h"
#include "terrain_correction_constants.h"
#include "terrain_correction_kernel.h"
#include "tie_point_geocoding.h"

namespace alus::terraincorrection {

void FillGetPositionMetadata(GetPositionMetadata& get_position_metadata,
                             cuda::KernelArray<snapengine::PosVector>& k_sensor_positions,
                             cuda::KernelArray<snapengine::PosVector>& k_sensor_velocities,
                             const ComputationMetadata& comp_metadata, double line_time_interval_in_days) {
    get_position_metadata.sensor_position = k_sensor_positions;
    get_position_metadata.sensor_velocity = k_sensor_velocities;
    get_position_metadata.orbit_state_vectors = comp_metadata.orbit_state_vectors;
    get_position_metadata.first_line_utc = comp_metadata.first_line_time_mjd;
    get_position_metadata.line_time_interval = line_time_interval_in_days;
    get_position_metadata.wavelength =
        snapengine::constants::lightSpeed / (comp_metadata.radar_frequency * snapengine::constants::oneMillion);
    get_position_metadata.range_spacing = comp_metadata.range_spacing;
    get_position_metadata.near_edge_slant_range = comp_metadata.slant_range_to_first_pixel;
}

void FillHostGetPositionMetadata(GetPositionMetadata& h_get_position_metadata,
                                 GetPositionMetadata& d_get_position_metadata,
                                 std::vector<snapengine::PosVector>& h_positions,
                                 std::vector<snapengine::PosVector>& h_velocities,
                                 std::vector<snapengine::OrbitStateVectorComputation>& h_orbit_state_vectors) {
    h_get_position_metadata = d_get_position_metadata;

    CHECK_CUDA_ERR(
        cudaMemcpy(h_orbit_state_vectors.data(), d_get_position_metadata.orbit_state_vectors.array,
                   d_get_position_metadata.orbit_state_vectors.size * sizeof(snapengine::OrbitStateVectorComputation),
                   cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_velocities.data(), d_get_position_metadata.sensor_velocity.array,
                              d_get_position_metadata.sensor_velocity.size * sizeof(snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_positions.data(), d_get_position_metadata.sensor_position.array,
                              d_get_position_metadata.sensor_position.size * sizeof(snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    h_get_position_metadata.orbit_state_vectors.array = h_orbit_state_vectors.data();
    h_get_position_metadata.sensor_position.array = h_positions.data();
    h_get_position_metadata.sensor_velocity.array = h_velocities.data();
}

TerrainCorrection::TerrainCorrection(Dataset<double> coh_ds, const RangeDopplerTerrainMetadata& metadata,
                                     const snapengine::tiepointgrid::TiePointGrid& lat_tie_point_grid,
                                     const snapengine::tiepointgrid::TiePointGrid& lon_tie_point_grid,
                                     const PointerHolder* srtm_3_tiles, size_t srtm_3_tiles_length,
                                     int selected_band_id)
    : coh_ds_{std::move(coh_ds)},
      metadata_{metadata},
      d_srtm_3_tiles_(srtm_3_tiles),
      d_srtm_3_tiles_length_(srtm_3_tiles_length),
      selected_band_id_(selected_band_id),
      lat_tie_point_grid_{lat_tie_point_grid},
      lon_tie_point_grid_{lon_tie_point_grid} {}

void TerrainCorrection::ExecuteTerrainCorrection(std::string_view output_file_name, size_t tile_width,
                                                 size_t tile_height) {
    coh_ds_.LoadRasterBand(1);
    auto const coh_ds_y_size{static_cast<size_t>(coh_ds_.GetYSize())};

    snapengine::geocoding::TiePointGeocoding source_geocoding(lat_tie_point_grid_, lon_tie_point_grid_);
    snapengine::old::Product target_product =
        CreateTargetProduct(&source_geocoding, output_file_name);
    auto const target_x_size{target_product.dataset_.GetXSize()};
    auto const target_y_size{target_product.dataset_.GetYSize()};

    int diff_lat = static_cast<int>(std::abs(target_product.geocoding_->GetPixelCoordinates(0, 0).lat -
                                             target_product.geocoding_->GetPixelCoordinates(0, target_y_size - 1).lat));

    // Populate GeoTransformParameters
    double target_geo_transform_array[6];
    target_product.dataset_.GetGdalDataset()->GetGeoTransform(target_geo_transform_array);
    GeoTransformParameters const target_geo_transform{GeoTransformConstruct::buildFromGDAL(target_geo_transform_array)};

    const auto line_time_interval_in_days{(metadata_.last_line_time->GetMjd() - metadata_.first_line_time->GetMjd()) /
                                          static_cast<double>(coh_ds_y_size - 1)};

    const auto pos_vector_items_length = coh_ds_y_size;
    const auto pos_vector_array_bytes_size = sizeof(snapengine::PosVector) * pos_vector_items_length;

    cuda::KernelArray<snapengine::PosVector> kernel_sensor_positions{nullptr, pos_vector_items_length};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_positions.array, pos_vector_array_bytes_size));
    CHECK_CUDA_ERR(cudaMemset(kernel_sensor_positions.array, 0, pos_vector_array_bytes_size));

    cuda::KernelArray<snapengine::PosVector> kernel_sensor_velocities{nullptr, pos_vector_items_length};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_velocities.array, pos_vector_array_bytes_size));
    CHECK_CUDA_ERR(cudaMemset(kernel_sensor_velocities.array, 0, pos_vector_array_bytes_size));

    const auto computation_metadata = CreateComputationMetadata();
    const auto start{std::chrono::steady_clock::now()};
    CalculateVelocitiesAndPositions(static_cast<int>(coh_ds_y_size), metadata_.first_line_time->GetMjd(), line_time_interval_in_days,
                                    computation_metadata.orbit_state_vectors, kernel_sensor_velocities,
                                    kernel_sensor_positions);
    const auto end{std::chrono::steady_clock::now()};
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Orbit state vectors computation took: " << duration << std::endl;

    GetPositionMetadata d_get_position_metadata{};
    FillGetPositionMetadata(d_get_position_metadata, kernel_sensor_positions, kernel_sensor_velocities,
                            computation_metadata, line_time_interval_in_days);

    GetPositionMetadata h_get_position_metadata{};
    std::vector<snapengine::OrbitStateVectorComputation> h_orbit_state_vectors(
        computation_metadata.orbit_state_vectors.size);
    std::vector<snapengine::PosVector> h_velocities(d_get_position_metadata.sensor_velocity.size);
    std::vector<snapengine::PosVector> h_positions(d_get_position_metadata.sensor_position.size);
    FillHostGetPositionMetadata(h_get_position_metadata, d_get_position_metadata, h_positions, h_velocities,
                                h_orbit_state_vectors);

    snapengine::resampling::Tile target_image{
        0, 0, static_cast<size_t>(target_x_size), static_cast<size_t>(target_y_size), false, false, nullptr};
    std::vector<TcTile> tiles =
        CalculateTiles(target_image, {0, 0, target_x_size, target_y_size}, tile_width, tile_height);

    auto const tile_loop_start{std::chrono::steady_clock::now()};

    boost::asio::thread_pool thread_pool;
    std::for_each(tiles.begin(), tiles.end(), [&](auto&& tile) {

        boost::asio::post(thread_pool, TileProcessor(tile, this, h_get_position_metadata, d_get_position_metadata,
            target_geo_transform, diff_lat, computation_metadata, target_product));
    });

    thread_pool.wait();

    printf("Tile processing finished\n");
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
std::vector<TcTile> TerrainCorrection::CalculateTiles(snapengine::resampling::Tile& base_image, Rectangle dest_bounds,
                                                      int tile_width, int tile_height) {
    std::vector<TcTile> output_tiles;
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
            TcTile tile{{0, 0, 0, 0, static_cast<double>(intersection.x), static_cast<double>(intersection.y),
                         static_cast<size_t>(intersection.width), static_cast<size_t>(intersection.height)},
                        {}};
            output_tiles.push_back(tile);
        }
    }

    return output_tiles;
}

snapengine::old::Product TerrainCorrection::CreateTargetProduct(
    const snapengine::geocoding::Geocoding* source_geocoding, const std::string_view output_filename) {
    const char* const output_format = "GTiff";

    double pixel_spacing_in_meter = std::trunc(metadata_.azimuth_spacing * 100 + 0.5) / 100.0;
    double pixel_spacing_in_degree = pixel_spacing_in_meter / SEMI_MAJOR_AXIS * RTOD;

    OGRSpatialReference target_crs;
    // EPSG:4326 is a WGS84 code
    target_crs.importFromEPSG(4326);

    std::vector<double> image_boundary = ComputeImageBoundary(source_geocoding, coh_ds_.GetXSize(), coh_ds_.GetYSize());

    double pixel_size_x = pixel_spacing_in_degree;
    double pixel_size_y = pixel_spacing_in_degree;

    // Calculates a rectangle coordinates (x0, y0, width, height) for new product based on source image diagonals
    std::vector<double> target_bounds{image_boundary[3] < image_boundary[2] ? image_boundary[3] : image_boundary[2],
                                      image_boundary[1] < image_boundary[0] ? image_boundary[1] : image_boundary[0],
                                      std::abs(image_boundary[3] - image_boundary[2]),
                                      std::abs(image_boundary[1] - image_boundary[0])};


    if (OGRErr error = target_crs.Validate() != OGRERR_NONE) {
        printf("ERROR: %d\n", error);  // TODO: implement some real error (SNAPGPU-163)
    }

    GDALDriver* output_driver;
    output_driver = GetGDALDriverManager()->GetDriverByName(output_format);

    if (output_driver == nullptr) {
        // TODO: throw exception (SNAPGPU-163)
    }

    char** output_driver_options = nullptr;
    GDALDataset* output_dataset;
    int a{static_cast<int>(std::floor((target_bounds[2]) / pixel_size_x))};
    int b{static_cast<int>(std::floor((target_bounds[3]) / pixel_size_y))};

    output_dataset = output_driver->Create(output_filename.data(), a, b, 1, GDT_Float64, output_driver_options);

    double output_geo_transform[6] = {target_bounds[0], pixel_size_x, 0, target_bounds[1] + target_bounds[3], 0,
                                      -pixel_size_y};

    // Apply shear and translate
    output_geo_transform[0] = output_geo_transform[0] + output_geo_transform[1] * (-0.5);
    output_geo_transform[3] = output_geo_transform[3] + output_geo_transform[5] * (-0.5);

    output_dataset->SetGeoTransform(output_geo_transform);
    char* projection_wkt;
    target_crs.exportToWkt(&projection_wkt);
    output_dataset->SetProjection(projection_wkt);
    auto band_info = metadata_.band_info.at(0);
    output_dataset->GetRasterBand(selected_band_id_)
        ->SetNoDataValue(band_info.no_data_value_used && band_info.no_data_value.has_value()
                             ? band_info.no_data_value.value()
                             : coh_ds_.GetGdalDataset()->GetRasterBand(selected_band_id_)->GetNoDataValue());
    CPLFree(projection_wkt);

    GeoTransformParameters geo_transform_parameters{output_geo_transform[0], output_geo_transform[3],
                                                    output_geo_transform[1], output_geo_transform[5]};

    snapengine::old::Product target_product{std::unique_ptr<snapengine::geocoding::CrsGeocoding>{}, *output_dataset,
                                            "TIFF"};
    target_product.geocoding_ = std::make_unique<snapengine::geocoding::CrsGeocoding>(geo_transform_parameters);

    return target_product;
}

ComputationMetadata TerrainCorrection::CreateComputationMetadata() {
    std::vector<snapengine::OrbitStateVectorComputation> comp_orbits;
    for (auto&& o : metadata_.orbit_state_vectors2) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }

    cuda::KernelArray<snapengine::OrbitStateVectorComputation> kernel_orbits{nullptr, comp_orbits.size()};
    CHECK_CUDA_ERR(
        cudaMalloc(&kernel_orbits.array, sizeof(snapengine::OrbitStateVectorComputation) * kernel_orbits.size));
    CHECK_CUDA_ERR(cudaMemcpy(kernel_orbits.array, comp_orbits.data(),
                              sizeof(snapengine::OrbitStateVectorComputation) * kernel_orbits.size,
                              cudaMemcpyHostToDevice));
    cuda_arrays_to_clean_.push_back(kernel_orbits.array);

    ComputationMetadata md{};
    md.orbit_state_vectors = kernel_orbits;
    md.first_line_time_mjd = metadata_.first_line_time->GetMjd();
    md.last_line_time_mjd = metadata_.last_line_time->GetMjd();
    md.first_near_lat = metadata_.first_near_lat;
    md.first_near_long = metadata_.first_near_long;
    md.first_far_lat = metadata_.first_far_lat;
    md.first_far_long = metadata_.first_far_long;
    md.last_near_lat = metadata_.last_near_lat;
    md.last_near_long = metadata_.last_near_long;
    md.last_far_lat = metadata_.last_far_lat;
    md.last_far_long = metadata_.last_far_long;
    md.radar_frequency = metadata_.radar_frequency;
    md.range_spacing = metadata_.range_spacing;
    md.line_time_interval = metadata_.line_time_interval;
    md.avg_scene_height = metadata_.avg_scene_height;
    md.slant_range_to_first_pixel = metadata_.slant_range_to_first_pixel;
    md.first_valid_pixel = metadata_.first_valid_pixel;
    md.last_valid_pixel = metadata_.last_valid_pixel;
    md.first_valid_line_time = metadata_.first_valid_line_time;
    md.last_valid_line_time = metadata_.last_valid_line_time;

    return md;
}

TerrainCorrection::~TerrainCorrection() { FreeCudaArrays(); }
void TerrainCorrection::FreeCudaArrays() {
    for (auto&& a : cuda_arrays_to_clean_) {
        cudaFree(a);
    }
    cuda_arrays_to_clean_.erase(cuda_arrays_to_clean_.begin(), cuda_arrays_to_clean_.end());
}

void SetTileSourceCoordinates(TcTileCoordinates& tile_coordinates, const Rectangle& source_rectangle) {
    tile_coordinates.source_height = source_rectangle.height;
    tile_coordinates.source_width = source_rectangle.width;
    tile_coordinates.source_x_0 = source_rectangle.x;
    tile_coordinates.source_y_0 = source_rectangle.y;
}

void TerrainCorrection::TileProcessor::Execute() {
    thrust::host_vector<double> target_tile_data(tile_.tc_tile_coordinates.target_width *
                                                 tile_.tc_tile_coordinates.target_height);
    tile_.target_tile_data_buffer = {target_tile_data.data(), target_tile_data.size()};

    snapengine::resampling::ResamplingRaster resampling_raster{0, 0, INVALID_SUB_SWATH_INDEX, {}, nullptr, false};

    auto& tile_coordinates = tile_.tc_tile_coordinates;

    snapengine::resampling::Tile source_tile{0, 0, 0, 0, false, false, nullptr};
    TerrainCorrectionKernelArgs args{
        static_cast<unsigned int>(terrain_correction_->coh_ds_.GetXSize()),
        static_cast<unsigned int>(terrain_correction_->coh_ds_.GetYSize()),
        snapengine::srtm3elevationmodel::NO_DATA_VALUE,  // TODO: value should originate from DEM dataset (SNAPGPU-193)
        target_product_.dataset_.GetNoDataValue(1),
        terrain_correction_->metadata_.avg_scene_height,
        target_geo_transform_,
        false,  // TODO: should come from arguments (SNAPGPU-193)
        diff_lat_,
        comp_metadata_,
        {},
        resampling_raster,
        {},
        {const_cast<PointerHolder*>(terrain_correction_->d_srtm_3_tiles_), terrain_correction_->d_srtm_3_tiles_length_},
        {}};

    args.metadata = comp_metadata_;
    args.get_position_metadata = d_get_position_metadata_;

    args.valid_pixels.size = tile_coordinates.target_width * tile_coordinates.target_height;
    CHECK_CUDA_ERR(cudaMalloc(&args.valid_pixels.array, args.valid_pixels.size * sizeof(bool)));

    args.resampling_raster.source_rectangle = GetSourceRectangle(tile_coordinates, args);

    auto& source_rectangle = args.resampling_raster.source_rectangle;
    SetTileSourceCoordinates(tile_coordinates, args.resampling_raster.source_rectangle);
    std::vector<double> source_tile_data(source_rectangle.width * source_rectangle.height);
    source_tile = {static_cast<int>(tile_coordinates.source_x_0),
                   static_cast<int>(tile_coordinates.source_y_0),
                   tile_coordinates.source_width,
                   tile_coordinates.source_height,
                   false,
                   false,
                   nullptr};
    args.resampling_raster.source_tile_i = &source_tile;

    {
        std::unique_lock lock(gdal_read_mutex_);
        CHECK_GDAL_ERROR(terrain_correction_->coh_ds_.GetGdalDataset()
                             ->GetRasterBand(terrain_correction_->selected_band_id_)
                             ->RasterIO(GF_Read, source_rectangle.x, source_rectangle.y, source_rectangle.width,
                                        source_rectangle.height, source_tile_data.data(), source_rectangle.width,
                                        source_rectangle.height, GDALDataType::GDT_Float64, 0, 0));
    }
    CHECK_CUDA_ERR(cudaMalloc(&source_tile.data_buffer, sizeof(double) * source_tile_data.size()));
    CHECK_CUDA_ERR(cudaMemcpy(source_tile.data_buffer, source_tile_data.data(),
                              sizeof(double) * source_tile_data.size(), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(LaunchTerrainCorrectionKernel(tile_, args));

    {
        std::unique_lock lock(gdal_write_mutex_);
        CHECK_GDAL_ERROR(target_product_.dataset_.GetGdalDataset()
                             ->GetRasterBand(terrain_correction_->selected_band_id_)
                             ->RasterIO(GF_Write, tile_coordinates.target_x_0, tile_coordinates.target_y_0,
                                        tile_coordinates.target_width, tile_coordinates.target_height,
                                        tile_.target_tile_data_buffer.array, tile_coordinates.target_width,
                                        tile_coordinates.target_height, GDT_Float64, 0, 0));
    }

    CHECK_CUDA_ERR(cudaFree(args.resampling_raster.source_tile_i->data_buffer));
    CHECK_CUDA_ERR(cudaFree(args.valid_pixels.array));
}
TerrainCorrection::TileProcessor::TileProcessor(TcTile& tile, TerrainCorrection* terrain_correction,
                                                GetPositionMetadata& h_get_position_metadata,
                                                GetPositionMetadata& d_get_position_metadata,
                                                GeoTransformParameters target_geo_transform, int diff_lat,
                                                const ComputationMetadata& comp_metadata,
                                                snapengine::old::Product& target_product)
    : tile_(tile),
      terrain_correction_(terrain_correction),
      host_get_position_metadata_(h_get_position_metadata),
      d_get_position_metadata_(d_get_position_metadata),
      target_geo_transform_(target_geo_transform),
      diff_lat_(diff_lat),
      comp_metadata_(comp_metadata),
      target_product_(target_product) {}
}  // namespace alus::terraincorrection
