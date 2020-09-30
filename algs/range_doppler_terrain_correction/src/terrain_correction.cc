#include "terrain_correction.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "product.h"
#include "shapes_util.cuh"
#include "tie_point_geocoding.cuh"
#include "product_data.h"
#include "crs_geocoding.cuh"
#include "cuda_util.hpp"
#include "dem.hpp"
#include "gdal_util.hpp"
#include "raster_properties.hpp"
#include "tc_tile.h"
#include "srtm3_elevation_model.h"

#include "local_dem.cuh"
#include "terrain_correction.cuh"

#define UNUSED(x) (void)(x)  // TODO: delete me
#define DEBUG printf("Reached here => %s:%d\n", __FILE__, __LINE__)

namespace alus {

TerrainCorrection::TerrainCorrection(alus::Dataset coh_ds, /*alus::Dataset metadata,*/ alus::Dataset dem)
    : coh_ds_{std::move(coh_ds)},
      //      metadata_ds_{std::move(metadata)},
      dem_ds_{std::move(dem)},
      coh_ds_elevations_(coh_ds_.GetDataBuffer().size()) {
    FillMetadata();
}

/**
 * Method for launching RangeDoppler Terrain Correction algorithm.
 *
 * @todo WIP: The method is going to be renamed and completed as part of SNAPGPU-119 and SNAPGUP-121 issues
 */
// void TerrainCorrection::DoWork() {
//    coh_ds_elevations_.resize(coh_ds_.getDataBuffer().size());
//    LocalDemCuda();
//}

void TerrainCorrection::LocalDemCpu() {
    auto const result = dem_ds_.GetLocalDemFor(coh_ds_, 0, 0, coh_ds_.GetXSize(), coh_ds_.GetYSize());

    const auto [min, max] = std::minmax_element(std::begin(result), std::end(result));
    std::cout << "Our area has lowest point at " << *min << " and highest at " << *max << std::endl;
}

void TerrainCorrection::LocalDemCuda(alus::Dataset* target_dataset) {
    auto const h_dem_array = dem_ds_.GetData();

    double* d_dem_array;
    double* d_product_array;

    coh_ds_elevations_.resize(target_dataset->GetXSize() * target_dataset->GetXSize());
    try {
        CHECK_CUDA_ERR(cudaMalloc(&d_dem_array, sizeof(double) * h_dem_array.size()));
        CHECK_CUDA_ERR(cudaMalloc(&d_product_array, sizeof(double) * coh_ds_elevations_.size()));
        CHECK_CUDA_ERR(
            cudaMemcpy(d_dem_array, h_dem_array.data(), sizeof(double) * h_dem_array.size(), cudaMemcpyHostToDevice));

        struct LocalDemKernelArgs kernel_args {};
        kernel_args.dem_tile_width = dem_ds_.GetColumnCount();
        kernel_args.dem_tile_height = dem_ds_.GetRowCount();
        kernel_args.target_width = target_dataset->GetXSize();
        kernel_args.target_height = target_dataset->GetYSize();
        dem_ds_.FillGeoTransform(kernel_args.dem_geo_transform.originLon,
                                 kernel_args.dem_geo_transform.originLat,
                                 kernel_args.dem_geo_transform.pixelSizeLon,
                                 kernel_args.dem_geo_transform.pixelSizeLat);
        target_dataset->FillGeoTransform(kernel_args.target_geo_transform.originLon,
                                         kernel_args.target_geo_transform.originLat,
                                         kernel_args.target_geo_transform.pixelSizeLon,
                                         kernel_args.target_geo_transform.pixelSizeLat);

        CHECK_CUDA_ERR(cudaGetLastError());

        RunElevationKernel(d_dem_array, d_product_array, kernel_args);

        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpy(coh_ds_elevations_.data(),
                                  d_product_array,
                                  sizeof(double) * coh_ds_elevations_.size(),
                                  cudaMemcpyDeviceToHost));
    } catch (alus::CudaErrorException const& cudaEx) {
        cudaFree(d_dem_array);
        cudaFree(d_product_array);

        throw;
    }
    cudaFree(d_dem_array);
    cudaFree(d_product_array);
}

std::vector<double> TerrainCorrection::ExecuteTerrainCorrection(const char* output_file_name,
                                                                size_t tile_width,
                                                                size_t tile_height) {
    auto const start{std::chrono::steady_clock::now()};

    dem_ds_.GetDataset()->LoadRasterBand(1);
    coh_ds_.LoadRasterBand(1);
    auto const coh_ds_x_size{static_cast<size_t>(coh_ds_.GetXSize())};
    auto const coh_ds_y_size{static_cast<size_t>(coh_ds_.GetYSize())};

    assert(lat_tie_points_.size() == 21 * 6);
    assert(lon_tie_points_.size() == 21 * 6);
    alus::snapengine::tiepointgrid::TiePointGrid lat_grid{0, 0, 1163, 300, 21, 6, lat_tie_points_.data()};
    alus::snapengine::tiepointgrid::TiePointGrid lon_grid{0, 0, 1163, 300, 21, 6, lon_tie_points_.data()};

    alus::snapengine::geocoding::TiePointGeocoding source_geocoding(lat_grid, lon_grid);
    alus::snapengine::geocoding::Geocoding* target_geocoding = nullptr;
    alus::snapengine::Product target_product =
        CreateTargetProduct(&source_geocoding, target_geocoding, coh_ds_x_size, coh_ds_y_size, output_file_name);
    auto const target_x_size{target_product.dataset_.GetXSize()};
    auto const target_y_size{target_product.dataset_.GetYSize()};

    int a = target_y_size;
    printf("%d\n", a);
    auto const target_preparation_end{std::chrono::steady_clock::now()};

    int diff_lat = static_cast<int>(std::abs(target_product.geocoding_->GetPixelCoordinates(0, 0).lat -
                                             target_product.geocoding_->GetPixelCoordinates(0, target_y_size - 1).lat));

    // Populate GeoTransformParameters
    double target_geo_transform_array[6], dem_geo_transform_array[6];
    target_product.dataset_.GetGdalDataset()->GetGeoTransform(target_geo_transform_array);
    dem_ds_.GetDataset()->GetGdalDataset()->GetGeoTransform(dem_geo_transform_array);
    alus::GeoTransformParameters const target_geo_transform{alus::GeoTransformConstruct::buildFromGDAL
            (target_geo_transform_array)};
    alus::GeoTransformParameters const dem_geo_transform{alus::GeoTransformConstruct::buildFromGDAL
            (dem_geo_transform_array)};

    alus::snapengine::geocoding::CrsGeocoding dem_geocoding(dem_geo_transform);

    // Get GetPosition Metadata
    alus::terraincorrection::RangeDopplerKernelMetadata kernel_metadata =
        alus::terraincorrection::GetKernelMetadata(this->metadata_);
    auto const line_time_interval_in_days{(this->metadata_.last_line_time.getMjd() - this->metadata_.first_line_time
                                                       .getMjd()) / (coh_ds_.GetYSize() - 1)};

    alus::cudautil::KernelArray<alus::snapengine::PosVector> kernel_sensor_positions{nullptr, coh_ds_y_size};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_positions.array, sizeof(alus::snapengine::PosVector) * coh_ds_y_size));

    alus::cudautil::KernelArray<alus::snapengine::PosVector> kernel_sensor_velocities{nullptr, coh_ds_y_size};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_velocities.array, sizeof(alus::snapengine::PosVector) * coh_ds_y_size));

    CalculateVelocitiesAndPositions(coh_ds_y_size,
                                    kernel_metadata.first_line_time.getMjd(),
                                    line_time_interval_in_days,
                                    kernel_metadata.orbit_state_vectors,
                                    kernel_sensor_velocities,
                                    kernel_sensor_positions);
    alus::terraincorrection::GetPositionMetadata get_position_metadata = GetGetPositionMetadata(
        coh_ds_.GetYSize(), kernel_metadata, &kernel_sensor_positions, &kernel_sensor_velocities);

    alus::terraincorrection::GetPositionMetadata h_get_position_metadata = get_position_metadata;
    std::vector<alus::snapengine::OrbitStateVector> h_orbit_state_vectors(kernel_metadata.orbit_state_vectors.size);
    std::vector<alus::snapengine::PosVector> h_velocities(get_position_metadata.sensor_velocity.size);
    std::vector<alus::snapengine::PosVector> h_positions(get_position_metadata.sensor_position.size);
    CHECK_CUDA_ERR(
        cudaMemcpy(h_orbit_state_vectors.data(),
                   get_position_metadata.orbit_state_vector.array,
                   get_position_metadata.orbit_state_vector.size * sizeof(alus::snapengine::OrbitStateVector),
                   cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_velocities.data(),
                              get_position_metadata.sensor_velocity.array,
                              get_position_metadata.sensor_velocity.size * sizeof(alus::snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_positions.data(),
                              get_position_metadata.sensor_position.array,
                              get_position_metadata.sensor_position.size * sizeof(alus::snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    h_get_position_metadata.orbit_state_vector.array = h_orbit_state_vectors.data();
    h_get_position_metadata.sensor_position.array = h_positions.data();
    h_get_position_metadata.sensor_velocity.array = h_velocities.data();

    alus::snapengine::resampling::Tile base_image{0,
                                                  0,
                                                  target_x_size,
                                                  target_y_size,
                                                  false,
                                                  false,
                                                  nullptr};
    std::vector<alus::TcTile> tiles =
        CalculateTiles(base_image,
                       {0, 0, target_x_size, target_y_size},
                       tile_width,
                       tile_height);
    uint32_t tile_spent_time{0};

    auto const tile_loop_start{std::chrono::steady_clock::now()};
    int i = 0;
    int j = 0;
    for (auto tile : tiles) {
        FillDemCoordinates(tile, target_geocoding, &dem_geocoding);

        std::vector<double> source_tile_dem_data(tile.tc_tile_coordinates.dem_width * tile.tc_tile_coordinates.dem_height, 10.0);

        tile.dem_tile_data_buffer = alus::cudautil::GetKernelArray(source_tile_dem_data);

        // Prepares target tile values
        thrust::host_vector<double> target_tile_data(tile.tc_tile_coordinates.target_width * tile.tc_tile_coordinates.target_height);
        tile.target_tile_data_buffer = {target_tile_data.data(), target_tile_data.size()};

        std::vector<double> elevation_tile_data(tile.target_tile_data_buffer.size);
        tile.elevation_tile_data_buffer = alus::cudautil::GetKernelArray(elevation_tile_data);
        DemCuda(tile, dem_ds_.GetDataset()->GetNoDataValue(), dem_geo_transform, target_geo_transform);

        // Prepare TileData
        alus::snapengine::resampling::ResamplingRaster resampling_raster{0, 0, -1, 0, 0, nullptr, nullptr};
        tile.tile_data.resampling_raster = &resampling_raster;

        alus::Rectangle source_rectangle{};
        bool is_source_rectangle_calculated = GetSourceRectangle(tile,
                                                                 target_geo_transform,
                                                                 dem_ds_.GetDataset()->GetNoDataValue(),
                                                                 coh_ds_.GetXSize(),
                                                                 coh_ds_.GetYSize(),
                                                                 h_get_position_metadata,
                                                                 source_rectangle);
        auto tile_coordinates = tile.tc_tile_coordinates;

        std::vector<double> source_tile_data(tile.tc_tile_coordinates.source_width * tile.tc_tile_coordinates.source_height);
        tile.source_tile_data_buffer = alus::cudautil::GetKernelArray(source_tile_data);
        tile.dem_tile_data_buffer = alus::cudautil::GetKernelArray(source_tile_dem_data);

        TerrainCorrectionKernelArgs args{static_cast<unsigned int>(coh_ds_.GetXSize()),
                                         static_cast<unsigned int>(coh_ds_.GetYSize()),
                                         dem_ds_.GetDataset()->GetNoDataValue(),
                                         dem_geo_transform,
                                         target_geo_transform,
                                         nullptr,
                                         nullptr,
                                         diff_lat,
                                         {},
                                         {},
                                         {}};

        alus::snapengine::resampling::Tile source_tile {0,0,0,0, false, false, nullptr};
        if (is_source_rectangle_calculated) {
            CHECK_GDAL_ERROR(coh_ds_.GetGdalDataset()->GetRasterBand(1)->RasterIO(GF_Read,
                                                                                  tile_coordinates.source_x_0,
                                                                                  tile_coordinates.source_y_0,
                                                                                  tile_coordinates.source_width,
                                                                                  tile_coordinates.source_height,
                                                                                  source_tile_data.data(),
                                                                                  tile_coordinates.source_width,
                                                                                  tile_coordinates.source_height,
                                                                                  GDALDataType::GDT_Float64,
                                                                                  0,
                                                                                  0));
            j++;
            source_tile = {
                static_cast<int>(tile_coordinates.source_x_0),
                static_cast<int>(tile_coordinates.source_y_0),
                tile_coordinates.source_width,
                tile_coordinates.source_height,
                false,
                false,
                source_tile_data.data()
            };
            tile.tile_data.resampling_raster->source_tile_i = &source_tile;
            tile.tile_data.resampling_raster->source_rectangle = &source_rectangle;
            tile.tile_data.source_tile = &source_tile;
        }
        std::vector<double> corrected_data(tile_coordinates.source_width * tile_coordinates.source_height);
        args.metadata = kernel_metadata;
        args.get_position_metadata = get_position_metadata;
        auto const main_part_start{std::chrono::steady_clock::now()};
        RunTerrainCorrectionKernel(tile, args);  // TODO: add target no_data_value
        auto const main_part_end{std::chrono::steady_clock::now()};

        tile_spent_time +=
            std::chrono::duration_cast<std::chrono::milliseconds>(main_part_end - main_part_start).count();

        CHECK_GDAL_ERROR(
            target_product.dataset_.GetGdalDataset()->GetRasterBand(1)->RasterIO(GF_Write,
                                                                                 tile_coordinates.target_x_0,
                                                                                 tile_coordinates.target_y_0,
                                                                                 tile_coordinates.target_width,
                                                                                 tile_coordinates.target_height,
                                                                                 tile.target_tile_data_buffer.array,
                                                                                 tile_coordinates.target_width,
                                                                                 tile_coordinates.target_height,
                                                                                 GDT_Float64,
                                                                                 0,
                                                                                 0));
    }
    printf("Proportion: %f\n", static_cast<double>(j) / i);
    std::vector<double> corrected_data(coh_ds_.GetDataBuffer().size());
    //    delete target_geocoding;  // TODO: implement destructors

    auto const stop{std::chrono::steady_clock::now()};

    printf(
        "Overall running time: %lu\n"
        "\tData preparation: %lu\n"
        "\tTile loop duration: %lu\n"
        "\tMain calculations' duration for all tiles: %d\n"
        "\tTotal number of tiles: %lu\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(target_preparation_end - start).count(),
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - tile_loop_start).count(),
        tile_spent_time,
        tiles.size());

    return corrected_data;
}

std::vector<double> TerrainCorrection::ComputeImageBoundary(const alus::snapengine::geocoding::Geocoding* geocoding,
                                                            int source_width,
                                                            int source_height) {
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

std::vector<alus::TcTile> TerrainCorrection::CalculateTiles(snapengine::resampling::Tile& base_image,
                                                             alus::Rectangle dest_bounds,
                                                             int tile_width,
                                                             int tile_height) {
    std::vector<alus::TcTile> output_tiles;
    int x_count = base_image.width / tile_width + 1;
    int y_count = base_image.height / tile_height + 1;

    for (int x_index = 0; x_index < x_count; x_index++) {
        for (int y_index = 0; y_index < y_count; y_index++) {
            alus::Rectangle rectangle{
                base_image.x_0 + x_index * tile_width, base_image.y_0 + y_index * tile_height, tile_width, tile_height};
            if (rectangle.x > dest_bounds.width || rectangle.y > dest_bounds.height) {
                continue;
            }
            alus::Rectangle intersection = alus::shapeutils::GetIntersection(rectangle, dest_bounds);
            if (intersection.width == 0 || intersection.height == 0) {
                continue;
            }
            alus::TcTile tile{{0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                static_cast<double>(intersection.x),
                                static_cast<double>(intersection.y),
                                intersection.width,
                                intersection.height},
                               {},
                               {},
                               {},
                               {},
                               {nullptr, nullptr, nullptr}};
            output_tiles.push_back(tile);
        }
    }

    return output_tiles;
}

void TerrainCorrection::FillDemCoordinates(alus::TcTile& tile,
                                           alus::snapengine::geocoding::Geocoding* target_geocoding,
                                           alus::snapengine::geocoding::Geocoding* dem_geocoding) {
    auto tile_coordinates = tile.tc_tile_coordinates;

    auto target_start_coordinates = target_geocoding->GetPixelCoordinates(tile_coordinates.target_x_0, tile_coordinates.target_y_0);
    auto target_end_coordinates = target_geocoding->GetPixelCoordinates(tile_coordinates.target_x_0 + tile_coordinates.target_width,
                                                                        tile_coordinates.target_y_0 + tile_coordinates.target_height);

    alus::Coordinates dem_start_coordinates{std::min(target_start_coordinates.lon, target_end_coordinates.lon),
                                            std::max(target_start_coordinates.lat, target_end_coordinates.lat)};

    alus::Coordinates dem_end_coordinates{std::max(target_start_coordinates.lon, target_end_coordinates.lon),
                                          std::min(target_start_coordinates.lat, target_end_coordinates.lat)};

    auto dem_start_indices = dem_geocoding->GetPixelPosition(dem_start_coordinates);
    auto dem_end_indices = dem_geocoding->GetPixelPosition(dem_end_coordinates);
    tile.tc_tile_coordinates.dem_width = std::ceil(dem_end_indices.x - dem_start_indices.x);
    tile.tc_tile_coordinates.dem_height = std::ceil(dem_end_indices.y - dem_start_indices.y);

    tile.tc_tile_coordinates.dem_x_0 = dem_start_indices.x;
    tile.tc_tile_coordinates.dem_y_0 = dem_start_indices.y;
}

// Only the first three vectors are being copied as later the list of vectors should be populated from .dim file
std::vector<alus::snapengine::OrbitStateVector> GetOrbitStateVectorsStub() {
    std::vector<alus::snapengine::OrbitStateVector> vectors;
    vectors.emplace_back(alus::snapengine::old::Utc("15-JUL-2019 16:04:33.800577"),
                         3727040.7077331543,
                         1103842.85256958,
                         5902738.6076049805,
                         -5180.233733266592,
                         -3857.165526404977,
                         3982.913521885872);
    vectors.emplace_back(alus::snapengine::old::Utc("15-JUL-2019 16:04:34.800577"),
                         3721858.106201172,
                         1099985.447479248,
                         5906718.189788818,
                         -5184.967764496803,
                         -3857.643955528736,
                         3976.251023232937);
    vectors.emplace_back(alus::snapengine::old::Utc("15-JUL-2019 16:04:35.800577"),
                         3716670.7736206055,
                         1096127.5664367676,
                         5910691.107452393,
                         -5189.69604575634,
                         -3858.1173707023263,
                         3969.5840579867363);
    return vectors;
}

void TerrainCorrection::FillMetadata() {
    metadata_.product = "S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6";
    metadata_.product_type = alus::metadata::ProductType::SLC;
    metadata_.sph_descriptor = "Sentinel-1 IW Level-1 SLC Product";
    metadata_.mission = "SENTINEL-1A";
    metadata_.acquisition_mode = alus::metadata::AcquisitionMode::IW;
    metadata_.antenna_pointing = alus::metadata::AntennaDirection::RIGHT;
    metadata_.beams = "";
    metadata_.swath = alus::metadata::Swath::IW1;
    metadata_.proc_time = alus::snapengine::old::Utc("15-JUL-2019 18:36:47.607732");
    metadata_.processing_system_identifier = "ESA Sentinel-1 IPF 003.10";
    metadata_.orbit_cycle = 175;
    metadata_.rel_orbit = 58;
    metadata_.abs_orbit = 28130;
    metadata_.state_vector_time = alus::snapengine::old::Utc("15-JUL-2019 16:03:37.964246");
    metadata_.vector_source = "";
    metadata_.incidence_near = 29.786540985107422;
    metadata_.incidence_far = 36.43135039339361;
    metadata_.slice_num = 24;
    metadata_.first_line_time = alus::snapengine::old::Utc("15-JUL-2019 16:04:43.800577");
    metadata_.last_line_time = alus::snapengine::old::Utc("15-JUL-2019 16:04:46.885967");
    metadata_.first_near_lat = 58.21324157714844;
    metadata_.first_near_long = 21.98597526550293;
    metadata_.first_far_lat = 58.392906188964844;
    metadata_.first_far_long = 23.64056968688965;
    metadata_.last_near_lat = 58.3963737487793;
    metadata_.last_near_long = 21.90845489501953;
    metadata_.last_far_lat = 58.57649612426758;
    metadata_.last_far_long = 23.571735382080078;
    metadata_.pass = alus::metadata::Pass::ASCENDING;
    metadata_.sample_type = alus::metadata::SampleType::COMPLEX;
    metadata_.mds1_tx_rx_polar = alus::metadata::Polarisation::VH;
    metadata_.mds2_tx_rx_polar = alus::metadata::Polarisation::VH;
    metadata_.azimuth_looks = 1.0;
    metadata_.range_looks = 1.0;
    metadata_.range_spacing = 2.329562;
    metadata_.azimuth_spacing = 13.91157;
    metadata_.pulse_repetition_frequency = 1717.128973878037;
    metadata_.radar_frequency = 5405.000454334349;
    metadata_.line_time_interval = 0.002055556299999998;
    metadata_.total_size = 158871714;
    metadata_.num_output_lines = 1500;
    metadata_.num_samples_per_line = 23278;
    metadata_.subset_offset_x = 0;
    metadata_.subset_offset_y = 3004;
    metadata_.srgr_flag = false;
    metadata_.avg_scene_height = 23.65084248584435;
    metadata_.map_projection = "";
    metadata_.is_terrain_corrected = false;
    metadata_.dem = "";
    metadata_.geo_ref_system = "";
    metadata_.lat_pixel_res = 99999.0;
    metadata_.long_pixel_res = 99999.0;
    metadata_.slant_range_to_first_pixel = 799303.6132771898;
    metadata_.ant_elev_corr_flag = false;
    metadata_.range_spread_comp_flag = false;
    metadata_.replica_power_corr_flag = false;
    metadata_.abs_calibration_flag = false;
    metadata_.calibration_factor = 99999.0;
    metadata_.chirp_power = 99999.0;
    metadata_.inc_angle_comp_flag = false;
    metadata_.ref_inc_angle = 99999.0;
    metadata_.ref_slant_range = 99999.0;
    metadata_.ref_slant_range_exp = 99999.0;
    metadata_.rescaling_factor = 99999.0;
    metadata_.bistatic_correction_applied = true;
    metadata_.range_sampling_rate = 64.34523812571427;
    metadata_.range_bandwidth = 56.5;
    metadata_.azimuth_bandwidth = 327.0;
    metadata_.multilook_flag = false;
    metadata_.coregistered_stack = false;
    metadata_.external_calibration_file = "";
    metadata_.orbit_state_vector_file =
        "Sentinel Precise S1A_OPER_AUX_POEORB_OPOD_20190804T120708_V20190714T225942_20190716T005942.EOF.zip";
    metadata_.metadata_version = "6.0";
    metadata_.centre_lat = 58.86549898479201;
    metadata_.centre_lon = 24.04291372551365;
    metadata_.centre_heading = 349.45641790421365;
    metadata_.centre_heading_2 = 169.4528423214929;
    metadata_.first_valid_pixel = 1919;
    metadata_.last_valid_pixel = 22265;
    metadata_.slr_time_to_first_valid_pixel = 0.0026811016131539564;
    metadata_.slr_time_to_last_valid_pixel = 0.0028392018828145094;
    metadata_.first_valid_line_time = 6.165218838437437E8;
    metadata_.last_valid_line_time = 6.165218868469114E8;
    metadata_.orbit_state_vectors = GetOrbitStateVectorsStub();

    lat_tie_points_ = {
        58.213176727294920, 58.223548889160156, 58.233741760253906, 58.243762969970700, 58.253620147705080,
        58.263313293457030, 58.272853851318360, 58.282241821289060, 58.291488647460940, 58.300598144531250,
        58.309570312500000, 58.318412780761720, 58.327129364013670, 58.335720062255860, 58.344196319580080,
        58.352554321289060, 58.360801696777344, 58.368938446044920, 58.376968383789060, 58.384895324707030,
        58.392719268798830, 58.249801635742190, 58.260181427001950, 58.270381927490234, 58.280406951904300,
        58.290267944335940, 58.299968719482420, 58.309513092041016, 58.318908691406250, 58.328159332275390,
        58.337272644042970, 58.346248626708984, 58.355094909667970, 58.363815307617190, 58.372413635253906,
        58.380889892578125, 58.389255523681640, 58.397502899169920, 58.405643463134766, 58.413677215576170,
        58.421607971191406, 58.429439544677734, 58.286430358886720, 58.296813964843750, 58.307022094726560,
        58.317050933837890, 58.326919555664060, 58.336624145507810, 58.346172332763670, 58.355571746826170,
        58.364830017089844, 58.373947143554690, 58.382926940917970, 58.391780853271484, 58.400505065917970,
        58.409103393554690, 58.417587280273440, 58.425952911376950, 58.434207916259766, 58.442352294921875,
        58.450389862060550, 58.458324432373050, 58.466156005859375, 58.323055267333984, 58.333450317382810,
        58.343658447265625, 58.353698730468750, 58.363567352294920, 58.373279571533200, 58.382831573486330,
        58.392238616943360, 58.401500701904300, 58.410621643066406, 58.419605255126950, 58.428462982177734,
        58.437191009521484, 58.445796966552734, 58.454284667968750, 58.462654113769530, 58.470912933349610,
        58.479061126708984, 58.487102508544920, 58.495040893554690, 58.502876281738280, 58.359683990478516,
        58.370082855224610, 58.380298614501950, 58.390342712402344, 58.400218963623050, 58.409931182861330,
        58.419494628906250, 58.428901672363280, 58.438167572021484, 58.447296142578125, 58.456287384033200,
        58.465145111083984, 58.473880767822266, 58.482490539550780, 58.490978240966800, 58.499355316162110,
        58.507614135742190, 58.515766143798830, 58.523811340332030, 58.531753540039060, 58.539592742919920,
        58.396308898925780, 58.406711578369140, 58.416934967041016, 58.426982879638670, 58.436862945556640,
        58.446586608886720, 58.456150054931640, 58.465564727783200, 58.474834442138670, 58.483966827392580,
        58.492961883544920, 58.501827239990234, 58.510562896728516, 58.519180297851560, 58.527671813964844,
        58.536052703857420, 58.544315338134766, 58.552471160888670, 58.560520172119140, 58.568466186523440,
        58.576309204101560};

    lon_tie_points_ = {
        21.985961914062500, 22.075984954833984, 22.165037155151367, 22.253160476684570, 22.340394973754883,
        22.426778793334960, 22.512342453002930, 22.597120285034180, 22.681142807006836, 22.764436721801758,
        22.847030639648438, 22.928949356079100, 23.010215759277344, 23.090854644775390, 23.170883178710938,
        23.250326156616210, 23.329200744628906, 23.407526016235350, 23.485319137573242, 23.562595367431640,
        23.639371871948242, 21.970470428466797, 22.060586929321290, 22.149732589721680, 22.237947463989258,
        22.325275421142578, 22.411746978759766, 22.497400283813477, 22.582267761230470, 22.666378021240234,
        22.749759674072266, 22.832441329956055, 22.914443969726562, 22.995796203613280, 23.076519012451172,
        23.156633377075195, 23.236160278320312, 23.315116882324220, 23.393524169921875, 23.471397399902344,
        23.548755645751953, 23.625612258911133, 21.954977035522460, 22.045188903808594, 22.134426116943360,
        22.222734451293945, 22.310153961181640, 22.396717071533203, 22.482460021972656, 22.567415237426758,
        22.651613235473633, 22.735082626342773, 22.817850112915040, 22.899940490722656, 22.981376647949220,
        23.062185287475586, 23.142381668090820, 23.221992492675780, 23.301033020019530, 23.379522323608400,
        23.457477569580078, 23.534915924072266, 23.611854553222656, 21.939485549926758, 22.029788970947266,
        22.119121551513672, 22.207523345947266, 22.295032501220703, 22.381685256958008, 22.467517852783203,
        22.552562713623047, 22.636848449707030, 22.720405578613280, 22.803258895874023, 22.885435104370117,
        22.966957092285156, 23.047849655151367, 23.128131866455078, 23.207824707031250, 23.286947250366210,
        23.365518569946290, 23.443555831909180, 23.521076202392578, 23.598094940185547, 21.923992156982422,
        22.014390945434570, 22.103816986083984, 22.192310333251953, 22.279911041259766, 22.366655349731445,
        22.452577590942383, 22.537710189819336, 22.622085571289062, 22.705728530883790, 22.788669586181640,
        22.870931625366210, 22.952537536621094, 23.033514022827150, 23.113880157470703, 23.193656921386720,
        23.272863388061523, 23.351516723632812, 23.429636001586914, 23.507238388061523, 23.584337234497070,
        21.908441543579100, 21.998935699462890, 22.088455200195312, 22.177040100097656, 22.264732360839844,
        22.351568222045900, 22.437580108642578, 22.522804260253906, 22.607265472412110, 22.690998077392578,
        22.774024963378906, 22.856372833251953, 22.938066482543945, 23.019128799438477, 23.099578857421875,
        23.179439544677734, 23.258728027343750, 23.337465286254883, 23.415666580200195, 23.493349075317383,
        23.570529937744140};

}

// TODO: add target filename argument
alus::snapengine::Product TerrainCorrection::CreateTargetProduct(
    const alus::snapengine::geocoding::Geocoding* source_geocoding,
    snapengine::geocoding::Geocoding*& target_geocoding,
    int source_width,
    int source_height,
    const char* output_file_name) const {
    const char* const OUTPUT_FORMAT = "GTiff";

    // TODO: move to some constants header if used in other functions
    const double SEMI_MAJOR_AXIS{6378137.0};
    const double RTOD{57.29577951308232};

    double pixel_spacing_in_meter = static_cast<int>(metadata_.azimuth_spacing * 100 + 0.5) / 100.0;
    double pixel_spacing_in_degree = pixel_spacing_in_meter / SEMI_MAJOR_AXIS * RTOD;

    OGRSpatialReference target_crs;
    // EPSG:4326 is a WGS84 code
    target_crs.importFromEPSG(4326);

    std::vector<double> image_boundary = ComputeImageBoundary(source_geocoding, source_width, source_height);

    double pixel_size_x = pixel_spacing_in_degree;
    double pixel_size_y = pixel_spacing_in_degree;

    // Calculates a rectangle coordinates (x0, y0, width, height) for new product based on source image diagonals
    std::vector<double> target_bounds{image_boundary[3] < image_boundary[2] ? image_boundary[3] : image_boundary[2],
                                      image_boundary[1] < image_boundary[0] ? image_boundary[1] : image_boundary[0],
                                      std::abs(image_boundary[3] - image_boundary[2]),
                                      std::abs(image_boundary[1] - image_boundary[0])};

    OGRErr error = target_crs.Validate();
    if (target_crs.Validate() != OGRERR_NONE) {
        printf("ERROR: %d\n", error);  // TODO: implement some real error
    }

    GDALDriver* output_driver;
    output_driver = GetGDALDriverManager()->GetDriverByName(OUTPUT_FORMAT);

    if (output_driver == nullptr) {
        // TODO: throw exception
    }

    char** output_driver_options = nullptr;
    GDALDataset* output_dataset;
    int a{static_cast<int>(std::floor((target_bounds[2]) / pixel_size_x))};
    int b{static_cast<int>(std::floor((target_bounds[3]) / pixel_size_y))};

    output_dataset = output_driver->Create(output_file_name, a, b, 1, GDT_Float64, output_driver_options);

    double output_geo_transform[6] = {
        target_bounds[0], pixel_size_x, 0, target_bounds[1] + target_bounds[3], 0, -pixel_size_y};

    // Apply shear and translate
    output_geo_transform[0] = output_geo_transform[0] + output_geo_transform[1] * (-0.5);
    output_geo_transform[3] = output_geo_transform[3] + output_geo_transform[5] * (-0.5);

    output_dataset->SetGeoTransform(output_geo_transform);
    char* projection_wkt;
    target_crs.exportToWkt(&projection_wkt);
    output_dataset->SetProjection(projection_wkt);
    CPLFree(projection_wkt);

    target_geocoding = new alus::snapengine::geocoding::CrsGeocoding(
        {output_geo_transform[0], output_geo_transform[3], output_geo_transform[1], output_geo_transform[5]});
    alus::snapengine::Product target_product{
        target_geocoding, *output_dataset, "TIFF"};  // TODO: add destructor product class

    return target_product;
}

TerrainCorrection::~TerrainCorrection() { cudaFree(d_local_dem_); }

}  // namespace alus
