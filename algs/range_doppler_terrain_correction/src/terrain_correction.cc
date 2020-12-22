#include "terrain_correction.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

#include "crs_geocoding.cuh"
#include "cuda_util.hpp"
#include "dem.hpp"
#include "gdal_util.hpp"
#include "product_old.h"
#include "raster_properties.hpp"
#include "sar_utils.h"
#include "shapes_util.cuh"
#include "tc_tile.h"
#include "terrain_correction.cuh"
#include "tie_point_geocoding.cuh"

#define UNUSED(x) (void)(x)  // TODO: delete me
#define DEBUG printf("Reached here => %s:%d\n", __FILE__, __LINE__)

namespace alus::terraincorrection {

//TerrainCorrection::TerrainCorrection(alus::Dataset coh_ds, /*alus::Dataset metadata,*/ alus::Dataset dem)
//    : coh_ds_{std::move(coh_ds)},
//      //      metadata_ds_{std::move(metadata)},
//      dem_ds_{std::move(dem)},
//      coh_ds_elevations_(coh_ds_.GetDataBuffer().size()) {
//}

TerrainCorrection::TerrainCorrection(Dataset coh_ds,
                                     RangeDopplerTerrainMetadata metadata,
                                     const Metadata::TiePoints& lat_tie_points,
                                     const Metadata::TiePoints& lon_tie_points)
    : coh_ds_{std::move(coh_ds)},
      metadata_{std::move(metadata)},
      lat_tie_points_{lat_tie_points},
      lon_tie_points_{lon_tie_points} {}

void TerrainCorrection::ExecuteTerrainCorrection(const std::string& output_file_name, size_t tile_width, size_t tile_height) {
    auto const start{std::chrono::steady_clock::now()};

    //dem_ds_.GetDataset()->LoadRasterBand(1);
    coh_ds_.LoadRasterBand(1);
    auto const coh_ds_x_size{static_cast<size_t>(coh_ds_.GetXSize())};
    auto const coh_ds_y_size{static_cast<size_t>(coh_ds_.GetYSize())};

    assert(lat_tie_points_.values.size() == lat_tie_points_.grid_width * lat_tie_points_.grid_height);
    assert(lon_tie_points_.values.size() == lon_tie_points_.grid_width * lon_tie_points_.grid_height);
    const alus::snapengine::tiepointgrid::TiePointGrid lat_grid{0,
                                                                0,
                                                                1163,
                                                                300,
                                                                lat_tie_points_.grid_width,
                                                                lat_tie_points_.grid_height,
                                                                const_cast<float*>(lat_tie_points_.values.data())};
    const alus::snapengine::tiepointgrid::TiePointGrid lon_grid{0,
                                                                0,
                                                                1163,
                                                                300,
                                                                lon_tie_points_.grid_width,
                                                                lon_tie_points_.grid_height,
                                                                const_cast<float*>(lon_tie_points_.values.data())};

    alus::snapengine::geocoding::TiePointGeocoding source_geocoding(lat_grid, lon_grid);
    alus::snapengine::geocoding::Geocoding* target_geocoding = nullptr;
    alus::snapengine::old::Product target_product = CreateTargetProduct(
        &source_geocoding, target_geocoding, coh_ds_x_size, coh_ds_y_size, metadata_.azimuth_spacing, output_file_name);
    auto const target_x_size{target_product.dataset_.GetXSize()};
    auto const target_y_size{target_product.dataset_.GetYSize()};

    int a = target_y_size;
    printf("%d\n", a);
    auto const target_preparation_end{std::chrono::steady_clock::now()};

    int diff_lat = static_cast<int>(std::abs(target_product.geocoding_->GetPixelCoordinates(0, 0).lat -
                                             target_product.geocoding_->GetPixelCoordinates(0, target_y_size - 1).lat));

    // Populate GeoTransformParameters
    double target_geo_transform_array[6];
    //, dem_geo_transform_array[6];
    target_product.dataset_.GetGdalDataset()->GetGeoTransform(target_geo_transform_array);
    //dem_ds_.GetDataset()->GetGdalDataset()->GetGeoTransform(dem_geo_transform_array);
    alus::GeoTransformParameters const target_geo_transform{alus::GeoTransformConstruct::buildFromGDAL
            (target_geo_transform_array)};
    //alus::GeoTransformParameters const dem_geo_transform{alus::GeoTransformConstruct::buildFromGDAL
    //        (dem_geo_transform_array)};

    //alus::snapengine::geocoding::CrsGeocoding dem_geocoding(dem_geo_transform);

    const auto line_time_interval_in_days{(this->metadata_.last_line_time.GetMjd() - this->metadata_.first_line_time
                                                       .GetMjd()) / (coh_ds_.GetYSize() - 1)};

    const auto pos_vector_items_length = coh_ds_y_size;
    const auto pos_vector_array_bytes_size = sizeof(alus::snapengine::PosVector) * pos_vector_items_length;

    alus::cuda::KernelArray<alus::snapengine::PosVector> kernel_sensor_positions{nullptr, pos_vector_items_length};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_positions.array, pos_vector_array_bytes_size));
    CHECK_CUDA_ERR(cudaMemset(kernel_sensor_positions.array, 0, pos_vector_array_bytes_size));

    alus::cuda::KernelArray<alus::snapengine::PosVector> kernel_sensor_velocities{nullptr, pos_vector_items_length};
    CHECK_CUDA_ERR(cudaMalloc(&kernel_sensor_velocities.array, pos_vector_array_bytes_size));
    CHECK_CUDA_ERR(cudaMemset(kernel_sensor_velocities.array, 0, pos_vector_array_bytes_size));

    const auto computation_metadata = CreateComputationMetadata();
    CalculateVelocitiesAndPositions(coh_ds_y_size,
                                    metadata_.first_line_time.GetMjd(),
                                    line_time_interval_in_days,
                                    computation_metadata.orbit_state_vectors,
                                    kernel_sensor_velocities,
                                    kernel_sensor_positions);

    alus::terraincorrection::GetPositionMetadata get_position_metadata{};
    get_position_metadata.sensor_position = kernel_sensor_positions;
    get_position_metadata.sensor_velocity = kernel_sensor_velocities;
    get_position_metadata.orbit_state_vector = computation_metadata.orbit_state_vectors;
    get_position_metadata.first_line_utc = metadata_.first_line_time.GetMjd();
    get_position_metadata.line_time_interval = line_time_interval_in_days;
    get_position_metadata.wavelength = snapengine::constants::lightSpeed / (metadata_.radar_frequency *
                                                                            snapengine::constants::oneMillion);
    get_position_metadata.range_spacing = metadata_.range_spacing;
    get_position_metadata.near_edge_slant_range = metadata_.slant_range_to_first_pixel;

    alus::terraincorrection::GetPositionMetadata h_get_position_metadata = get_position_metadata;
    std::vector<alus::snapengine::OrbitStateVectorComputation> h_orbit_state_vectors(computation_metadata.orbit_state_vectors.size);
    std::vector<alus::snapengine::PosVector> h_velocities(get_position_metadata.sensor_velocity.size);
    std::vector<alus::snapengine::PosVector> h_positions(get_position_metadata.sensor_position.size);
    CHECK_CUDA_ERR(
        cudaMemcpy(h_orbit_state_vectors.data(),
                   get_position_metadata.orbit_state_vector.array,
                   get_position_metadata.orbit_state_vector.size * sizeof(alus::snapengine::OrbitStateVectorComputation),
                   cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_velocities.data(),
                              get_position_metadata.sensor_velocity.array,
                              get_position_metadata.sensor_velocity.size * sizeof(alus::snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_positions.data(),
                              get_position_metadata.sensor_position.array,
                              get_position_metadata.sensor_position.size * sizeof(alus::snapengine::PosVector),
                              cudaMemcpyDeviceToHost));
    h_get_position_metadata.orbit_state_vector.size = h_orbit_state_vectors.size();
    h_get_position_metadata.orbit_state_vector.array = h_orbit_state_vectors.data();
    h_get_position_metadata.sensor_position.size = h_positions.size();
    h_get_position_metadata.sensor_position.array = h_positions.data();
    h_get_position_metadata.sensor_velocity.size = h_velocities.size();
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
        //FillDemCoordinates(tile, target_geocoding, &dem_geocoding);

        //std::vector<double> source_tile_dem_data(tile.tc_tile_coordinates.dem_width * tile.tc_tile_coordinates
        //                                                                                    .dem_height, 10.0);

        //tile.dem_tile_data_buffer = alus::cuda::GetKernelArray(source_tile_dem_data);

        // Prepares target tile values
        thrust::host_vector<double> target_tile_data(tile.tc_tile_coordinates.target_width * tile.tc_tile_coordinates.target_height);
        tile.target_tile_data_buffer = {target_tile_data.data(), target_tile_data.size()};

        //std::vector<double> elevation_tile_data(tile.target_tile_data_buffer.size);
        //tile.elevation_tile_data_buffer = alus::cuda::GetKernelArray(elevation_tile_data);
        //DemCuda(tile, dem_ds_.GetDataset()->GetNoDataValue(), dem_geo_transform, target_geo_transform);

        // Prepare TileData
        alus::snapengine::resampling::ResamplingRaster resampling_raster{0, 0, -1, 0, 0, nullptr, nullptr};
        tile.tile_data.resampling_raster = &resampling_raster;

        alus::Rectangle source_rectangle{};
        bool is_source_rectangle_calculated = GetSourceRectangle(tile,
                                                                 target_geo_transform,
                                                                 0.0, // TODO: no data value
                                                                 metadata_.avg_scene_height,
                                                                 coh_ds_.GetXSize(),
                                                                 coh_ds_.GetYSize(),
                                                                 h_get_position_metadata,
                                                                 source_rectangle);
        auto tile_coordinates = tile.tc_tile_coordinates;

        std::vector<double> source_tile_data(tile.tc_tile_coordinates.source_width * tile.tc_tile_coordinates.source_height);
        tile.source_tile_data_buffer = alus::cuda::GetKernelArray(source_tile_data);
        //tile.dem_tile_data_buffer = alus::cuda::GetKernelArray(source_tile_dem_data);

        TerrainCorrectionKernelArgs args{static_cast<unsigned int>(coh_ds_.GetXSize()),
                                         static_cast<unsigned int>(coh_ds_.GetYSize()),
                                         0.0, // TODO: no data value
                                         metadata_.avg_scene_height,
                         //                dem_geo_transform,
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
        args.metadata = computation_metadata;
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
                     //          {},
                     //          {},
                               {nullptr, nullptr, nullptr}};
            output_tiles.push_back(tile);
        }
    }

    return output_tiles;
}

alus::snapengine::old::Product TerrainCorrection::CreateTargetProduct(
    const alus::snapengine::geocoding::Geocoding* source_geocoding,
    snapengine::geocoding::Geocoding*& target_geocoding,
    int source_width,
    int source_height,
    double azimuth_spacing,
    const std::string& output_filename) {
    const char* const OUTPUT_FORMAT = "GTiff";

    // TODO: move to some constants header if used in other functions
    const double SEMI_MAJOR_AXIS{6378137.0};
    const double RTOD{57.29577951308232};

    double pixel_spacing_in_meter = std::trunc(azimuth_spacing * 100 + 0.5) / 100.0;
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

    output_dataset = output_driver->Create(output_filename.c_str(), a, b, 1, GDT_Float64, output_driver_options);

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
    alus::snapengine::old::Product target_product{
        target_geocoding, *output_dataset, "TIFF"};  // TODO: add destructor product class

    return target_product;
}

ComputationMetadata TerrainCorrection::CreateComputationMetadata() {

    std::vector<snapengine::OrbitStateVectorComputation> comp_orbits;
    for (auto && o : metadata_.orbit_state_vectors2) {
        comp_orbits.push_back({o.time_mjd_, o.x_pos_, o.y_pos_, o.z_pos_, o.x_vel_, o.y_vel_, o.z_vel_});
    }

    cuda::KernelArray<snapengine::OrbitStateVectorComputation> kernel_orbits{nullptr, comp_orbits.size()};
    CHECK_CUDA_ERR(
        cudaMalloc(&kernel_orbits.array, sizeof(snapengine::OrbitStateVectorComputation) * kernel_orbits.size));
    CHECK_CUDA_ERR(cudaMemcpy(kernel_orbits.array,
                              comp_orbits.data(),
                              sizeof(snapengine::OrbitStateVectorComputation) * kernel_orbits.size,
                              cudaMemcpyHostToDevice));
    cuda_arrays_to_clean_.push_back(kernel_orbits.array);

    ComputationMetadata md{};
    md.orbit_state_vectors = kernel_orbits;
    md.first_line_time_mjd = metadata_.first_line_time.GetMjd();
    md.last_line_time_mjd = metadata_.last_line_time.GetMjd();
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

TerrainCorrection::~TerrainCorrection() {
    FreeCudaArrays();
}
void TerrainCorrection::FreeCudaArrays() {
    for (auto&& a : cuda_arrays_to_clean_) {
        cudaFree(a);
    }
    cuda_arrays_to_clean_.erase(cuda_arrays_to_clean_.begin(), cuda_arrays_to_clean_.end());
}

}  // namespace alus
