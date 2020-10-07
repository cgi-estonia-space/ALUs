#include "terrain_correction.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <geocoding.cuh>
#include <srtm3_elevation_calc.cuh>

#include "orbit_state_vectors.h"
#include "position_data.h"
//#include "product_data.h"
#include "tc_tile.h"

#include "crs_geocoding.cuh"
#include "local_dem.cuh"
#include "range_doppler_geocoding.cuh"
#include "sar_geocoding.cuh"
#include "tie_point_geocoding.cuh"

#define UNUSED(x) (void)(x)

__global__ void CalculateVelocitiesAndPositionsKernel(
    const double first_line_utc,
    const double line_time_interval,
    alus::cuda::KernelArray<alus::snapengine::OrbitStateVectorComputation> vectors,
    alus::cuda::KernelArray<alus::snapengine::PosVector> velocities,
    alus::cuda::KernelArray<alus::snapengine::PosVector> positions) {
    const auto block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    const auto index = block_id * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
                       (threadIdx.y * blockDim.x) + threadIdx.x;
    if (index >= velocities.size) {
        return;
    }

    const double time = first_line_utc + index * line_time_interval;
    const double dt =
        (vectors.array[vectors.size - 1].timeMjd_ - vectors.array[0].timeMjd_) / static_cast<double>(vectors.size - 1);

    alus::s1tbx::orbitstatevectors::GetPositionVelocity(
        time, vectors.array, vectors.size, dt, &positions.array[index], &velocities.array[index]);
}

void CalculateVelocitiesAndPositions(const int source_image_height,
                                     const double first_line_utc,
                                     const double line_time_interval,
                                     alus::cuda::KernelArray<alus::snapengine::OrbitStateVectorComputation> vectors,
                                     alus::cuda::KernelArray<alus::snapengine::PosVector> velocities,
                                     alus::cuda::KernelArray<alus::snapengine::PosVector> positions) {
    dim3 block_dim{32, 32};
    dim3 grid_dim{static_cast<unsigned int>(source_image_height / (32 * 32) + 1)};

    CalculateVelocitiesAndPositionsKernel<<<grid_dim, block_dim>>>(
        first_line_utc, line_time_interval, vectors, velocities, positions);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
};

__device__ void SetNoDataValue(int index, double value, double *dest_data) { dest_data[index] = value; }

__global__ void TerrainCorrectionKernel(alus::TcTileCoordinates tile_coordinates,
                                        double avg_scene_height,
                                        //alus::cuda::KernelArray<double> elevations,
                                        alus::cuda::KernelArray<double> target,
                                        TerrainCorrectionKernelArgs args) {
    auto const thread_x = threadIdx.x + blockIdx.x * blockDim.x;
    auto const thread_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (thread_x >= tile_coordinates.target_width || thread_y >= tile_coordinates.target_height) {
        return;
    }
    auto const target_tile_index = thread_y * tile_coordinates.target_width + thread_x;

    double const ALTITUDE = avg_scene_height;//elevations.array[target_tile_index];

    if (ALTITUDE == args.dem_no_data_value) {
        SetNoDataValue(target_tile_index, 0.0, target.array);  // TODO: use target no_data_value
        return;
    }

    auto const lon = args.target_geo_transform.pixelSizeLon * (thread_x + tile_coordinates.target_x_0) +
                     args.target_geo_transform.pixelSizeLon / 2 +
                     args.target_geo_transform.originLon;  // TODO: remove geocoding from kernel args
    auto const lat = args.target_geo_transform.pixelSizeLat * (thread_y + tile_coordinates.target_y_0) +
                     args.target_geo_transform.pixelSizeLat / 2 + args.target_geo_transform.originLat;
    alus::Coordinates coordinates{lon < 180.0 ? lon : lon - 360.0, lat};

    alus::s1tbx::PositionData position_data{};
    if (!alus::terraincorrection::GetPositionImpl(
            coordinates.lat, coordinates.lon, ALTITUDE, position_data, args.get_position_metadata)) {
        SetNoDataValue(target_tile_index, 0.0, target.array);
        return;
    }

    if (!alus::s1tbx::sargeocoding::IsValidCellImpl(position_data.range_index,
                                                    position_data.azimuth_index,
                                                    args.diff_lat,
                                                    args.source_image_width - 1,
                                                    args.source_image_height - 1)) {
        SetNoDataValue(target_tile_index, 0.0, target.array);
        return;
    }

    int sub_swath_index = -1;  // TODO: check value
    args.tile_data.resampling_raster->sub_swath_index = sub_swath_index;
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];

    alus::snapengine::resampling::ResamplingIndex resampling_index{
        0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    args.tile_data.image_resampling_index = &resampling_index;

    double v = alus::terraincorrection::rangedopplergeocoding::GetPixelValue(
        position_data.azimuth_index,
        position_data.range_index,
        1,
        args.source_image_width,
        args.source_image_height,
        &args.tile_data,
        args.tile_data.source_tile == nullptr ? nullptr : args.tile_data.source_tile->data_buffer,
        sub_swath_index);
    target.array[target_tile_index] = v;
}
//bool DemCuda(alus::TcTile tile,
//             double dem_no_data_value,
//             alus::GeoTransformParameters dem_geo_transform,
//             alus::GeoTransformParameters target_geo_transform) {
//    thrust::device_vector<double> d_dem_array(tile.dem_tile_data_buffer.array,
//                                              tile.dem_tile_data_buffer.array + tile.dem_tile_data_buffer.size);
//    thrust::device_vector<double> d_product_array(tile.elevation_tile_data_buffer.size);
//
//    struct LocalDemKernelArgs kernel_args {
//        tile.tc_tile_coordinates.dem_x_0, tile.tc_tile_coordinates.dem_y_0, tile.tc_tile_coordinates.dem_width,
//            tile.tc_tile_coordinates.dem_height, tile.tc_tile_coordinates.target_x_0,
//            tile.tc_tile_coordinates.target_y_0, tile.tc_tile_coordinates.target_width,
//            tile.tc_tile_coordinates.target_height, dem_no_data_value, dem_geo_transform, target_geo_transform
//    };
//
//    RunElevationKernel(
//        thrust::raw_pointer_cast(d_dem_array.data()), thrust::raw_pointer_cast(d_product_array.data()), kernel_args);
//
//    thrust::device_vector<double>::iterator iterator;
//    thrust::find(thrust::device, d_product_array.begin(), d_product_array.end(), kernel_args.dem_no_data_value);
//    thrust::copy(d_product_array.begin(), d_product_array.end(), tile.elevation_tile_data_buffer.array);
//    if (iterator == d_product_array.end()) {
//        return true;
//    }
//    return false;
//}

void RunTerrainCorrectionKernel(alus::TcTile tile, TerrainCorrectionKernelArgs args) {
    const int BLOCK_DIM = 32;
    dim3 block_size{BLOCK_DIM, BLOCK_DIM};
    dim3 main_kernel_grid_size{tile.tc_tile_coordinates.target_width / block_size.x + 1,
                               tile.tc_tile_coordinates.target_height / block_size.y + 1};

//    double *d_elevations;
//    CHECK_CUDA_ERR(cudaMalloc(&d_elevations, sizeof(double) * tile.elevation_tile_data_buffer.size));
//    CHECK_CUDA_ERR(cudaMemcpy(d_elevations,
//                              tile.elevation_tile_data_buffer.array,
//                              sizeof(double) * tile.elevation_tile_data_buffer.size,
//                              cudaMemcpyHostToDevice));
//
//    alus::cuda::KernelArray<double> kernel_dem{d_elevations, tile.elevation_tile_data_buffer.size};
    thrust::device_vector<double> d_target(tile.target_tile_data_buffer.size);
    alus::cuda::KernelArray<double> kernel_target{thrust::raw_pointer_cast(d_target.data()), d_target.size()};

    // TODO: check with cuda-gdb that method really copies all the data
    alus::Rectangle *d_source_rectangle = nullptr;
    double *d_source_tile_data_buffer = nullptr;
    alus::snapengine::resampling::Tile *d_source_tile = nullptr;
    alus::snapengine::resampling::ResamplingRaster *d_resampling_raster = nullptr;

    tile.tile_data = CopyTileDataToDevice(
        tile.tile_data, d_source_tile, d_source_tile_data_buffer, d_source_rectangle, d_resampling_raster);

    args.tile_data = tile.tile_data;
    TerrainCorrectionKernel<<<main_kernel_grid_size, block_size>>>(
        tile.tc_tile_coordinates, args.avg_scene_height, kernel_target, args);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    CHECK_CUDA_ERR(cudaMemcpy(tile.target_tile_data_buffer.array,
                              kernel_target.array,
                              sizeof(double) * kernel_target.size,
                              cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(cudaFree(args.tile_data.source_tile));
    CHECK_CUDA_ERR(cudaFree(args.tile_data.resampling_raster));
    CHECK_CUDA_ERR(cudaFree(args.tile_data.image_resampling_index));
//    CHECK_CUDA_ERR(cudaFree(d_elevations));
    CHECK_CUDA_ERR(cudaFree(d_source_tile));
    CHECK_CUDA_ERR(cudaFree(d_source_tile_data_buffer));
    CHECK_CUDA_ERR(cudaFree(d_source_rectangle));
    CHECK_CUDA_ERR(cudaFree(d_resampling_raster));
}

bool GetSourceRectangle(alus::TcTile &tile,
                        alus::GeoTransformParameters target_geo_transform,
                        double dem_no_data_value,
                        double avg_scene_height,
                        int source_image_width,
                        int source_image_height,
                        alus::terraincorrection::GetPositionMetadata get_position_metadata,
                        alus::Rectangle &source_rectangle) {
    const int NUM_POINTS_PER_ROW = 5;
    const int NUM_POINTS_PER_COL = 5;
    const int X_OFFSET = tile.tc_tile_coordinates.target_width / (NUM_POINTS_PER_ROW - 1);
    const int Y_OFFSET = tile.tc_tile_coordinates.target_height / (NUM_POINTS_PER_COL - 1);
    const int MARGIN = 1;  // Specific value for Bilinear Interpolation

    int x_max = INT_MIN;
    int x_min = INT_MAX;
    int y_max = INT_MIN;
    int y_min = INT_MAX;

    for (int i = 0; i < NUM_POINTS_PER_COL; i++) {
        const double Y = i == NUM_POINTS_PER_COL - 1
                             ? tile.tc_tile_coordinates.target_y_0 + tile.tc_tile_coordinates.target_height - 1
                             : tile.tc_tile_coordinates.target_y_0 + i * Y_OFFSET;

        for (int j = 0; j < NUM_POINTS_PER_ROW; j++) {
            const double X = j == NUM_POINTS_PER_ROW - 1
                                 ? tile.tc_tile_coordinates.target_x_0 + tile.tc_tile_coordinates.target_width - 1
                                 : tile.tc_tile_coordinates.target_x_0 + j * X_OFFSET;

//            int dem_index =
//                static_cast<int>(Y - tile.tc_tile_coordinates.target_y_0) * tile.tc_tile_coordinates.target_width +
//                static_cast<int>(X - tile.tc_tile_coordinates.target_x_0);
            double altitude = avg_scene_height;//tile.elevation_tile_data_buffer.array[dem_index];
            if (altitude == dem_no_data_value) {
                continue;
            }

            alus::Coordinates coordinates{target_geo_transform.originLon + X * target_geo_transform.pixelSizeLon +
                                              target_geo_transform.pixelSizeLon / 2,
                                          target_geo_transform.originLat + Y * target_geo_transform.pixelSizeLat +
                                              target_geo_transform.pixelSizeLat / 2};

            alus::s1tbx::PositionData position_data{};

            if (!alus::terraincorrection::GetPositionImpl(
                    coordinates.lat, coordinates.lon, altitude, position_data, get_position_metadata)) {
                continue;
            }
            // TODO: maybe use int as in java code. However, it might reduce precision
            if (x_max < position_data.range_index) {
                x_max = std::ceil(position_data.range_index);
            }

            if (x_min > position_data.range_index) {
                x_min = std::floor(position_data.range_index);
            }

            if (y_max < position_data.azimuth_index) {
                y_max = std::ceil(position_data.azimuth_index);
            }

            if (y_min > position_data.azimuth_index) {
                y_min = std::floor(position_data.azimuth_index);
            }
        }
    }

    x_min = std::max(x_min - MARGIN, 0);
    x_max = std::min(x_max + MARGIN, source_image_width - 1);
    y_min = std::max(y_min - MARGIN, 0);
    y_max = std::min(y_max + MARGIN, source_image_height - 1);

    if (x_min > x_max || y_min > y_max) {
        return false;
    }

    source_rectangle = {x_min, y_min, x_max - x_min + 1, y_max - y_min + 1};
    tile.tc_tile_coordinates.source_height = source_rectangle.height;
    tile.tc_tile_coordinates.source_width = source_rectangle.width;
    tile.tc_tile_coordinates.source_x_0 = source_rectangle.x;
    tile.tc_tile_coordinates.source_y_0 = source_rectangle.y;
    return true;
}

alus::snapengine::resampling::TileData CopyTileDataToDevice(
    alus::snapengine::resampling::TileData h_tile_data,
    alus::snapengine::resampling::Tile *d_tile,
    double *d_source_tile_data_buffer,
    alus::Rectangle *d_source_rectangle,
    alus::snapengine::resampling::ResamplingRaster *d_resampling_raster) {
    if (h_tile_data.resampling_raster->source_rectangle != nullptr) {
        CHECK_CUDA_ERR(cudaMalloc(&d_source_rectangle, sizeof(alus::Rectangle)));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_rectangle,
                                  h_tile_data.resampling_raster->source_rectangle,
                                  sizeof(alus::Rectangle),
                                  cudaMemcpyHostToDevice));
    }

    if (h_tile_data.source_tile != nullptr) {
        auto data_buffer_size = h_tile_data.source_tile->height * h_tile_data.source_tile->width;
        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile_data_buffer, sizeof(double) * data_buffer_size));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile_data_buffer,
                                  h_tile_data.source_tile->data_buffer,
                                  sizeof(double) * data_buffer_size,
                                  cudaMemcpyHostToDevice));

        h_tile_data.source_tile->data_buffer = d_source_tile_data_buffer;

        CHECK_CUDA_ERR(cudaMalloc(&d_tile, sizeof(alus::snapengine::resampling::Tile)));
        CHECK_CUDA_ERR(cudaMemcpy(
            d_tile, h_tile_data.source_tile, sizeof(alus::snapengine::resampling::Tile), cudaMemcpyHostToDevice));
    }

    h_tile_data.resampling_raster->source_rectangle = d_source_rectangle;
    h_tile_data.resampling_raster->source_tile_i = d_tile;
    CHECK_CUDA_ERR(cudaMalloc(&d_resampling_raster, sizeof(alus::snapengine::resampling::ResamplingRaster)));
    CHECK_CUDA_ERR(cudaMemcpy(d_resampling_raster,
                              h_tile_data.resampling_raster,
                              sizeof(alus::snapengine::resampling::ResamplingRaster),
                              cudaMemcpyHostToDevice));

    return {d_resampling_raster, d_tile, nullptr};
}

void SRTM3DemCuda(alus::PointerArray dem_tiles,
                  double *elevations,
                  alus::TcTileCoordinates tile_coordinates,
                  alus::GeoTransformParameters target_geo_transform) {
    // Populates longitude and latitude vectors
    auto const pixel_count = tile_coordinates.target_width * tile_coordinates.target_height;
    thrust::device_vector<double> lons(pixel_count);
    thrust::device_vector<double> lats(pixel_count);
    dim3 pos_kernel_block_size{32, 32};
    dim3 pos_kernel_grid_size{tile_coordinates.target_width / pos_kernel_block_size.x + 1,
                              tile_coordinates.target_height / pos_kernel_block_size.y + 1};
    GetLatLonGrid<<<pos_kernel_grid_size, pos_kernel_block_size>>>(thrust::raw_pointer_cast(lons.data()),
                                                                   thrust::raw_pointer_cast(lats.data()),
                                                                   tile_coordinates,
                                                                   target_geo_transform);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());

    dim3 elevation_kernel_block_size{1024};
    dim3 elevation_kernel_grid_size{static_cast<unsigned int>(lons.size() / elevation_kernel_block_size.x + 1)};

    DemCudaKernel<<<elevation_kernel_grid_size, elevation_kernel_block_size>>>(thrust::raw_pointer_cast(lons.data()),
                                                                               thrust::raw_pointer_cast(lats.data()),
                                                                               elevations,
                                                                               pixel_count,
                                                                               dem_tiles);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    CHECK_CUDA_ERR(cudaGetLastError());
}

__global__ void GetLatLonGrid(double *lons,
                              double *lats,
                              alus::TcTileCoordinates tile_coordinates,
                              alus::GeoTransformParameters target_geo_transform) {
    auto const index_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto const index_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (index_x >= tile_coordinates.target_width || index_y >= tile_coordinates.target_height) {
        return;
    }

    auto const pixel_x = index_x + tile_coordinates.target_x_0;
    auto const pixel_y = index_y + tile_coordinates.target_y_0;

    auto const index = index_y * tile_coordinates.target_width + index_x;

    lats[index] = target_geo_transform.originLat + target_geo_transform.pixelSizeLat / 2 +
                  target_geo_transform.pixelSizeLat * pixel_y;
    lons[index] = target_geo_transform.originLon + target_geo_transform.pixelSizeLon / 2 +
                  target_geo_transform.pixelSizeLon * pixel_x;
}

__global__ void DemCudaKernel(
    double *lons, double *lats, double *results, size_t const size, alus::PointerArray dem_tiles) {
    auto const index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= size) {
        return;
    }

    results[index] = alus::snapengine::srtm3elevationmodel::GetElevation(lats[index], lons[index], &dem_tiles);
}