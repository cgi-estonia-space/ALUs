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
#pragma once

#include "crs_geocoding.cuh"
#include "cuda_util.cuh"
#include "geocoding.cuh"
#include "get_position.cuh"
#include "orbit_state_vector_computation.h"
#include "pointer_holders.h"
#include "raster_properties.hpp"
#include "resampling.h"
#include "tc_tile.h"
#include "computation_metadata.h"
#include "tie_point_geocoding.cuh"

struct TerrainCorrectionKernelArgs {
    unsigned int source_image_width;   // TODO: is it needed?
    unsigned int source_image_height;  // TODO: is it needed?
    double dem_no_data_value;
    double avg_scene_height;
    //alus::GeoTransformParameters dem_geo_transform;
    alus::GeoTransformParameters target_geo_transform;
    alus::snapengine::geocoding::TiePointGeocoding *source_geocoding;
    alus::snapengine::geocoding::CrsGeocoding *target_geocoding;

    int diff_lat;
    alus::terraincorrection::ComputationMetadata metadata;
    alus::terraincorrection::GetPositionMetadata get_position_metadata;
    alus::snapengine::resampling::TileData tile_data;
};

void CalculateVelocitiesAndPositions(const int source_image_height,
                                     const double first_line_utc,
                                     const double line_time_interval,
                                     alus::cuda::KernelArray<alus::snapengine::OrbitStateVectorComputation> vectors,
                                     alus::cuda::KernelArray<alus::snapengine::PosVector> velocities,
                                     alus::cuda::KernelArray<alus::snapengine::PosVector> positions);

bool DemCuda(alus::TcTile tile,
             double dem_no_data_value,
             alus::GeoTransformParameters dem_geo_transform,
             alus::GeoTransformParameters target_geo_transform);

// TODO: maybe make less arguments
void RunTerrainCorrectionKernel(alus::TcTile tile, TerrainCorrectionKernelArgs args);

/**
 * Copy of RangeDopplerGeocodingOp.GetSourceRectangle()
 *
 * @param tile
 * @return
 * @todo finish doc
 */
bool GetSourceRectangle(alus::TcTile &tile,
                        alus::GeoTransformParameters target_geo_transform,
                        double dem_no_data_value,
                        double avg_scene_height,
                        int source_image_width,
                        int source_image_height,
                        alus::terraincorrection::GetPositionMetadata get_position_metadata,
                        alus::Rectangle &source_rectangle);

alus::terraincorrection::GetPositionMetadata GetGetPositionMetadata(
    int src_image_height,
    const alus::terraincorrection::ComputationMetadata range_doppler_metadata,
    alus::cuda::KernelArray<alus::snapengine::PosVector> *sensor_positions,
    alus::cuda::KernelArray<alus::snapengine::PosVector> *sensor_velocities);

alus::snapengine::resampling::TileData CopyTileDataToDevice(
    alus::snapengine::resampling::TileData h_tile_data,
    alus::snapengine::resampling::Tile *d_tile,
    double *d_source_tile_data_buffer,
    alus::Rectangle *d_source_rectangle,
    alus::snapengine::resampling::ResamplingRaster *d_resampling_raster);

void SRTM3DemCuda(alus::PointerArray dem_tiles,
                  double *elevations,
                  alus::TcTileCoordinates tile_coordinates,
                  alus::GeoTransformParameters target_geo_transform);

__global__ void DemCudaKernel(double *lons, double *lats, double *results, size_t size, alus::PointerArray dem_tiles);

__global__ void GetLatLonGrid(double *lons,
                              double *lats,
                              alus::TcTileCoordinates tile_coordinates,
                              alus::GeoTransformParameters target_geo_transform);