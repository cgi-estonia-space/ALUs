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
#include "slave_pixpos_computation.h"

#include <cstdio>

#include "../../../snap-engine/srtm3_elevation_calc.cuh"
#include "backgeocoding_constants.h"
#include "backgeocoding_utils.cuh"
#include "position_data.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"
#include "snap-engine-utilities/engine-utilities/eo/geo_utils.cuh"

#include "../../../snap-engine/srtm3_elevation_model_constants.h"
#include "cuda_util.h"

/**
 * The contents of this file refer to BackGeocodingOp.computeSlavePixPos in SNAP's java code.
 * They are from s1tbx module.
 */

namespace alus {
namespace backgeocoding {

// exclusively supports SRTM3 digital elevation map and none other
__global__ void SlavePixPos(SlavePixPosData calc_data) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const size_t my_index = calc_data.num_pixels * idx + idy;
    double geo_pos_lat;
    double geo_pos_lon;
    double alt;
    s1tbx::PositionData pos_data;

    pos_data.azimuth_index = 0;
    pos_data.range_index = 0;

    if (idx >= calc_data.num_lines || idy >= calc_data.num_pixels) {
        return;
    }

    geo_pos_lat = (snapengine::srtm3elevationmodel::RASTER_HEIGHT - (calc_data.lat_max_idx + idx)) *
                      snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE -
                  60.0;
    geo_pos_lon =
        (calc_data.lon_min_idx + idy) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 180.0;

    calc_data.device_lats[my_index] = geo_pos_lat;
    calc_data.device_lons[my_index] = geo_pos_lon;

    alt = snapengine::srtm3elevationmodel::GetElevation(geo_pos_lat, geo_pos_lon, &calc_data.tiles);
    if (alt == calc_data.dem_no_data_value && !calc_data.mask_out_area_without_elevation) {
        alt = snapengine::earthgravitationalmodel96computation::GetEGM96(geo_pos_lat, geo_pos_lon, calc_data.max_lats,
                                                                         calc_data.max_lons, calc_data.egm);
    }

    if (alt != calc_data.dem_no_data_value) {
        snapengine::geoutils::Geo2xyzWgs84Impl(geo_pos_lat, geo_pos_lon, alt, pos_data.earth_point);

        if (GetPosition(calc_data.device_master_subswath, calc_data.device_master_utils, calc_data.m_burst_index,
                        &pos_data, calc_data.device_master_orbit_state_vectors, calc_data.nr_of_master_vectors,
                        calc_data.master_dt, idx, idy)) {
            calc_data.device_master_az[my_index] = pos_data.azimuth_index;
            calc_data.device_master_rg[my_index] = pos_data.range_index;

            if (GetPosition(calc_data.device_slave_subswath, calc_data.device_slave_utils, calc_data.s_burst_index,
                            &pos_data, calc_data.device_slave_orbit_state_vectors, calc_data.nr_of_slave_vectors,
                            calc_data.slave_dt, idx, idy)) {
                calc_data.device_slave_az[my_index] = pos_data.azimuth_index;
                calc_data.device_slave_rg[my_index] = pos_data.range_index;

                // race condition is not important. we need to know that we have atleast 1 valid index.
                (*calc_data.device_valid_index_counter)++;
            }
        }
    } else {
        calc_data.device_master_az[calc_data.num_pixels * idx + idy] = INVALID_INDEX;
        calc_data.device_master_rg[calc_data.num_pixels * idx + idy] = INVALID_INDEX;
    }
}

__global__ void FillXAndY(double* device_x_points, double* device_y_points, size_t points_size,
                          double placeholder_value) {
    const size_t idx = threadIdx.x + (blockDim.x * blockIdx.x);

    if (idx >= points_size) {
        return;
    }
    device_x_points[idx] = placeholder_value;
    device_y_points[idx] = placeholder_value;
}

cudaError_t LaunchSlavePixPos(SlavePixPosData calc_data) {
    // CC7.5 does not launch with 24x24
    // TODO use smarted launcher configuration, ie occupancy calculator
    dim3 block_size(16, 16);
    dim3 grid_size(cuda::GetGridDim(block_size.x, calc_data.num_lines),
                   cuda::GetGridDim(block_size.y, calc_data.num_pixels));

    SlavePixPos<<<grid_size, block_size>>>(calc_data);
    return cudaGetLastError();
}

cudaError_t LaunchFillXAndY(double* device_x_points, double* device_y_points, size_t points_size,
                            double placeholder_value) {
    dim3 block_size(1024);
    dim3 grid_size(cuda::GetGridDim(block_size.x, points_size));

    FillXAndY<<<grid_size, block_size>>>(device_x_points, device_y_points, points_size, placeholder_value);
    return cudaGetLastError();
}

}  // namespace backgeocoding
}  // namespace alus
