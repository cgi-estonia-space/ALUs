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

#include "backgeocoding_constants.h"
#include "backgeocoding_utils.cuh"
#include "copdem_cog_30m_calc.cuh"
#include "cuda_util.h"
#include "dem_calc.cuh"
#include "position_data.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.cuh"
#include "snap-engine-utilities/engine-utilities/eo/geo_utils.cuh"

/**
 * The contents of this file refer to BackGeocodingOp.computeSlavePixPos in SNAP's java code.
 * They are from s1tbx module.
 */


__constant__ double master_osv_lut[3000];
__constant__ double slave_osv_lut[3000];


void OSVLUTToConstantMem(const std::vector<double>& master, const std::vector<double>& slave)
{
    if(master.size() > 3000 || slave.size() > 3000)
    {
        throw std::runtime_error("OSV LUT error");
    }
    cudaMemcpyToSymbol(master_osv_lut, master.data(), master.size()*sizeof(double), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(slave_osv_lut, slave.data(), slave.size() * sizeof(double), 0, cudaMemcpyHostToDevice);
}

namespace alus {
namespace backgeocoding {

// exclusively supports SRTM3 digital elevation map and none other



__global__ void SlavePixPos(SlavePixPosData calc_data) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const size_t my_index = calc_data.num_pixels * idx + idy;
    double geo_pos_lat{};
    double geo_pos_lon{};
    double alt{calc_data.dem_property->no_data_value};
    bool valid_coord{false};
    s1tbx::PositionData pos_data;

    pos_data.azimuth_index = 0;
    pos_data.range_index = 0;

    if (idx >= calc_data.num_lines || idy >= calc_data.num_pixels) {
        return;
    }

    geo_pos_lat = (calc_data.dem_property->grid_total_height_pixels - (calc_data.lat_max_idx + idx)) *
                      calc_data.dem_property->tile_pixel_size_deg_y - calc_data.dem_property->grid_max_lat;

    if (calc_data.dem_type == dem::Type::COPDEM_COG30m) {
        const auto* prop = calc_data.dem_property;
        geo_pos_lon = (calc_data.lon_min_idx + idy) * prop->tile_pixel_size_deg_x - calc_data.dem_property->grid_max_lon;
        alt = dem::CopDemCog30mGetElevation(geo_pos_lat, geo_pos_lon, &calc_data.tiles, calc_data.dem_property);
        valid_coord = true;
    } else if (calc_data.dem_type == dem::Type::SRTM3) {
        geo_pos_lon = (calc_data.lon_min_idx + idy) * calc_data.dem_property->tile_pixel_size_deg_x -
                      calc_data.dem_property->grid_max_lon;
        alt = snapengine::dem::GetElevation(geo_pos_lat, geo_pos_lon, &calc_data.tiles, calc_data.dem_property);
        valid_coord = true;
    }

    calc_data.device_lats[my_index] = geo_pos_lat;
    calc_data.device_lons[my_index] = geo_pos_lon;

    if (alt == calc_data.dem_no_data_value && !calc_data.mask_out_area_without_elevation && valid_coord) {
        alt = snapengine::earthgravitationalmodel96computation::GetEGM96(geo_pos_lat, geo_pos_lon, calc_data.max_lats,
                                                                         calc_data.max_lons, calc_data.egm);
    }

    if (alt != calc_data.dem_no_data_value && valid_coord) {
        snapengine::geoutils::Geo2xyzWgs84Impl(geo_pos_lat, geo_pos_lon, alt, pos_data.earth_point);

#if 1
        if (GetPositionLUT(calc_data.device_master_subswath, calc_data.device_master_utils, calc_data.m_burst_index,
                        &pos_data, calc_data.device_master_orbit_state_vectors, calc_data.nr_of_master_vectors,
                        calc_data.master_dt, idx, idy, master_osv_lut)) {
            calc_data.device_master_az[my_index] = pos_data.azimuth_index;
            calc_data.device_master_rg[my_index] = pos_data.range_index;

            if (GetPositionLUT(calc_data.device_slave_subswath, calc_data.device_slave_utils, calc_data.s_burst_index,
                            &pos_data, calc_data.device_slave_orbit_state_vectors, calc_data.nr_of_slave_vectors,
                            calc_data.slave_dt, idx, idy, slave_osv_lut)) {
                calc_data.device_slave_az[my_index] = pos_data.azimuth_index;
                calc_data.device_slave_rg[my_index] = pos_data.range_index;

                // race condition is not important. we need to know that we have atleast 1 valid index.
                (*calc_data.device_valid_index_counter)++;
            }
#else
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
#endif
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

cudaError_t LaunchSlavePixPos(SlavePixPosData calc_data, cudaStream_t stream) {
    // CC7.5 does not launch with 24x24
    // TODO use smarted launcher configuration, ie occupancy calculator
    dim3 block_size(16, 16);
    dim3 grid_size(cuda::GetGridDim(block_size.x, calc_data.num_lines),
                   cuda::GetGridDim(block_size.y, calc_data.num_pixels));

    SlavePixPos<<<grid_size, block_size, 0, stream>>>(calc_data);
    return cudaGetLastError();

}

cudaError_t LaunchFillXAndY(double* device_x_points, double* device_y_points, size_t points_size,
                            double placeholder_value, cudaStream_t stream) {
    dim3 block_size(1024);
    dim3 grid_size(cuda::GetGridDim(block_size.x, points_size));

    FillXAndY<<<grid_size, block_size, 0, stream>>>(device_x_points, device_y_points, points_size, placeholder_value);
    return cudaGetLastError();
}

}  // namespace backgeocoding
}  // namespace alus
