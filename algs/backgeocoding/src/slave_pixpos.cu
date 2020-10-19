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
#include "orbit_state_vector.h"
#include "backgeocoding_constants.h"
#include "slave_pixpos.cuh"

#include "earth_gravitational_model96.cuh"
#include "general_constants.h"
#include "geo_utils.cuh"
#include "position_data.h"
#include "sar_geocoding.cuh"
#include "srtm3_elevation_calc.cuh"

/**
 * The contents of this file refer to BackGeocodingOp.computeSlavePixPos in SNAP's java code.
 * They are from s1tbx module.
 */

namespace alus {
namespace backgeocoding{

inline __device__ int GetPosition(s1tbx::DeviceSubswathInfo *subswath_info,
                                  s1tbx::DeviceSentinel1Utils *sentinel1_utils,
                                  int burst_index,
                                  s1tbx::PositionData *position_data,
                                  snapengine::OrbitStateVector *orbit,
                                  const int num_orbit_vec,
                                  const double dt,
                                  int idx,
                                  int idy){

    const double zero_doppler_time_in_days = s1tbx::sargeocoding::GetZeroDopplerTime(sentinel1_utils->line_time_interval,
                                                                                     sentinel1_utils->wavelength,
                                                                                     position_data->earth_point,
                                                                                     orbit,
                                                                                     num_orbit_vec,
                                                                                     dt);

    if (zero_doppler_time_in_days == s1tbx::sargeocoding::NON_VALID_ZERO_DOPPLER_TIME) {
        return 0;
    }

    const double zero_doppler_time = zero_doppler_time_in_days * snapengine::constants::secondsInDay;
    position_data->azimuth_index =
        burst_index * subswath_info->lines_per_burst +
        (zero_doppler_time - subswath_info->device_burst_first_line_time[burst_index]) /
                                       subswath_info->azimuth_time_interval;


    //this is here to force a very specific compilation. This changes values of the above operation closer to snap
    //TODO: consider getting rid of it once the whole algorithm is complete and we no longer need snap to debug.
    //https://jira.devzone.ee/browse/SNAPGPU-169
    if(idx == 55 && idy == 53){
        printf("AZ index %.10f zero doppler time: %.10f FLT %.10f AZTI %.10f zero doppler in days %.10f\n",
               position_data->azimuth_index, zero_doppler_time,
               subswath_info->device_burst_first_line_time[burst_index],
               subswath_info->azimuth_time_interval,
               zero_doppler_time_in_days);
    }

    cuda::KernelArray<snapengine::OrbitStateVector> orbit_vectors;

    orbit_vectors.array = orbit;
    orbit_vectors.size = num_orbit_vec;
    const double slantRange = s1tbx::sargeocoding::ComputeSlantRangeImpl(
        zero_doppler_time_in_days, orbit_vectors, position_data->earth_point, position_data->sensor_pos);

    if (!sentinel1_utils->srgr_flag) {
        position_data->range_index = (slantRange - subswath_info->slr_time_to_first_pixel * snapengine::constants::lightSpeed) /
            sentinel1_utils->range_spacing;
    } else {
        //TODO: implement this some day, as we don't need it for first demo.
        /*position_data->range_index = s1tbx::sargeocoding::computeRangeIndex(
            su.srgrFlag, su.sourceImageWidth, su.firstLineUTC, su.lastLineUTC,
            su.rangeSpacing, zeroDopplerTimeInDays, slantRange, su.nearEdgeSlantRange, su.srgrConvParams);*/
    }

    if (!sentinel1_utils->near_range_on_left) {
        position_data->range_index = sentinel1_utils->source_image_width - 1 - position_data->range_index;
    }

    return 1;
}

//exclusively supports SRTM3 digital elevation map and none other
__global__ void SlavePixPos(SlavePixPosData calc_data){
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const size_t my_index = calc_data.num_pixels * idx + idy;
    double geo_pos_lat;
    double geo_pos_lon;
    double alt;
    s1tbx::PositionData pos_data;

    pos_data.azimuth_index = 0;
    pos_data.range_index = 0;


    if(idx < calc_data.num_lines && idy < calc_data.num_pixels){
        geo_pos_lat = (snapengine::srtm3elevationmodel::RASTER_HEIGHT - (calc_data.lat_max_idx + idx)) *
                          snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 60.0;
        geo_pos_lon = (calc_data.lon_min_idx + idy) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 180.0;

        calc_data.device_lats[my_index] = geo_pos_lat;
        calc_data.device_lons[my_index] = geo_pos_lon;

        alt= snapengine::srtm3elevationmodel::GetElevation(geo_pos_lat, geo_pos_lon, &calc_data.tiles);
        if(alt == calc_data.dem_no_data_value && !calc_data.mask_out_area_without_elevation) {
            alt = snapengine::earthgravitationalmodel96::GetEGM96(
                geo_pos_lat, geo_pos_lon, calc_data.max_lats, calc_data.max_lons, calc_data.egm);
        }

        if(alt != calc_data.dem_no_data_value ){
            snapengine::geoutils::Geo2xyzWgs84Impl(geo_pos_lat,geo_pos_lon, alt, pos_data.earth_point);


            if(GetPosition(calc_data.device_master_subswath,
                            calc_data.device_master_utils,
                            calc_data.m_burst_index,
                            &pos_data,
                            calc_data.device_master_orbit_state_vectors,
                            calc_data.nr_of_master_vectors,
                            calc_data.master_dt, idx, idy)) {

                calc_data.device_master_az[my_index] = pos_data.azimuth_index;
                calc_data.device_master_rg[my_index] = pos_data.range_index;

                if (GetPosition(calc_data.device_slave_subswath,
                                calc_data.device_slave_utils,
                                calc_data.s_burst_index,
                                &pos_data,
                                calc_data.device_slave_orbit_state_vectors,
                                calc_data.nr_of_slave_vectors,
                                calc_data.slave_dt, idx, idy)) {

                    calc_data.device_slave_az[my_index] = pos_data.azimuth_index;
                    calc_data.device_slave_rg[my_index] = pos_data.range_index;

                    //race condition is not important. we need to know that we have atleast 1 valid index.
                    (*calc_data.device_valid_index_counter)++;
                }
            }
        }else{
            calc_data.device_master_az[calc_data.num_pixels * idx + idy] = INVALID_INDEX;
            calc_data.device_master_rg[calc_data.num_pixels * idx + idy] = INVALID_INDEX;
        }
    }
}

cudaError_t LaunchSlavePixPos(dim3 grid_size, dim3 block_size, SlavePixPosData calc_data){
    SlavePixPos<<<grid_size, block_size>>>(calc_data);
    return cudaGetLastError();
}

} //namespace
} //namespace
