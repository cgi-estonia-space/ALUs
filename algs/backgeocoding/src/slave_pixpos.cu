#include "slave_pixpos.cuh"

#include "srtm3_elevation_calc.cuh"

namespace alus {

//exclusively supports SRTM3 digital elevation map and none other
__global__ void SlavePixPos(SlavePixPosData calc_data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geo_pos_lat;
    double geo_pos_lon;
    double alt;


    if(idx < calc_data.num_pixels && idy < calc_data.num_lines){
        geo_pos_lat = (snapengine::srtm3elevationmodel::RASTER_HEIGHT - calc_data.lat_max_idx + idy) *
                          snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 60.0;
        geo_pos_lon = (calc_data.lon_min_idx + idx) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 180.0;
        
        alt= snapengine::srtm3elevationmodel::GetElevation(geo_pos_lat, geo_pos_lon, &calc_data.tiles);
        if(idx==0 && idy == 0){
            printf("altitude number: %f\n", alt); //just getting rid of warnings. Continue working from here.
        }

    }
}

cudaError_t LaunchSlavePixPos(dim3 grid_size, dim3 block_size, SlavePixPosData calc_data){
    SlavePixPos<<<grid_size, block_size>>>(calc_data);
    return cudaGetLastError();
}

} //namespace
