#pragma once
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "shapes.h"
#include "srtm3_elevation_model_constants.h"
#include "pointer_holders.h"

namespace alus {

struct SlavePixPosData{
    int num_lines;
    int num_pixels;

    int m_burst_index;
    int s_burst_index;

    int lat_max_idx;
    int lat_min_idx;
    int lon_min_idx;
    int lon_max_idx;

    PointerArray tiles;
};

cudaError_t LaunchSlavePixPos(dim3 grid_size, dim3 block_size, SlavePixPosData calc_data);

} //namespace
