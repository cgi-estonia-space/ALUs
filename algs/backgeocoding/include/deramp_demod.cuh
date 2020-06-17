#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "general_constants.h"
#include "shapes.h"
#include "subswath_info.cuh"

namespace alus {

cudaError_t LaunchDerampDemod(
    dim3 grid_size,
    dim3 block_size,
    Rectangle rectangle,
    double *slave_i,
    double *slave_q,
    double *demod_phase,
    double *demod_i,
    double *demod_q,
    DeviceSubswathInfo *sub_swath,
    int s_burst_index);

} //alus
