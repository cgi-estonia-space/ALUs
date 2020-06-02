#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Constants.hpp"
#include "shapes.h"
#include "subswath_info.cuh"

namespace alus {

cudaError_t launchDerampDemod(dim3 gridSize,
    dim3 blockSize,
    Rectangle rectangle,
    double *slaveI,
    double *slaveQ,
    double *demodPhase,
    double *demodI,
    double *demodQ,
    DeviceSubswathInfo *subSwath,
    int sBurstIndex);

} //alus
