#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Constants.hpp"
#include "Shapes.hpp"
#include "SubSwathInfo.hpp"

namespace slap{

cudaError_t launchDerampDemod(dim3 gridSize,
    dim3 blockSize,
    Rectangle rectangle,
    double *slaveI,
    double *slaveQ,
    double *demodPhase,
    double *demodI,
    double *demodQ,
    SubSwathInfo *subSwath,
    int sBurstIndex);

} //slap
