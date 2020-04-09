#pragma once
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Shapes.hpp"
#include "SRTM3ElevationModel.h"

namespace slap{

struct SlavePixPosData{
    int numLines;
    int numPixels;

    int mBurstIndex;
    int sBurstIndex;

    int latMaxIdx;
    int latMinIdx;
    int lonMinIdx;
    int lonMaxIdx;
};

cudaError_t launchSlavePixPos(dim3 gridSize, dim3 bockSize, SlavePixPosData calcData);

} //namespace
