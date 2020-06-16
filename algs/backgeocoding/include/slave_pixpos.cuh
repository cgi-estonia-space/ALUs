#pragma once
#include <cstdio>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "shapes.h"
#include "srtm3_elevation_model_constants.h"
#include "pointer_holders.h"

namespace alus {

struct SlavePixPosData{
    int numLines;
    int numPixels;

    int mBurstIndex;
    int sBurstIndex;

    int latMaxIdx;
    int latMinIdx;
    int lonMinIdx;
    int lonMaxIdx;

    PointerArray tiles;
};

cudaError_t launchSlavePixPos(dim3 gridSize, dim3 bockSize, SlavePixPosData calcData);

} //namespace
