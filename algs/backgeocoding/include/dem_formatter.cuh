#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace slap{

struct DemFormatterData{
    double m00, m01, m02, m10, m11, m12;
    double noDataValue;
    int xSize, ySize;
};

cudaError_t launchDemFormatter(dim3 gridSize, dim3 blockSize, double *target, double *source, DemFormatterData data);

}//namespace
