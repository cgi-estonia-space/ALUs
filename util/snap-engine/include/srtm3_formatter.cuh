#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


namespace alus {
namespace snapengine{

struct DemFormatterData{
    double m00, m01, m02, m10, m11, m12;
    double no_data_value;
    int x_size, y_size;
    int max_lats;
    int max_lons;
    double* egm;
};

cudaError_t LaunchDemFormatter(dim3 grid_size, dim3 block_size, double *target, double *source, DemFormatterData data);

}//namespace
}//namespace
