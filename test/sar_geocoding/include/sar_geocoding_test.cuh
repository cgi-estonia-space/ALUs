#pragma once
#include "pos_vector.h"
#include "orbit_state_vector_computation.h"

namespace alus {
namespace tests{

struct ZeroDopplerTimeData{
    int data_size;
    double *device_line_time_interval;
    double *device_wavelengths;
    alus::snapengine::PosVector *device_earth_points;
    alus::snapengine::OrbitStateVectorComputation *orbit;
    int num_orbit_vec;
    double dt;
};

cudaError_t LaunchZeroDopplerTimeTest(dim3 grid_size, dim3 block_size, double *results, ZeroDopplerTimeData data);

}//namespace
}//namespace