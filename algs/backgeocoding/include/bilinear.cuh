#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace alus {

cudaError_t LaunchBilinearInterpolation(
                        dim3 grid_size,
                        dim3 block_size,
                        double *x_pixels,
                        double *y_pixels,
                        double *demod_phase,
                        double *demod_i,
                        double *demod_q,
                        int *int_params,
                        double double_params,
                        float *results_i,
                        float *results_q);

} //namespace
