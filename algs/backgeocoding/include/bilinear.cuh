#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace alus {

cudaError_t launchBilinearInterpolation(dim3 gridSize,
						dim3 blockSize,
						double *xPixels,
                        double *yPixels,
                        double *demodPhase,
                        double *demodI,
                        double *demodQ,
                        int *intParams,
                        double doubleParams,
                        float *resultsI,
                        float *resultsQ);

} //namespace
