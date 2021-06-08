#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace alus {
namespace cuda {

int GetGridDim(int blockDim, int dataDim);

// An overload for calling "cudaSetDevice()" from CUDA API in host code.
cudaError_t CudaSetDevice(int device_nr);

// An overload for calling "cudaDeviceReset()" from CUDA API in host code.
cudaError_t CudaDeviceReset();

// An overload for calling "cudaDeviceSynchronize()" from CUDA API in host code.
cudaError_t CudaDeviceSynchronize();

}  // namespace cuda

class CudaErrorException final : public std::runtime_error {
public:
    CudaErrorException(cudaError_t cudaError, std::string file, int line)
        : std::runtime_error("CUDA error (" + std::to_string(static_cast<int>(cudaError)) + ")-'" +
                             std::string{cudaGetErrorString(cudaError)} + "' at " + file + ":" + std::to_string(line)),
          m_cudaError{cudaError},
          m_file{file},
          m_line{line} {}

private:
    cudaError_t const m_cudaError;
    std::string m_file;
    int const m_line;
};

inline void checkCudaError(cudaError_t const cudaErr, const char* file, int const line) {
    if (cudaErr != cudaSuccess) {
        throw alus::CudaErrorException(cudaErr, file, line);
    }
}

inline void ReportIfCudaError(const cudaError_t cuda_err, const char* file, const int line) {
    if (cuda_err != cudaSuccess) {
        std::cerr << "CUDA error (" << cuda_err << ")-'" << cudaGetErrorString(cuda_err) << "' at " << file << ":"
                  << line;
    }
}

}  // namespace alus

#define CHECK_CUDA_ERR(x) alus::checkCudaError(x, __FILE__, __LINE__)
#define REPORT_WHEN_CUDA_ERR(x) alus::ReportIfCudaError(x, __FILE__, __LINE__)
#define PRINT_DEVICE_FUNC_OCCUPANCY(dev_func, block_size) \
    std::cout << #dev_func << " occupancy " << alus::cuda::GetOccupancyPercentageFor(dev_func, block_size) << "%";
