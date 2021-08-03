/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */

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
    CudaErrorException(cudaError_t cuda_error, std::string file, int line)
        : std::runtime_error("CUDA error (" + std::to_string(static_cast<int>(cuda_error)) + ")-'" +
                             std::string{cudaGetErrorString(cuda_error)} + "' at " + file + ":" + std::to_string(line)),
          cuda_error_{static_cast<int>(cuda_error)},
          file_{std::move(file)},
          line_{line} {}

    CudaErrorException(std::string what)
        : std::runtime_error(std::move(what)),
          cuda_error_{},
          file_{},
          line_{} {}

private:
    const int cuda_error_;
    std::string file_;
    const int line_;
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
