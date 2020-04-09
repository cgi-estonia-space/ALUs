#pragma once

#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace slap {
class CudaErrorException final : public std::runtime_error {
   public:
    CudaErrorException(cudaError_t cudaError, std::string file, int line)
        : std::runtime_error("CUDA error (" +
                             std::to_string(static_cast<int>(cudaError)) +
                             ")-'" + std::string{cudaGetErrorString(cudaError)} +
                             "' at " + file + ":" + std::to_string(line)),
          m_cudaError{cudaError},
          m_file{file},
          m_line{line} {}

   private:
    cudaError_t const m_cudaError;
    std::string m_file;
    int const m_line;
};

int getGridDim(int blockDim, int dataDim);

}  // namespace slap

inline void checkCudaError(cudaError_t const cudaErr, const char* file,
                           int const line) {
    if (cudaErr != cudaSuccess) {
        throw slap::CudaErrorException(cudaErr, file, line);
    }
}

#define CHECK_CUDA_ERR(x) checkCudaError(x, __FILE__, __LINE__)
