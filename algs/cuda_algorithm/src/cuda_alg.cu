#include <algorithm>
#include <array>
#include <iostream>
#include <stdio.h>

#include <cuda/cuda.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cuda_runtime_api.h>


__global__ void square(float* dOut, float* dIn)
{
    int threadIndex = threadIdx.x;
    float f = dIn[threadIndex];
    dOut[threadIndex] = f * f;
}

__global__ void cube(float* dOut, float* dIn)
{
    int threadIndex = threadIdx.x;
    float f = dIn[threadIndex];
    dOut[threadIndex] = f * f * f;
}

void cuda_kernel()
{
    constexpr size_t SERIES_SPAN{200};
    constexpr size_t SERIES_BUFFER_SIZE{SERIES_SPAN * sizeof(float)};

    // Host data buffers.
    std::array<float, SERIES_SPAN> hIn;
    std::iota(hIn.begin(), hIn.end(), 0);
    std::array<float, SERIES_SPAN> hOut;

    float* dIn;
    float* dOut;

    cudaMalloc((void**)&dIn, SERIES_BUFFER_SIZE);
    cudaMalloc((void**)&dOut, SERIES_BUFFER_SIZE);

    cudaMemcpy(dIn, hIn.data(), SERIES_BUFFER_SIZE, cudaMemcpyHostToDevice);

    cube<<<1, SERIES_SPAN>>>(dOut, dIn);

    cudaMemcpy(hOut.data(), dOut, SERIES_BUFFER_SIZE, cudaMemcpyDeviceToHost);

    int col{0};
    for (auto const result : hOut)
    {
        std::cout << result;
        if (col % 4 != 3)
            std::cout << '\t' << '\t' << '\t';
        else
            std::cout << std::endl;
        col++;
    }

    cudaFree(dIn);
    cudaFree(dOut);
}