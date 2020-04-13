#include "terrain_correction.hpp"

#include <algorithm>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <numeric>

#include "cuda_util.hpp"
#include "dem.hpp"
#include "local_dem.cuh"

namespace slap {

TerrainCorrection::TerrainCorrection(slap::Dataset cohDs,
                                     slap::Dataset metadata, slap::Dataset dem)
    : m_cohDs{std::move(cohDs)},
      m_metadataDs{std::move(metadata)},
      m_demDs{std::move(dem)},
      m_cohDsElevations(m_cohDs.getBand1Data().size()) {}

void TerrainCorrection::doWork() {

    auto const startCpu = std::chrono::steady_clock::now();
    auto const result = m_demDs.getLocalDemFor(
        m_cohDs, 0, 0, m_cohDs.getBand1Xsize(), m_cohDs.getBand1Ysize());
    auto const stopCpu = std::chrono::steady_clock::now();
    localDemCuda();
    auto const stopGpu = std::chrono::steady_clock::now();
    const auto cpuMillis = std::chrono::duration_cast<std::chrono::milliseconds>(stopCpu - startCpu).count();
    const auto gpuMillis = std::chrono::duration_cast<std::chrono::milliseconds>(stopGpu - stopCpu).count();
    std::cout << "CPU spent " <<  cpuMillis << std::endl;
    std::cout << "GPU spent " << gpuMillis << std::endl;

    {
        const auto [min, max] =
        std::minmax_element(std::begin(result), std::end(result));
        std::cout << "CPU lowest point at " << *min << " and highest at "
                  << *max << std::endl;
    }

    {
        const auto [min, max] =
        std::minmax_element(std::begin(m_cohDsElevations), std::end(m_cohDsElevations));
        std::cout << "CUDA lowest point at " << *min << " and highest at "
                  << *max << std::endl;
    }

    auto mismatches = std::mismatch(result.cbegin(), result.cend(), m_cohDsElevations.cbegin(), m_cohDsElevations.cend());
    std::cout << "Mismatch first " << (mismatches.first == result.cend() ? "cend" : std::to_string(*mismatches.first)) << " second " << (mismatches.second == m_cohDsElevations.cend() ? "cend" : std::to_string(*mismatches.second)) << std::endl;
    auto const averageCPU = std::accumulate(result.cbegin(), result.cend(), 0.0) / result.size();
    auto const averageGPU = std::accumulate(m_cohDsElevations.cbegin(), m_cohDsElevations.cend(), 0.0) / m_cohDsElevations.size();
    std::cout << "Diff in avg " << averageGPU - averageCPU << std::endl;
    auto const dist1 = std::distance(mismatches.first, result.cend());
    auto const dist2 = std::distance(mismatches.second, m_cohDsElevations.cend());
    std::cout << "avg CPU " << averageCPU << " avg GPU " << averageGPU << std::endl;
    std::cout << "distances " << dist1 << " " << dist2 << std::endl;
    std::cout << "Time diff " << gpuMillis - cpuMillis << std::endl;
}

void TerrainCorrection::localDemCPU() {
    auto const result = m_demDs.getLocalDemFor(
        m_cohDs, 0, 0, m_cohDs.getBand1Xsize(), m_cohDs.getBand1Ysize());

    const auto [min, max] =
        std::minmax_element(std::begin(result), std::end(result));
    std::cout << "Our area has lowest point at " << *min << " and highest at "
              << *max << std::endl;
}

void TerrainCorrection::localDemCuda() {
    auto const h_demArray = m_demDs.getData();

    double* d_demArray;
    double* d_productArray;

    try {
        CHECK_CUDA_ERR(
            cudaMalloc(&d_demArray, sizeof(double) * h_demArray.size()));
        CHECK_CUDA_ERR(cudaMalloc(
            &d_productArray, sizeof(double) * m_cohDsElevations.size()));
        CHECK_CUDA_ERR(cudaMemcpy(d_demArray, h_demArray.data(),
                                  sizeof(double) * h_demArray.size(),
                                  cudaMemcpyHostToDevice));

        struct LocalDemKernelArgs kernelArgs;
        kernelArgs.demCols = m_demDs.getColumnCount();
        kernelArgs.demRows = m_demDs.getRowCount();
        kernelArgs.targetCols = m_cohDs.getBand1Xsize();
        kernelArgs.targetRows = m_cohDs.getBand1Ysize();
        m_demDs.fillGeoTransform(
            kernelArgs.demOriginLon, kernelArgs.demOriginLat,
            kernelArgs.demPixelSizeLon, kernelArgs.demPixelSizeLat);
        m_cohDs.fillGeoTransform(
            kernelArgs.targetOriginLon, kernelArgs.targetOriginLat,
            kernelArgs.targetPixelSizeLon, kernelArgs.targetPixelSizeLat);

        CHECK_CUDA_ERR(cudaGetLastError());

        runElevationKernel(d_demArray, d_productArray, kernelArgs);

        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpy(m_cohDsElevations.data(), d_productArray,
                                  sizeof(double) * m_cohDsElevations.size(),
                                  cudaMemcpyDeviceToHost));
    } catch (slap::CudaErrorException const& cudaEx) {
        cudaFree(d_demArray);
        cudaFree(d_productArray);

        throw;
    }
    cudaFree(d_demArray);
    cudaFree(d_productArray);
}

TerrainCorrection::~TerrainCorrection() {}
}  // namespace slap
