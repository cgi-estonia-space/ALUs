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

#include <vector>

#include <cuda_runtime.h>

#include "cuda_util.h"
#include "kernel_array.h"

namespace alus::s1tbx {

/**
 * NoiseVector to be used on CUDA devices.
 */
struct DeviceNoiseVector {
    double time_mjd;
    int line;
    cuda::KernelArray<int> pixels;
    cuda::KernelArray<float> noise_lut;
};

/**
 * Port of SNAP's NoiseVector class. Original implementation can be found in Sentinel1Utils.java file.
 */
struct NoiseVector {
    double time_mjd;
    int line;
    std::vector<int> pixels;
    std::vector<float> noise_lut;

    [[nodiscard]] DeviceNoiseVector ToDeviceVector() const;
};

inline DeviceNoiseVector NoiseVector::ToDeviceVector() const {
    cuda::KernelArray<int> d_pixels{nullptr, pixels.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_pixels.array, sizeof(int) * pixels.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_pixels.array, pixels.data(), sizeof(int) * pixels.size(), cudaMemcpyHostToDevice));

    cuda::KernelArray<float> d_lut{nullptr, noise_lut.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_lut.array, sizeof(float) * noise_lut.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_lut.array, noise_lut.data(), sizeof(float) * noise_lut.size(), cudaMemcpyHostToDevice));

    return {time_mjd, line, d_pixels, d_lut};
}
}  // namespace alus::s1tbx