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

#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_util.h"
#include "kernel_array.h"

namespace alus::s1tbx {

/**
 * NoiseAzimuthVector that is to be used on CUDA devices.
 */
struct DeviceNoiseAzimuthVector {
    const int first_azimuth_line;
    const int first_range_sample;
    const int last_azimuth_line;
    const int last_range_sample;
    cuda::KernelArray<int> lines;
    cuda::KernelArray<float> noise_azimuth_lut;
};

/**
 * Port of SNAP's NoiseAzimuthVector. Original class location: Sentinel1Utils.java.
 */
struct NoiseAzimuthVector {
    const std::string swath;
    const int first_azimuth_line;
    const int first_range_sample;
    const int last_azimuth_line;
    const int last_range_sample;
    const std::vector<int> lines;
    const std::vector<float> noise_azimuth_lut;

    [[nodiscard]] DeviceNoiseAzimuthVector ToDeviceVector() const;
};

inline DeviceNoiseAzimuthVector NoiseAzimuthVector::ToDeviceVector() const {
    cuda::KernelArray<int> d_lines{nullptr, lines.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_lines.array, sizeof(int) * lines.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_lines.array, lines.data(), sizeof(int) * lines.size(), cudaMemcpyHostToDevice));

    cuda::KernelArray<float> d_noise_azimuth_lut{nullptr, noise_azimuth_lut.size()};
    CHECK_CUDA_ERR(cudaMalloc(&d_noise_azimuth_lut.array, sizeof(int) * noise_azimuth_lut.size()));
    CHECK_CUDA_ERR(cudaMemcpy(d_noise_azimuth_lut.array, noise_azimuth_lut.data(),
                              sizeof(int) * noise_azimuth_lut.size(), cudaMemcpyHostToDevice));

    return {first_azimuth_line, first_range_sample, last_azimuth_line, last_range_sample, d_lines, d_noise_azimuth_lut};
}
}  // namespace alus::s1tbx