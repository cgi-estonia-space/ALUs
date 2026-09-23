/** Full-resolution pair-specific ETAD correction generation. SPDX-License-Identifier: GPL-3.0-or-later */
#include "etad_correction.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "backgeocoding_constants.h"
#include "cuda_copies.h"
#include "cuda_util.h"
#include "etad_computation.cuh"

namespace alus::backgeocoding {
namespace {

__global__ void ComputeEtadCorrection(EtadBurstDescriptor reference, EtadBurstDescriptor secondary,
                                      Rectangle target_area, const double* secondary_x, const double* secondary_y,
                                      float* output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= target_area.width || y >= target_area.height) {
        return;
    }

    const size_t index = static_cast<size_t>(y) * target_area.width + x;
    const double mapped_x = secondary_x[index];
    const double mapped_y = secondary_y[index];
    constexpr float INVALID = std::numeric_limits<float>::quiet_NaN();
    if (!isfinite(mapped_x) || !isfinite(mapped_y) ||
        (mapped_x == INVALID_INDEX && mapped_y == INVALID_INDEX)) {
        output[index] = INVALID;
        return;
    }

    const double reference_line =
        static_cast<double>(target_area.y + y) - static_cast<double>(reference.first_output_line);
    const double reference_sample = static_cast<double>(target_area.x + x);
    const double reference_azimuth = reference.slc_geometry.first_line_time +
                                     reference_line * reference.slc_geometry.azimuth_time_interval;
    const double reference_range = reference.slc_geometry.first_range_time +
                                   reference_sample * reference.slc_geometry.range_time_interval;

    const double secondary_line = mapped_y - static_cast<double>(secondary.first_output_line);
    if (secondary_line < 0 || secondary_line >= static_cast<double>(secondary.slc_geometry.lines) || mapped_x < 0 ||
        mapped_x >= static_cast<double>(secondary.slc_geometry.samples)) {
        output[index] = INVALID;
        return;
    }
    const double secondary_azimuth = secondary.slc_geometry.first_line_time +
                                     secondary_line * secondary.slc_geometry.azimuth_time_interval;
    const double secondary_range =
        secondary.slc_geometry.first_range_time + mapped_x * secondary.slc_geometry.range_time_interval;

    const double reference_phase =
        s1tbx::etad::SampleGrid(reference.phase, reference.etad_geometry, reference_azimuth, reference_range);
    const double reference_height =
        s1tbx::etad::SampleGrid(reference.height, reference.etad_geometry, reference_azimuth, reference_range);
    const double secondary_phase =
        s1tbx::etad::SampleGrid(secondary.phase, secondary.etad_geometry, secondary_azimuth, secondary_range);
    const double secondary_height =
        s1tbx::etad::SampleGrid(secondary.height, secondary.etad_geometry, secondary_azimuth, secondary_range);
    const double secondary_gradient =
        s1tbx::etad::SampleGrid(secondary.gradient, secondary.etad_geometry, secondary_azimuth, secondary_range);
    const double correction = s1tbx::etad::DifferentialPhase(reference_phase, secondary_phase, reference_height,
                                                             secondary_height, secondary_gradient);
    output[index] = isfinite(correction) ? static_cast<float>(correction) : INVALID;
}

}  // namespace

EtadCorrection::DeviceBurst EtadCorrection::Upload(const s1tbx::etad::PreparedBurst& burst) {
    DeviceBurst device;
    device.phase = cuda::CudaPtr<double>(burst.layers.phase.values.size());
    device.height = cuda::CudaPtr<double>(burst.layers.height.values.size());
    device.gradient = cuda::CudaPtr<double>(burst.layers.gradient.values.size());
    cuda::CopyArrayH2D(device.phase.Get(), burst.layers.phase.values.data(), burst.layers.phase.values.size());
    cuda::CopyArrayH2D(device.height.Get(), burst.layers.height.values.data(), burst.layers.height.values.size());
    cuda::CopyArrayH2D(device.gradient.Get(), burst.layers.gradient.values.data(), burst.layers.gradient.values.size());
    device.descriptor = {
        burst.slc_geometry,
        burst.etad_geometry,
        {device.phase.Get(), burst.layers.phase.rows, burst.layers.phase.columns},
        {device.height.Get(), burst.layers.height.rows, burst.layers.height.columns},
        {device.gradient.Get(), burst.layers.gradient.rows, burst.layers.gradient.columns},
        burst.association.first_output_line,
    };
    return device;
}

EtadCorrection::EtadCorrection(const s1tbx::etad::PreparedPair& pair) {
    if (pair.reference == nullptr || pair.secondary == nullptr || pair.reference->bursts.empty() ||
        pair.secondary->bursts.empty()) {
        throw std::invalid_argument("ETAD correction requires prepared reference and secondary bursts");
    }
    reference_.reserve(pair.reference->bursts.size());
    for (size_t i = 0; i < pair.reference->bursts.size(); ++i) {
        const auto& burst = pair.reference->bursts.at(i);
        if (burst.association.split_burst_index != i) {
            throw std::invalid_argument("reference ETAD bursts are not in split-product order");
        }
        reference_.push_back(Upload(burst));
    }
    secondary_.reserve(pair.secondary->bursts.size());
    for (size_t i = 0; i < pair.secondary->bursts.size(); ++i) {
        const auto& burst = pair.secondary->bursts.at(i);
        if (burst.association.split_burst_index != i) {
            throw std::invalid_argument("secondary ETAD bursts are not in split-product order");
        }
        secondary_.push_back(Upload(burst));
    }
}

void EtadCorrection::Compute(int reference_burst_index, int secondary_burst_index, Rectangle target_area,
                             const double* secondary_x, const double* secondary_y, float* output) const {
    if (reference_burst_index < 0 || static_cast<size_t>(reference_burst_index) >= reference_.size() ||
        secondary_burst_index < 0 || static_cast<size_t>(secondary_burst_index) >= secondary_.size()) {
        throw std::out_of_range("ETAD burst index is outside prepared acquisition");
    }
    CHECK_CUDA_ERR(LaunchEtadCorrection(reference_.at(reference_burst_index).descriptor,
                                        secondary_.at(secondary_burst_index).descriptor, target_area, secondary_x,
                                        secondary_y, output));
}

cudaError_t LaunchEtadCorrection(EtadBurstDescriptor reference, EtadBurstDescriptor secondary,
                                 Rectangle target_area, const double* secondary_x, const double* secondary_y,
                                 float* output) {
    constexpr dim3 BLOCK(32, 8);
    const dim3 grid(cuda::GetGridDim(BLOCK.x, target_area.width), cuda::GetGridDim(BLOCK.y, target_area.height));
    ComputeEtadCorrection<<<grid, BLOCK>>>(reference, secondary, target_area, secondary_x, secondary_y, output);
    return cudaGetLastError();
}

}  // namespace alus::backgeocoding
