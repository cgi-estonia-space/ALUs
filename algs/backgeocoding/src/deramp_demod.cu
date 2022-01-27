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
#include "deramp_demod_computation.h"

#include "cuda_util.h"
#include "snap-engine-utilities/engine-utilities/eo/constants.h"

/**
 * The contents of this file refer to BackGeocodingOp.performDerampDemod method
 * with computeDerampDemodPhase merged into it for faster use. They are from s1tbx module.
 */

namespace alus {
namespace backgeocoding {

__global__ void DerampDemod(alus::Rectangle rectangle, int16_t* slave_i, int16_t* slave_q, double* demod_phase,
                            double* demod_i, double* demod_q, alus::s1tbx::DeviceSubswathInfo* sub_swath,
                            int s_burst_index) {
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const int global_index = rectangle.width * idy + idx;
    const int first_line_in_burst = s_burst_index * sub_swath->lines_per_burst;
    const int y = rectangle.y + idy;
    const int x = rectangle.x + idx;
    double ta, kt, deramp, demod;
    double value_i, value_q, value_phase, cos_phase, sin_phase;

    if (idx >= rectangle.width || idy >= rectangle.height) {
        return;
    }
    ta = (y - first_line_in_burst) * sub_swath->azimuth_time_interval;
    kt = sub_swath->device_doppler_rate[s_burst_index * sub_swath->doppler_size_y + x];
    deramp = -snapengine::eo::constants::PI * kt *
             pow(ta - sub_swath->device_reference_time[s_burst_index * sub_swath->doppler_size_y + x], 2);
    demod = -snapengine::eo::constants::TWO_PI *
            sub_swath->device_doppler_centroid[s_burst_index * sub_swath->doppler_size_y + x] * ta;
    value_phase = deramp + demod;

    demod_phase[global_index] = value_phase;

    value_i = static_cast<double>(slave_i[global_index]);
    value_q = static_cast<double>(slave_q[global_index]);

    sincos(value_phase, &sin_phase, &cos_phase);
    demod_i[global_index] = value_i * cos_phase - value_q * sin_phase;
    demod_q[global_index] = value_i * sin_phase + value_q * cos_phase;
}

cudaError_t LaunchDerampDemod(alus::Rectangle rectangle, int16_t* slave_i, int16_t* slave_q, double* demod_phase,
                              double* demod_i, double* demod_q, alus::s1tbx::DeviceSubswathInfo* sub_swath,
                              int s_burst_index) {
    dim3 block_size(24, 24);
    dim3 grid_size(cuda::GetGridDim(block_size.x, rectangle.width), cuda::GetGridDim(block_size.y, rectangle.height));

    DerampDemod<<<grid_size, block_size>>>(rectangle, slave_i, slave_q, demod_phase, demod_i, demod_q, sub_swath,
                                           s_burst_index);
    return cudaGetLastError();
}

}  // namespace backgeocoding
}  // namespace alus
