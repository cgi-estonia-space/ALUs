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
#include "bilinear_computation.h"

#include "cuda_util.h"
#include "bilinear_interpolation.cuh"
#include "pointer_holders.h"
#include "backgeocoding_constants.h"

/**
 * The contents of this file refer to BackGeocodingOp.performInterpolation method on SNAP's code s1tbx module.
 */

namespace alus {
namespace backgeocoding {

inline __device__ int GetSamples(
    PointerArray *tiles, int *x, int *y, double *samples, int width, int height, double no_value, int use_no_data) {
    double *values = (double *)tiles->array[0].pointer;
    const int value_width = tiles->array[0].x;
    int i = 0, j = 0, is_valid = 1;
    while (i < height) {
        j = 0;
        while (j < width) {
            samples[i * width + j] = values[value_width * y[i] + x[j]];
            if (use_no_data) {
                if (no_value == samples[i * width + j]) {
                    is_valid = 0;
                }
            }
            ++j;
        }
        ++i;
    }
    return is_valid;
}

__global__ void BilinearInterpolation(double *x_pixels,
                                      double *y_pixels,
                                      double *demod_phase,
                                      double *demod_i,
                                      double *demod_q,
                                      BilinearParams params,
                                      float *results_i,
                                      float *results_q) {
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];
    snapengine::resampling::ResamplingIndex index{0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    const int raster_width = 2;
    // TODO: this needs to come from tile information
    const int use_no_data_phase = 0;
    const int use_no_data_i = 0;
    const int use_no_data_q = 0;

    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    double sample_phase = 0.0;
    double sample_i = 0.0;
    double sample_q = 0.0;
    double cos_phase = 0.0;
    double sin_phase = 0.0;
    double reramp_remod_i = 0.0;
    double reramp_remod_q = 0.0;

    const int thread_data_index = idy * params.point_width + idx;

    PointerArray p_array;
    PointerHolder p_holder;
    p_array.array = &p_holder;
    p_holder.x = params.demod_width;
    p_holder.y = params.demod_height;

    if (idx >= params.point_width || idy >= params.point_height) {
        return;
    }
    const double x = x_pixels[thread_data_index];
    const double y = y_pixels[thread_data_index];

    if ((x == INVALID_INDEX && y == INVALID_INDEX) ||
        !(y >= params.subswath_start && y < params.subswath_end)) {
        results_i[thread_data_index] = params.no_data_value;
        results_q[thread_data_index] = params.no_data_value;
    } else {
        snapengine::bilinearinterpolation::ComputeIndex(x - params.rectangle_x + 0.5,
                                                        y - params.rectangle_y + 0.5,
                                                        params.demod_width,
                                                        params.demod_height,
                                                        &index);
        p_holder.pointer = demod_phase;
        sample_phase = snapengine::bilinearinterpolation::Resample(
            &p_array, &index, raster_width, params.no_data_value, use_no_data_phase, GetSamples);
        p_holder.pointer = demod_i;
        sample_i = snapengine::bilinearinterpolation::Resample(
            &p_array, &index, raster_width, params.no_data_value, use_no_data_i, GetSamples);
        p_holder.pointer = demod_q;
        sample_q = snapengine::bilinearinterpolation::Resample(
            &p_array, &index, raster_width, params.no_data_value, use_no_data_q, GetSamples);

        if (!params.disable_reramp) {
            cos_phase = cos(sample_phase);
            sin_phase = sin(sample_phase);
            reramp_remod_i = sample_i * cos_phase + sample_q * sin_phase;
            reramp_remod_q = -sample_i * sin_phase + sample_q * cos_phase;
            results_i[thread_data_index] = reramp_remod_i;
            results_q[thread_data_index] = reramp_remod_q;
        } else {
            results_i[thread_data_index] = sample_i;
            results_q[thread_data_index] = sample_q;
        }
    }

}

cudaError_t LaunchBilinearInterpolation(double *x_pixels,
                                        double *y_pixels,
                                        double *demod_phase,
                                        double *demod_i,
                                        double *demod_q,
                                        BilinearParams params,
                                        float *results_i,
                                        float *results_q) {
    dim3 block_size(24, 24);
    dim3 grid_size(cuda::GetGridDim(block_size.x, params.point_width),
                   cuda::GetGridDim(block_size.y, params.point_height));

    BilinearInterpolation<<<grid_size, block_size>>>(
        x_pixels, y_pixels, demod_phase, demod_i, demod_q, params, results_i, results_q);
    return cudaGetLastError();
}

}  // namespace backgeocoding
}  // namespace alus
