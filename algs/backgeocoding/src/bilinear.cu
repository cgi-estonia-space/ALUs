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
#include "bilinear.cuh"

#include "bilinear_interpolation.cuh"
#include "pointer_holders.h"

/**
 * The contents of this file refer to BackGeocodingOp.performInterpolation method on SNAP's code s1tbx module.
 */

namespace alus {
namespace backgeocoding{

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
                                      int *int_params,
                                      double double_params,
                                      float *results_i,
                                      float *results_q) {
    double index_i[2];
    double index_j[2];
    double index_ki[1];
    double index_kj[1];
    snapengine::resampling::ResamplingIndex index {0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    const int raster_width = 2;
    const int point_width = int_params[0];
    const int point_height = int_params[1];
    const int demod_width = int_params[2];
    const int demod_height = int_params[3];
    const int start_x = int_params[4];
    const int start_y = int_params[5];
    const int rectangle_x = int_params[10];
    const int rectangle_y = int_params[11];
    const int disable_reramp = int_params[12];
    const int subswath_start = int_params[13];
    const int subswath_end = int_params[14];
    // TODO: this needs to come from tile information
    const int use_no_data_phase = 0;
    const int use_no_data_i = 0;
    const int use_no_data_q = 0;

    const double no_data_value = double_params;

    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const double x = x_pixels[(idy * point_width) + idx];
    const double y = y_pixels[(idy * point_width) + idx];
    double sample_phase = 0.0;
    double sample_i = 0.0;
    double sample_q = 0.0;
    double cos_phase = 0.0;
    double sin_phase = 0.0;
    double reramp_remod_i = 0.0;
    double reramp_remod_q = 0.0;

    PointerArray p_array;
    PointerHolder p_holder;
    p_array.array = &p_holder;
    p_holder.x = demod_width;
    p_holder.y = demod_height;

    // y stride + x index
    const int target_index =
        (start_x + idx) - (int_params[8] - (((start_y + idy) - int_params[9]) * int_params[7] + int_params[6]));

    if (idx < point_width && idy < point_height) {
        if ((x == no_data_value && y == no_data_value) || !(y >= subswath_start && y < subswath_end)) {
            results_i[(idy * point_width) + idx] = no_data_value;
            results_q[(idy * point_width) + idx] = no_data_value;
        } else {
            snapengine::bilinearinterpolation::ComputeIndex(x - rectangle_x + 0.5,
                                                            y - rectangle_y + 0.5,
                                                            demod_width,
                                                            demod_height, &index);
            p_holder.pointer = demod_phase;
            sample_phase = snapengine::bilinearinterpolation::Resample(
                &p_array, &index, raster_width, no_data_value, use_no_data_phase, GetSamples);
            p_holder.pointer = demod_i;
            sample_i = snapengine::bilinearinterpolation::Resample(
                &p_array, &index, raster_width, no_data_value, use_no_data_i, GetSamples);
            p_holder.pointer = demod_q;
            sample_q = snapengine::bilinearinterpolation::Resample(
                &p_array, &index, raster_width, no_data_value, use_no_data_q, GetSamples);

            if (!disable_reramp) {
                cos_phase = cos(sample_phase);
                sin_phase = sin(sample_phase);
                reramp_remod_i = sample_i * cos_phase + sample_q * sin_phase;
                reramp_remod_q = -sample_i * sin_phase + sample_q * cos_phase;
                results_i[target_index] = reramp_remod_i;
                results_q[target_index] = reramp_remod_q;
            } else {
                results_i[target_index] = sample_i;
                results_q[target_index] = sample_q;
            }
        }
    }
}

cudaError_t LaunchBilinearInterpolation(dim3 grid_size,
                                        dim3 block_size,
                                        double *x_pixels,
                                        double *y_pixels,
                                        double *demod_phase,
                                        double *demod_i,
                                        double *demod_q,
                                        int *int_params,
                                        double double_params,
                                        float *results_i,
                                        float *results_q) {

    BilinearInterpolation<<<grid_size, block_size>>>(
        x_pixels, y_pixels, demod_phase, demod_i, demod_q, int_params, double_params, results_i, results_q);
    return cudaGetLastError();
}

}  // namespace alus
}  // namespace alus
