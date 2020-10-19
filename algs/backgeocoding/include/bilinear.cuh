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
#include <cuda_runtime.h>

namespace alus {
namespace backgeocoding {

struct BilinearParams {
    int point_width;
    int point_height;
    int demod_width;
    int demod_height;
    int start_x;
    int start_y;

    int scanline_offset;
    int scanline_stride;
    int min_x;
    int min_y;

    int rectangle_x;
    int rectangle_y;
    bool disable_reramp;
    int subswath_start;
    int subswath_end;

    double no_data_value;
};

cudaError_t LaunchBilinearInterpolation(double *x_pixels,
                                        double *y_pixels,
                                        double *demod_phase,
                                        double *demod_i,
                                        double *demod_q,
                                        BilinearParams params,
                                        float *results_i,
                                        float *results_q);

}  // namespace backgeocoding
}  // namespace alus
