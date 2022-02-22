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

namespace alus::featurextractiongabor {

void LaunchPatchMeanReduction(const float* d_src, float* d_result, int patch_size, int edge_size, int n_x_patches,
                              int n_y_patches, cudaStream_t stream = nullptr);

void LaunchPatchStdDevReduction(const float* d_src, const float* d_patch_means, float* d_result, int patch_size,
                                int edge_size, int n_x_patches, int n_y_patches, cudaStream_t stream = nullptr);
}  // namespace alus::featurextractiongabor
