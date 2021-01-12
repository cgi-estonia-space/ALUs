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
#include "range_doppler_geocoding.cuh"
#include "range_doppler_geocoding_test.cuh"

namespace alus {
namespace tests {
__global__ void RangeDopplerGeocodingTester(double azimuth_index,
                                            double range_index,
                                            int margin,
                                            int source_image_width,
                                            int source_image_height,
                                            snapengine::resampling::ResamplingRaster resampling_raster,
                                            snapengine::resampling::ResamplingIndex resampling_index,
                                            int &sub_swath_index,
                                            double *d_result) {
    *d_result = terraincorrection::rangedopplergeocoding::GetPixelValue(azimuth_index,
                                                                              range_index,
                                                                              margin,
                                                                              source_image_width,
                                                                              source_image_height,
                                                                              resampling_raster,
                                                                              resampling_index,
                                                                              sub_swath_index);
};

cudaError_t LaunchGetPixelValue(dim3 grid_size,
                                dim3 block_size,
                                double azimuth_index,
                                double range_index,
                                int margin,
                                int source_image_width,
                                int source_image_height,
                                snapengine::resampling::ResamplingRaster resampling_raster,
                                snapengine::resampling::ResamplingIndex resampling_index,
                                int *sub_swath_index,
                                double *d_result) {
    RangeDopplerGeocodingTester<<<grid_size, block_size>>>(azimuth_index,
                                                           range_index,
                                                           margin,
                                                           source_image_width,
                                                           source_image_height,
                                                           resampling_raster,
                                                           resampling_index,
                                                           *sub_swath_index,
                                                           d_result);

    cudaDeviceSynchronize();

    return cudaGetLastError();
}
}  // namespace tests
}  // namespace alus