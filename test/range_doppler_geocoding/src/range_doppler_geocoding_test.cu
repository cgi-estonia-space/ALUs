#include "range_doppler_geocoding.cuh"
#include "range_doppler_geocoding_test.cuh"

namespace alus {
namespace tests {
__global__ void RangeDopplerGeocodingTester(double azimuth_index,
                                            double range_index,
                                            int margin,
                                            int source_image_width,
                                            int source_image_height,
                                            alus::snapengine::resampling::TileData *tile_data,
                                            double *band_data_buffer,
                                            int &sub_swath_index,
                                            double *d_result) {
    *d_result = alus::terraincorrection::rangedopplergeocoding::GetPixelValue(azimuth_index,
                                                                              range_index,
                                                                              margin,
                                                                              source_image_width,
                                                                              source_image_height,
                                                                              tile_data,
                                                                              band_data_buffer,
                                                                              sub_swath_index);
};

cudaError_t LaunchGetPixelValue(dim3 grid_size,
                                dim3 block_size,
                                double azimuth_index,
                                double range_index,
                                int margin,
                                int source_image_width,
                                int source_image_height,
                                alus::snapengine::resampling::TileData *tile_data,
                                double *band_data_buffer,
                                int *sub_swath_index,
                                double *d_result) {
    RangeDopplerGeocodingTester<<<grid_size, block_size>>>(azimuth_index,
                                                           range_index,
                                                           margin,
                                                           source_image_width,
                                                           source_image_height,
                                                           tile_data,
                                                           band_data_buffer,
                                                           *sub_swath_index,
                                                           d_result);

    return cudaGetLastError();
}
}  // namespace tests
}  // namespace alus