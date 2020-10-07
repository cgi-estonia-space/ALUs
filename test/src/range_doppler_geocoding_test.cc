#include "range_doppler_geocoding_test.cuh"
#include "CudaFriendlyObject.h"
#include "cuda_util.hpp"
#include "dataset.h"
#include "gmock/gmock.h"
#include "resampling.h"
#include "tests_common.hpp"

using namespace alus;
using namespace alus::tests;
using namespace alus::snapengine::resampling;

namespace {

class RangeDopplerGeocodingTester : public cuda::CudaFriendlyObject {
   private:
    void CopyResamplingRasterToDevice() {
        // Prepare resampling raster tile
        CHECK_CUDA_ERR(cudaMalloc(&d_resampling_raster_tile_, sizeof(resampling_raster_tile_)));
        CHECK_CUDA_ERR(cudaMemcpy(d_resampling_raster_tile_,
                                  &resampling_raster_tile_,
                                  sizeof(resampling_raster_tile_),
                                  cudaMemcpyHostToDevice));
        resampling_raster_.source_tile_i = d_resampling_raster_tile_;

        CHECK_CUDA_ERR(cudaMalloc(&d_resampling_raster_, sizeof(resampling_raster_)));
        CHECK_CUDA_ERR(
            cudaMemcpy(d_resampling_raster_, &resampling_raster_, sizeof(resampling_raster_), cudaMemcpyHostToDevice));
    }

    void CopyIndexToDevice() {
        CHECK_CUDA_ERR(cudaMalloc((void **)&d_index_i_, sizeof(double) * 2));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_i_, &index_i_, sizeof(double) * 2, cudaMemcpyHostToDevice));
        resampling_index_.i = d_index_i_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_j_, sizeof(double) * 2));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_j_, &index_j_, sizeof(double) * 2, cudaMemcpyHostToDevice));
        resampling_index_.j = d_index_j_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_ki_, sizeof(double) * 1));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_ki_, &index_ki_, sizeof(double) * 1, cudaMemcpyHostToDevice));
        resampling_index_.ki = d_index_ki_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_kj_, sizeof(double) * 2));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_kj_, &index_kj_, sizeof(double) * 1, cudaMemcpyHostToDevice));
        resampling_index_.kj = d_index_kj_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_, sizeof(resampling_index_)));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_, &resampling_index_, sizeof(resampling_index_), cudaMemcpyHostToDevice));
    }

    void CopySourceTileToDevice() {
        long size = sizeof(double) * dataset_.GetDataBuffer().size();
        CHECK_CUDA_ERR(cudaMalloc(&d_data_buffer_, size));
        CHECK_CUDA_ERR(cudaMemcpy(d_data_buffer_, tile_data_.source_tile->data_buffer, size, cudaMemcpyHostToDevice));
        source_tile_.data_buffer = d_data_buffer_;

        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile_, sizeof(source_tile_)));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile_, &source_tile_, sizeof(source_tile_), cudaMemcpyHostToDevice));
    }

    void CopyTileDataToDevice() {
        tile_data_.image_resampling_index = d_index_;
        tile_data_.source_tile = d_source_tile_;
        tile_data_.resampling_raster = d_resampling_raster_;
        CHECK_CUDA_ERR(cudaMalloc(&d_tile_data_, sizeof(tile_data_)));
        CHECK_CUDA_ERR(cudaMemcpy(d_tile_data_, &tile_data_, sizeof(tile_data_), cudaMemcpyHostToDevice));
    }

   public:
    double azimuth_index_ = 1317.8823445919936;
    double range_index_ = 2097.633636393322;
    int source_image_width_ = 23278;
    int source_image_height_ = 1500;
    int margin_ = 1;
    int sub_swath_index_ = -1;
    double index_i_[2]{2097, 2098};
    double index_j_[2]{1317, 1318};
    double index_ki_[1]{0.6336363933219218};
    double index_kj_[1]{0.8823445919933874};

    Dataset dataset_{COH_1_TIF};

    Tile resampling_raster_tile_;
    ResamplingIndex resampling_index_{
        2098.133636393322, 1318.3823445919936, 23278, 1500, 2098, 1318, index_i_, index_j_, index_ki_, index_kj_};
    Tile source_tile_{0, 0, 23278, 1500, false, false, nullptr};
    ResamplingRaster resampling_raster_{
        2096.0041900186766, 1317.9935169605756, 0, 0, 0, nullptr, &resampling_raster_tile_};
    TileData tile_data_{&resampling_raster_, &source_tile_, &resampling_index_};

    double result_;

    // Devices variables
    Tile *d_resampling_raster_tile_;
    ResamplingRaster *d_resampling_raster_;
    ResamplingIndex *d_index_;
    double *d_index_i_;
    double *d_index_j_;
    double *d_index_ki_;
    double *d_index_kj_;
    double *d_data_buffer_;
    Tile *d_source_tile_;
    TileData *d_tile_data_;
    int *d_sub_swath_index_;
    double *d_result_;

    RangeDopplerGeocodingTester(double azimuth,
                                double range,
                                int margin,
                                int source_image_width,
                                int source_image_height,
                                double index_i[2],
                                double index_j[2],
                                double index_ki[1],
                                double index_kj[1],
                                double index_x,
                                double index_y,
                                int index_width,
                                int index_height,
                                double index_i0,
                                double index_j0,
                                double resampling_range,
                                double resampling_azimuth,
                                int resampling_sub_swath,
                                int resampling_min_x,
                                int resampling_min_y) {
        azimuth_index_ = azimuth;
        range_index_ = range;
        margin_ = margin;
        source_image_width_ = source_image_width;
        source_image_height_ = source_image_height;
        index_i_[0] = index_i[0];
        index_i_[1] = index_i[1];
        index_j_[0] = index_j[0];
        index_j_[1] = index_j[1];
        index_ki_[0] = index_ki[0];
        index_kj_[0] = index_kj[0];
        resampling_index_ = {
            index_x, index_y, index_width, index_height, index_i0, index_j0, index_i_, index_j_, index_ki_, index_kj_};
        source_tile_ = {0, 0, source_image_width, source_image_height, false, false, nullptr};
        resampling_raster_ = {resampling_range,
                              resampling_azimuth,
                              resampling_sub_swath,
                              resampling_min_x,
                              resampling_min_y,
                              nullptr,
                              &resampling_raster_tile_};
        tile_data_ = {&resampling_raster_, &source_tile_, &resampling_index_};

        dataset_.LoadRasterBand(1);
        tile_data_.source_tile->data_buffer = const_cast<double *>(dataset_.GetDataBuffer().data());
    }

    RangeDopplerGeocodingTester() {
        dataset_.LoadRasterBand(1);
        tile_data_.source_tile->data_buffer = const_cast<double *>(dataset_.GetDataBuffer().data());
    }

    ~RangeDopplerGeocodingTester() { this->DeviceFree(); }

    void HostToDevice() override {
        CopyIndexToDevice();
        CopyResamplingRasterToDevice();
        CopySourceTileToDevice();
        CopyTileDataToDevice();

        CHECK_CUDA_ERR(cudaMalloc(&d_sub_swath_index_, sizeof(double)));
        CHECK_CUDA_ERR(cudaMemcpy(d_sub_swath_index_, &sub_swath_index_, sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMalloc(&d_result_, sizeof(double)));
    }

    void DeviceToHost() override {
        CHECK_CUDA_ERR(cudaMemcpy(&result_, d_result_, sizeof(double), cudaMemcpyDeviceToHost));
    }
    void DeviceFree() override {
        cudaFree(d_resampling_raster_);
        cudaFree(d_resampling_raster_tile_);
        cudaFree(d_index_j_);
        cudaFree(d_index_ki_);
        cudaFree(d_index_kj_);
        cudaFree(d_index_);
        cudaFree(d_index_i_);
        cudaFree(d_source_tile_);
        cudaFree(d_data_buffer_);
        cudaFree(d_tile_data_);
        cudaFree(d_sub_swath_index_);
        cudaFree(d_result_);
    }
};

TEST(RangeDopplerGeocoding, DISABLED_GetPixelValue) {
    const double EXPECTED = 0.5128202235454891;
    RangeDopplerGeocodingTester tester;
    tester.HostToDevice();

    LaunchGetPixelValue(1,
                        1,
                        tester.azimuth_index_,
                        tester.range_index_,
                        tester.margin_,
                        tester.source_image_width_,
                        tester.source_image_height_,
                        tester.d_tile_data_,
                        tester.d_data_buffer_,
                        tester.d_sub_swath_index_,
                        tester.d_result_);

    tester.DeviceToHost();
    ASSERT_NEAR(EXPECTED, tester.result_, 1e-9);
}

TEST(RangeDopplerGeocoding, getPixelValueZeroCase) {
    const double EXPECTED = 0.0;
    double index_i[2]{1315.0, 1316.0};
    double index_j[2]{958.0, 959.0};
    double index_ki[1]{0.8210431430734388};
    double index_kj[1]{0.8210431430734388};
    RangeDopplerGeocodingTester tester(958.2326312466499,
                                       1315.8210431430734,
                                       1,
                                       23278,
                                       1500,
                                       index_i,
                                       index_j,
                                       index_ki,
                                       index_kj,
                                       0.8210431430734388,
                                       0.8210431430734388,
                                       23278,
                                       1500,
                                       1316.0,
                                       958.0,
                                       1314.2597546317913,
                                       958.3443380977734,
                                       0,
                                       858,
                                       734);
    tester.HostToDevice();

    LaunchGetPixelValue(1,
                        1,
                        tester.azimuth_index_,
                        tester.range_index_,
                        tester.margin_,
                        tester.source_image_width_,
                        tester.source_image_height_,
                        tester.d_tile_data_,
                        tester.d_data_buffer_,
                        tester.d_sub_swath_index_,
                        tester.d_result_);

    tester.DeviceToHost();
    ASSERT_DOUBLE_EQ(EXPECTED, tester.result_);
}
}  // namespace