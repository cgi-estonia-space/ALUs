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
#include "gmock/gmock.h"

#include "cuda_friendly_object.h"
#include "cuda_util.h"
#include "dataset.h"
#include "gdal_util.h"
#include "resampling.h"

#include "range_doppler_geocoding_test.cuh"

using namespace alus;
using namespace alus::tests;
using namespace alus::snapengine::resampling;

namespace {

class RangeDopplerGeocodingTest : public ::testing::Test {
   private:
    Tile *d_source_tile_;
    double *d_index_i_;
    double *d_index_j_;
    double *d_index_ki_;
    double *d_index_kj_;
    Dataset<double> dataset_{
        "./goods/S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.tif"};
    std::vector<double> tile_data_;
    float *d_tile_data_;

   protected:
    double range_index_;
    double azimuth_index_;
    int margin_{1};
    int source_image_width_{23278};
    int source_image_height_{1500};
    Tile source_tile_;
    ResamplingRaster raster_;
    ResamplingIndex index_;
    int sub_swath_index_;
    int *d_sub_swath_index_;

    void ReadTileData(const Rectangle &source_rectangle) {
        tile_data_.resize(source_rectangle.width * source_rectangle.height);
        CHECK_GDAL_ERROR(dataset_.GetGdalDataset()->GetRasterBand(1)->RasterIO(GF_Read,
                                                                               source_rectangle.x,
                                                                               source_rectangle.y,
                                                                               source_rectangle.width,
                                                                               source_rectangle.height,
                                                                               tile_data_.data(),
                                                                               source_rectangle.width,
                                                                               source_rectangle.height,
                                                                               GDALDataType::GDT_Float32,
                                                                               0,
                                                                               0));
    }

    void HostToDevice() {
        CHECK_CUDA_ERR(cudaMalloc(&d_tile_data_, sizeof(float ) * tile_data_.size()));
        CHECK_CUDA_ERR(
            cudaMemcpy(d_tile_data_, tile_data_.data(), sizeof(float) * tile_data_.size(), cudaMemcpyHostToDevice));
        source_tile_.data_buffer = d_tile_data_;

        CHECK_CUDA_ERR(cudaMalloc(&d_source_tile_, sizeof(Tile)));
        CHECK_CUDA_ERR(cudaMemcpy(d_source_tile_, &source_tile_, sizeof(Tile), cudaMemcpyHostToDevice));
        raster_.source_tile_i = d_source_tile_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_i_, sizeof(double) * 2));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_i_, index_.i, sizeof(double) * 2, cudaMemcpyHostToDevice));
        index_.i = d_index_i_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_j_, sizeof(double) * 2));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_j_, index_.j, sizeof(double) * 2, cudaMemcpyHostToDevice));
        index_.j = d_index_j_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_ki_, sizeof(double) * 1));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_ki_, index_.ki, sizeof(double), cudaMemcpyHostToDevice));
        index_.ki = d_index_ki_;

        CHECK_CUDA_ERR(cudaMalloc(&d_index_kj_, sizeof(double)));
        CHECK_CUDA_ERR(cudaMemcpy(d_index_kj_, index_.kj, sizeof(double), cudaMemcpyHostToDevice));
        index_.kj = d_index_kj_;

        CHECK_CUDA_ERR(cudaMalloc(&d_sub_swath_index_, sizeof(int)));
        CHECK_CUDA_ERR(cudaMemcpy(d_sub_swath_index_, &sub_swath_index_, sizeof(int), cudaMemcpyHostToDevice));
    }

   public:
    void DeviceFree() {
        cudaFree(d_source_tile_);
        cudaFree(d_index_i_);
        cudaFree(d_index_j_);
        cudaFree(d_index_ki_);
        cudaFree(d_index_kj_);
        cudaFree(d_tile_data_);
        cudaFree(d_sub_swath_index_);
    }

    virtual ~RangeDopplerGeocodingTest() { DeviceFree(); }
};

TEST_F(RangeDopplerGeocodingTest, GetPixelValueNoNewRectangleComputation) {
    const double expected_result = 0.42228986962026993;

    const Rectangle source_rectangle{16195, 1008, 1019, 454};

    this->range_index_ = 16478.63922364263;
    this->azimuth_index_ = 1459.017842614426;
    this->source_tile_ = {16195, 1008, 1019, 454, false, false, nullptr};
    this->raster_ = {16478.63922364263, 1459.017842614426, 0, source_rectangle, &source_tile_, true};

    double index_i[]{0, 0};
    double index_j[]{0, 0};
    double index_ki[]{0};
    double index_kj[]{0};
    this->index_ = {
        0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    sub_swath_index_ = 0;

    this->ReadTileData(source_rectangle);
    this->HostToDevice();

    double *d_result;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, sizeof(double)));

    CHECK_CUDA_ERR(LaunchGetPixelValue({1},
                                       {1},
                                       azimuth_index_,
                                       range_index_,
                                       margin_,
                                       source_image_width_,
                                       source_image_height_,
                                       raster_,
                                       index_,
                                       d_sub_swath_index_,
                                       d_result));

    double *computed_result;
    //    computed_result = (double *)malloc(sizeof(double));
    CHECK_CUDA_ERR(cudaMallocHost(&computed_result, sizeof(double)));
    CHECK_CUDA_ERR(cudaMemcpy(computed_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_THAT(*computed_result, ::testing::DoubleEq(expected_result));

    CHECK_CUDA_ERR(cudaFree(d_result));
    CHECK_CUDA_ERR(cudaFreeHost(computed_result));
}

TEST_F(RangeDopplerGeocodingTest, GetPixelValueZeroCaseAndNoNewRectangleComputation) {
    const double expected_result = 0.0;

    const Rectangle source_rectangle{16195, 1008, 1019, 454};

    this->range_index_ = 16658.41527141122;
    this->azimuth_index_ = 1448.4087841209837;
    this->source_tile_ = {16195, 1008, 1019, 454, false, false, nullptr};
    this->raster_ = {16658.41527141122, 1448.4087841209837, 0, source_rectangle, &source_tile_, true};

    double index_i[]{0, 0};
    double index_j[]{0, 0};
    double index_ki[]{0};
    double index_kj[]{0};
    this->index_ = {
        0, 0, 0, 0, 0, 0, index_i, index_j, index_ki, index_kj};

    sub_swath_index_ = 0;

    this->ReadTileData(source_rectangle);
    this->HostToDevice();

    double *d_result;
    CHECK_CUDA_ERR(cudaMalloc(&d_result, sizeof(double)));

    CHECK_CUDA_ERR(LaunchGetPixelValue({1},
                                       {1},
                                       azimuth_index_,
                                       range_index_,
                                       margin_,
                                       source_image_width_,
                                       source_image_height_,
                                       raster_,
                                       index_,
                                       d_sub_swath_index_,
                                       d_result));

    double *computed_result;
    CHECK_CUDA_ERR(cudaMallocHost(&computed_result, sizeof(double)));
    CHECK_CUDA_ERR(cudaMemcpy(computed_result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    EXPECT_THAT(*computed_result, ::testing::DoubleEq(expected_result));

    CHECK_CUDA_ERR(cudaFree(d_result));
    CHECK_CUDA_ERR(cudaFreeHost(computed_result));
}
}  // namespace