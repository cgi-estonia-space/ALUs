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
#include <fstream>
#include <vector>

#include "gmock/gmock.h"

#include "CudaFriendlyObject.h"
#include "comparators.h"
#include "cuda_util.hpp"
#include "tests_common.hpp"

#include "backgeocoding_constants.h"
#include "delaunay_triangulator.h"
#include "triangular_interpolation.cuh"

using namespace alus::tests;

namespace {

class TriangularInterpolationTester : public alus::cuda::CudaFriendlyObject {
   private:
    static const std::string az_rg_data_file_;
    static const std::string lats_lons_file_;
    static const std::string arrays_file_;
    static const std::string triangles_data_file_;

   public:
    static constexpr double RG_AZ_RATIO{0.16742323135844578};
    static constexpr double INVALID_INDEX{-9999.0};

    double *device_master_az_ = nullptr, *device_master_rg_ = nullptr;
    double *device_slave_az_ = nullptr, *device_slave_rg_ = nullptr;
    double *device_lat_ = nullptr, *device_lon_ = nullptr;
    double *device_rg_array_ = nullptr, *device_az_array_ = nullptr;
    double *device_lat_array_ = nullptr, *device_lon_array_ = nullptr;
    alus::delaunay::DelaunayTriangle2D *device_triangles_ = nullptr;

    std::vector<double> master_az_;
    std::vector<double> master_rg_;
    std::vector<double> slave_az_;
    std::vector<double> slave_rg_;
    std::vector<double> lats_;
    std::vector<double> lons_;

    std::vector<double> rg_array_;
    std::vector<double> az_array_;
    std::vector<double> lat_array_;
    std::vector<double> lon_array_;

    std::vector<double> results_rg_array_;
    std::vector<double> results_az_array_;
    std::vector<double> results_lat_array_;
    std::vector<double> results_lon_array_;

    std::vector<alus::delaunay::DelaunayTriangle2D> triangles_;

    size_t az_rg_width_, az_rg_height_, arr_width_, arr_height_, triangle_size_;

    TriangularInterpolationTester() { ReadTestData(); }
    ~TriangularInterpolationTester() { DeviceFree(); }

    void ReadTestData() {
        double temp_index;
        alus::delaunay::DelaunayTriangle2D temp_triangle;
        std::ifstream rg_az_stream(az_rg_data_file_);
        std::ifstream lats_lons_stream(lats_lons_file_);

        if (!rg_az_stream.is_open()) {
            throw std::ios::failure("masterSlaveAzRgData.txt is not open.");
        }
        if (!lats_lons_stream.is_open()) {
            throw std::ios::failure("pixelposLatsLons.txt is not open.");
        }

        rg_az_stream >> az_rg_width_ >> az_rg_height_;
        int coord_size = az_rg_width_ * az_rg_height_;

        master_az_.resize(coord_size);
        master_rg_.resize(coord_size);
        slave_rg_.resize(coord_size);
        slave_az_.resize(coord_size);
        lats_.resize(coord_size);
        lons_.resize(coord_size);

        for (int i = 0; i < coord_size; i++) {
            rg_az_stream >> master_az_.at(i) >> master_rg_.at(i) >> slave_az_.at(i) >> slave_rg_.at(i);
        }
        lats_lons_stream >> az_rg_width_ >> az_rg_height_;
        for (int i = 0; i < coord_size; i++) {
            lats_lons_stream >> lats_.at(i) >> lons_.at(i);
        }

        rg_az_stream.close();
        lats_lons_stream.close();

        std::ifstream arrays_stream(arrays_file_);

        if (!arrays_stream.is_open()) {
            throw std::ios::failure("pixelposArrays.txt is not open.");
        }

        arrays_stream >> arr_width_ >> arr_height_;
        int arr_size = arr_width_ * arr_height_;

        rg_array_.resize(arr_size);
        az_array_.resize(arr_size);
        lat_array_.resize(arr_size);
        lon_array_.resize(arr_size);

        results_rg_array_.resize(arr_size);
        results_az_array_.resize(arr_size);
        results_lat_array_.resize(arr_size);
        results_lon_array_.resize(arr_size);

        for (int i = 0; i < arr_size; i++) {
            arrays_stream >> az_array_.at(i) >> rg_array_.at(i) >> lat_array_.at(i) >> lon_array_.at(i);
        }

        arrays_stream.close();

        std::ifstream triangles_stream(this->triangles_data_file_);
        if (!triangles_stream.is_open()) {
            throw std::ios::failure("masterTrianglesTestData.txt is not open.");
        }
        triangles_stream >> triangle_size_;
        this->triangles_.resize(triangle_size_);

        // TODO: Seeing anything familiar? Figure out how to put this together with delaunay_test.cc once there is time.
        for (size_t i = 0; i < triangle_size_; i++) {
            triangles_stream >> temp_triangle.ax >> temp_triangle.ay >> temp_index;
            temp_index += 0.001;  // fixing a possible float inaccuracy
            temp_triangle.a_index = (int)temp_index;

            triangles_stream >> temp_triangle.bx >> temp_triangle.by >> temp_index;
            temp_index += 0.001;  // fixing a possible float inaccuracy
            temp_triangle.b_index = (int)temp_index;

            triangles_stream >> temp_triangle.cx >> temp_triangle.cy >> temp_index;
            temp_index += 0.001;  // fixing a possible float inaccuracy
            temp_triangle.c_index = (int)temp_index;

            triangles_.at(i) = temp_triangle;
        }

        triangles_stream.close();
    }

    void HostToDevice() {
        size_t array_size = arr_width_ * arr_height_;
        size_t az_rg_size = az_rg_width_ * az_rg_height_;

        CHECK_CUDA_ERR(cudaMalloc((void **)&device_rg_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_az_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_lat_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_lon_array_, array_size * sizeof(double)));

        CHECK_CUDA_ERR(cudaMalloc((void **)&device_master_az_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_master_rg_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_slave_az_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_slave_rg_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_lat_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void **)&device_lon_, az_rg_size * sizeof(double)));

        CHECK_CUDA_ERR(
            cudaMalloc((void **)&device_triangles_, triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2D)));

        CHECK_CUDA_ERR(
            cudaMemcpy(device_master_az_, master_az_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(device_master_rg_, master_rg_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(device_slave_az_, slave_az_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(
            cudaMemcpy(device_slave_rg_, slave_rg_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(device_lat_, lats_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(device_lon_, lons_.data(), az_rg_size * sizeof(double), cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMemcpy(device_triangles_,
                                  triangles_.data(),
                                  triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2D),
                                  cudaMemcpyHostToDevice));
    }
    void DeviceToHost() {
        size_t array_size = arr_width_ * arr_height_;

        CHECK_CUDA_ERR(cudaMemcpy(
            results_rg_array_.data(), device_rg_array_, array_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(
            results_az_array_.data(), device_az_array_, array_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(
            results_lat_array_.data(), device_lat_array_, array_size * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(
            results_lon_array_.data(), device_lon_array_, array_size * sizeof(double), cudaMemcpyDeviceToHost));
    }
    void DeviceFree() {
        if (device_rg_array_ != nullptr) {
            cudaFree(device_rg_array_);
            device_rg_array_ = nullptr;
        }
        if (device_az_array_ != nullptr) {
            cudaFree(device_az_array_);
            device_az_array_ = nullptr;
        }
        if (device_lat_array_ != nullptr) {
            cudaFree(device_lat_array_);
            device_lat_array_ = nullptr;
        }
        if (device_lon_array_ != nullptr) {
            cudaFree(device_lon_array_);
            device_lon_array_ = nullptr;
        }

        if (device_master_az_ != nullptr) {
            cudaFree(device_master_az_);
            device_master_az_ = nullptr;
        }
        if (device_master_rg_ != nullptr) {
            cudaFree(device_master_rg_);
            device_master_rg_ = nullptr;
        }
        if (device_slave_az_ != nullptr) {
            cudaFree(device_slave_az_);
            device_slave_az_ = nullptr;
        }
        if (device_slave_rg_ != nullptr) {
            cudaFree(device_slave_rg_);
            device_slave_rg_ = nullptr;
        }
        if (device_lat_ != nullptr) {
            cudaFree(device_lat_);
            device_lat_ = nullptr;
        }
        if (device_lon_ != nullptr) {
            cudaFree(device_lon_);
            device_lon_ = nullptr;
        }

        if (device_triangles_ != nullptr) {
            cudaFree(device_triangles_);
            device_triangles_ = nullptr;
        }
    }
};

const std::string TriangularInterpolationTester::az_rg_data_file_ = "./goods/backgeocoding/masterSlaveAzRgData.txt";
const std::string TriangularInterpolationTester::lats_lons_file_ = "./goods/backgeocoding/pixelposLatsLons.txt";
const std::string TriangularInterpolationTester::arrays_file_ = "./goods/backgeocoding/pixelposArrays.txt";
const std::string TriangularInterpolationTester::triangles_data_file_ =
    "./goods/backgeocoding/masterTrianglesTestData.txt";

void PrepareParams(TriangularInterpolationTester *tester,
                   alus::snapengine::triangularinterpolation::InterpolationParams *params,
                   alus::snapengine::triangularinterpolation::Window *window,
                   alus::snapengine::triangularinterpolation::Zdata *zdata) {
    window->linelo = 17000;
    window->linehi = 17099;
    window->pixlo = 4000;
    window->pixhi = 4099;
    window->lines = window->linehi - window->linelo + 1;
    window->pixels = window->pixhi - window->pixlo + 1;

    zdata[0].input_arr = tester->device_slave_az_;
    zdata[0].input_width = tester->az_rg_width_;
    zdata[0].input_height = tester->az_rg_height_;
    zdata[0].output_arr = tester->device_az_array_;
    zdata[0].output_width = window->lines;
    zdata[0].output_height = window->pixels;

    zdata[1].input_arr = tester->device_slave_rg_;
    zdata[1].input_width = tester->az_rg_width_;
    zdata[1].input_height = tester->az_rg_height_;
    zdata[1].output_arr = tester->device_rg_array_;
    zdata[1].output_width = window->lines;
    zdata[1].output_height = window->pixels;

    zdata[2].input_arr = tester->device_lat_;
    zdata[2].input_width = tester->az_rg_width_;
    zdata[2].input_height = tester->az_rg_height_;
    zdata[2].output_arr = tester->device_lat_array_;
    zdata[2].output_width = window->lines;
    zdata[2].output_height = window->pixels;

    zdata[3].input_arr = tester->device_lon_;
    zdata[3].input_width = tester->az_rg_width_;
    zdata[3].input_height = tester->az_rg_height_;
    zdata[3].output_arr = tester->device_lon_array_;
    zdata[3].output_width = window->lines;
    zdata[3].output_height = window->pixels;

    params->triangle_count = tester->triangle_size_;
    params->z_data_count = alus::backgeocoding::Z_DATA_SIZE;
    params->xy_ratio = TriangularInterpolationTester::RG_AZ_RATIO;
    params->invalid_index = TriangularInterpolationTester::INVALID_INDEX;
    params->x_scale = 1;
    params->y_scale = 1;
    params->offset = 0;
}

TEST(TriangularInterpolation, InterpolationTest) {
    TriangularInterpolationTester tester;
    tester.HostToDevice();

    alus::snapengine::triangularinterpolation::InterpolationParams params;
    alus::snapengine::triangularinterpolation::Window window;
    alus::snapengine::triangularinterpolation::Zdata zdata[alus::backgeocoding::Z_DATA_SIZE];
    alus::snapengine::triangularinterpolation::Zdata *device_zdata;

    PrepareParams(&tester, &params, &window, zdata);

    params.window = window;

    CHECK_CUDA_ERR(
        cudaMalloc((void **)&device_zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata)));
    CHECK_CUDA_ERR(
        cudaMemcpy(device_zdata,
                   zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                   cudaMemcpyHostToDevice));

    alus::snapengine::triangularinterpolation::LaunchInterpolation(tester.device_triangles_, device_zdata, params);

    tester.DeviceToHost();

    size_t size = window.lines * window.pixels;
    int slave_az_count = alus::EqualsArraysd(tester.results_az_array_.data(), tester.az_array_.data(), size, 0.00001);
    EXPECT_EQ(slave_az_count, 0) << "Slave azimuth results do not match. Mismatches: " << slave_az_count << '\n';

    int slave_rg_count = alus::EqualsArraysd(tester.results_rg_array_.data(), tester.rg_array_.data(), size, 0.00001);
    EXPECT_EQ(slave_rg_count, 0) << "Slave range results do not match. Mismatches: " << slave_rg_count << '\n';

    int lats_count = alus::EqualsArraysd(tester.results_lat_array_.data(), tester.lat_array_.data(), size, 0.00001);
    EXPECT_EQ(lats_count, 0) << "Latitude results do not match. Mismatches: " << lats_count << '\n';

    int lons_count = alus::EqualsArraysd(tester.results_lon_array_.data(), tester.lon_array_.data(), size, 0.00001);
    EXPECT_EQ(lons_count, 0) << "Longitude results do not match. Mismatches: " << lons_count << '\n';

    CHECK_CUDA_ERR(cudaFree(device_zdata));
}

TEST(DelaunayTest, BigCPUTriangulationTest) {
    TriangularInterpolationTester tester;

    alus::delaunay::DelaunayTriangulator triangulator;
    triangulator.TriangulateCPU2(tester.master_az_.data(),
                                 1.0,
                                 tester.master_rg_.data(),
                                 TriangularInterpolationTester::RG_AZ_RATIO,
                                 tester.az_rg_width_ * tester.az_rg_height_);
    std::cout << "nr of triangles: " << triangulator.triangle_count_ << std::endl;

    int count = alus::EqualsTriangles(
        triangulator.host_triangles_.data(), tester.triangles_.data(), triangulator.triangle_count_, 0.00001);
    EXPECT_EQ(count, 0) << "Triangle results do not match. Mismatches: " << count << '\n';
}

TEST(TriangularInterpolation, InterpolationAndTriangulation) {
    TriangularInterpolationTester tester;

    alus::delaunay::DelaunayTriangulator triangulator;

    alus::snapengine::triangularinterpolation::InterpolationParams params;
    alus::snapengine::triangularinterpolation::Window window;
    alus::snapengine::triangularinterpolation::Zdata zdata[alus::backgeocoding::Z_DATA_SIZE];
    alus::snapengine::triangularinterpolation::Zdata *device_zdata;

    triangulator.TriangulateCPU2(tester.master_az_.data(),
                                 1.0,
                                 tester.master_rg_.data(),
                                 TriangularInterpolationTester::RG_AZ_RATIO,
                                 tester.az_rg_width_ * tester.az_rg_height_);
    tester.triangle_size_ = triangulator.triangle_count_;
    tester.triangles_ = triangulator.host_triangles_;

    tester.HostToDevice();
    PrepareParams(&tester, &params, &window, zdata);
    params.window = window;

    CHECK_CUDA_ERR(
        cudaMalloc((void **)&device_zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata)));
    CHECK_CUDA_ERR(
        cudaMemcpy(device_zdata,
                   zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                   cudaMemcpyHostToDevice));

    alus::snapengine::triangularinterpolation::LaunchInterpolation(tester.device_triangles_, device_zdata, params);

    tester.DeviceToHost();

    int size = window.lines * window.pixels;
    int slave_az_count = alus::EqualsArraysd(tester.results_az_array_.data(), tester.az_array_.data(), size, 0.00001);
    EXPECT_EQ(slave_az_count, 0) << "Slave azimuth results do not match. Mismatches: " << slave_az_count << '\n';

    int slave_rg_count = alus::EqualsArraysd(tester.results_rg_array_.data(), tester.rg_array_.data(), size, 0.00001);
    EXPECT_EQ(slave_rg_count, 0) << "Slave range results do not match. Mismatches: " << slave_rg_count << '\n';

    int lats_count = alus::EqualsArraysd(tester.results_lat_array_.data(), tester.lat_array_.data(), size, 0.00001);
    EXPECT_EQ(lats_count, 0) << "Latitude results do not match. Mismatches: " << lats_count << '\n';

    int lons_count = alus::EqualsArraysd(tester.results_lon_array_.data(), tester.lon_array_.data(), size, 0.00001);
    EXPECT_EQ(lons_count, 0) << "Longitude results do not match. Mismatches: " << lons_count << '\n';

    CHECK_CUDA_ERR(cudaFree(device_zdata));
}

}  // namespace
