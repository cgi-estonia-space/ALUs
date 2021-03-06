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

#include "comparators.h"
#include "cuda_friendly_object.h"
#include "cuda_util.h"

#include "backgeocoding_constants.h"
#include "delaunay_triangulator.h"
#include "triangular_interpolation_computation.h"

namespace {

class TriangularInterpolationTester : public alus::cuda::CudaFriendlyObject {
private:
    static const std::string AZ_RG_DATA_FILE;
    static const std::string LATS_LONS_FILE;
    static const std::string ARRAYS_FILE;
    static const std::string TRIANGLES_DATA_FILE;

public:
    static constexpr double RG_AZ_RATIO{0.16742323135844578};
    static constexpr double INVALID_INDEX{-9999.0};

    double *device_master_az_ = nullptr, *device_master_rg_ = nullptr;
    double *device_slave_az_ = nullptr, *device_slave_rg_ = nullptr;
    double *device_lat_ = nullptr, *device_lon_ = nullptr;
    double *device_rg_array_ = nullptr, *device_az_array_ = nullptr;
    double *device_lat_array_ = nullptr, *device_lon_array_ = nullptr;
    alus::delaunay::DelaunayTriangle2D* device_triangles_ = nullptr;

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
        std::ifstream rg_az_stream(AZ_RG_DATA_FILE);
        std::ifstream lats_lons_stream(LATS_LONS_FILE);

        if (!rg_az_stream.is_open()) {
            throw std::ios::failure("masterSlaveAzRgData.txt is not open.");
        }
        if (!lats_lons_stream.is_open()) {
            throw std::ios::failure("pixelposLatsLons.txt is not open.");
        }

        rg_az_stream >> az_rg_width_ >> az_rg_height_;
        int coord_size = static_cast<int>(az_rg_width_) * static_cast<int>(az_rg_height_);

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

        std::ifstream arrays_stream(ARRAYS_FILE);

        if (!arrays_stream.is_open()) {
            throw std::ios::failure("pixelposArrays.txt is not open.");
        }

        arrays_stream >> arr_width_ >> arr_height_;
        const size_t arr_size = arr_width_ * arr_height_;

        rg_array_.resize(arr_size);
        az_array_.resize(arr_size);
        lat_array_.resize(arr_size);
        lon_array_.resize(arr_size);

        results_rg_array_.resize(arr_size);
        results_az_array_.resize(arr_size);
        results_lat_array_.resize(arr_size);
        results_lon_array_.resize(arr_size);

        for (size_t i = 0; i < arr_size; i++) {
            arrays_stream >> az_array_.at(i) >> rg_array_.at(i) >> lat_array_.at(i) >> lon_array_.at(i);
        }

        arrays_stream.close();

        std::ifstream triangles_stream(TriangularInterpolationTester::TRIANGLES_DATA_FILE);
        if (!triangles_stream.is_open()) {
            throw std::ios::failure("masterTrianglesTestData.txt is not open.");
        }
        triangles_stream >> triangle_size_;
        this->triangles_.resize(triangle_size_);

        // NOLINTNEXTLINE
        // TODO: Seeing anything familiar? Figure out how to put this together with delaunay_test.cc once there is time.
        for (size_t i = 0; i < triangle_size_; i++) {
            const double index_step{0.001};
            triangles_stream >> temp_triangle.ax >> temp_triangle.ay >> temp_index;
            temp_index += index_step;  // fixing a possible float inaccuracy
            temp_triangle.a_index = static_cast<int>(temp_index);

            triangles_stream >> temp_triangle.bx >> temp_triangle.by >> temp_index;
            temp_index += index_step;  // fixing a possible float inaccuracy
            temp_triangle.b_index = static_cast<int>(temp_index);

            triangles_stream >> temp_triangle.cx >> temp_triangle.cy >> temp_index;
            temp_index += index_step;  // fixing a possible float inaccuracy
            temp_triangle.c_index = static_cast<int>(temp_index);

            triangles_.at(i) = temp_triangle;
        }

        triangles_stream.close();
    }

    void HostToDevice() override {
        size_t array_size = arr_width_ * arr_height_;
        size_t az_rg_size = az_rg_width_ * az_rg_height_;

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_rg_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_az_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lat_array_, array_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lon_array_, array_size * sizeof(double)));

        CHECK_CUDA_ERR(cudaMalloc((void**)&device_master_az_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_master_rg_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_slave_az_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_slave_rg_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lat_, az_rg_size * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&device_lon_, az_rg_size * sizeof(double)));

        CHECK_CUDA_ERR(
            cudaMalloc((void**)&device_triangles_, triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2D)));

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

        CHECK_CUDA_ERR(cudaMemcpy(device_triangles_, triangles_.data(),
                                  triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2D), cudaMemcpyHostToDevice));
    }
    void DeviceToHost() override {
        size_t array_size = arr_width_ * arr_height_;

        CHECK_CUDA_ERR(cudaMemcpy(results_rg_array_.data(), device_rg_array_, array_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(results_az_array_.data(), device_az_array_, array_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(results_lat_array_.data(), device_lat_array_, array_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(results_lon_array_.data(), device_lon_array_, array_size * sizeof(double),
                                  cudaMemcpyDeviceToHost));
    }
    void DeviceFree() override {
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

const std::string TriangularInterpolationTester::AZ_RG_DATA_FILE = "./goods/backgeocoding/masterSlaveAzRgData.txt";
const std::string TriangularInterpolationTester::LATS_LONS_FILE = "./goods/backgeocoding/pixelposLatsLons.txt";
const std::string TriangularInterpolationTester::ARRAYS_FILE = "./goods/backgeocoding/pixelposArrays.txt";
const std::string TriangularInterpolationTester::TRIANGLES_DATA_FILE =
    "./goods/backgeocoding/masterTrianglesTestData.txt";

void PrepareParams(TriangularInterpolationTester* tester,
                   alus::snapengine::triangularinterpolation::TriangleInterpolationParams* params,
                   alus::snapengine::triangularinterpolation::Window* window,
                   alus::snapengine::triangularinterpolation::Zdata* zdata) {
    window->linelo = 17000;  // NOLINT
    window->linehi = 17099;  // NOLINT
    window->pixlo = 4000;    // NOLINT
    window->pixhi = 4099;    // NOLINT
    window->lines = static_cast<int>(window->linehi) - static_cast<int>(window->linelo) + 1;
    window->pixels = static_cast<int>(window->pixhi) - static_cast<int>(window->pixlo) + 1;

    zdata[0].input_arr = tester->device_slave_az_;
    zdata[0].input_width = tester->az_rg_width_;
    zdata[0].input_height = tester->az_rg_height_;
    zdata[0].output_arr = tester->device_az_array_;
    zdata[0].output_width = window->lines;
    zdata[0].output_height = window->pixels;
    zdata[0].min_int = std::numeric_limits<int>::max();
    zdata[0].max_int = std::numeric_limits<int>::lowest();

    zdata[1].input_arr = tester->device_slave_rg_;
    zdata[1].input_width = tester->az_rg_width_;
    zdata[1].input_height = tester->az_rg_height_;
    zdata[1].output_arr = tester->device_rg_array_;
    zdata[1].output_width = window->lines;
    zdata[1].output_height = window->pixels;
    zdata[1].min_int = std::numeric_limits<int>::max();
    zdata[1].max_int = std::numeric_limits<int>::lowest();

    zdata[2].input_arr = tester->device_lat_;
    zdata[2].input_width = tester->az_rg_width_;
    zdata[2].input_height = tester->az_rg_height_;
    zdata[2].output_arr = tester->device_lat_array_;
    zdata[2].output_width = window->lines;
    zdata[2].output_height = window->pixels;
    zdata[2].min_int = std::numeric_limits<int>::max();
    zdata[2].max_int = std::numeric_limits<int>::lowest();

    zdata[3].input_arr = tester->device_lon_;               // NOLINT
    zdata[3].input_width = tester->az_rg_width_;            // NOLINT
    zdata[3].input_height = tester->az_rg_height_;          // NOLINT
    zdata[3].output_arr = tester->device_lon_array_;        // NOLINT
    zdata[3].output_width = window->lines;                  // NOLINT
    zdata[3].output_height = window->pixels;                // NOLINT
    zdata[3].min_int = std::numeric_limits<int>::max();     // NOLINT
    zdata[3].max_int = std::numeric_limits<int>::lowest();  // NOLINT

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

    alus::snapengine::triangularinterpolation::TriangleInterpolationParams params;
    alus::snapengine::triangularinterpolation::Window window;
    alus::snapengine::triangularinterpolation::Zdata zdata[alus::backgeocoding::Z_DATA_SIZE];
    alus::snapengine::triangularinterpolation::Zdata* device_zdata;

    PrepareParams(&tester, &params, &window, zdata);

    params.window = window;

    CHECK_CUDA_ERR(cudaMalloc((void**)&device_zdata, alus::backgeocoding::Z_DATA_SIZE *
                                                         sizeof(alus::snapengine::triangularinterpolation::Zdata)));
    CHECK_CUDA_ERR(
        cudaMemcpy(device_zdata, zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                   cudaMemcpyHostToDevice));

    alus::snapengine::triangularinterpolation::LaunchInterpolation(tester.device_triangles_, device_zdata, params);

    tester.DeviceToHost();

    const size_t size = static_cast<size_t>(window.lines) * window.pixels;
    size_t slave_az_count = alus::EqualsArraysd(tester.results_az_array_.data(), tester.az_array_.data(),
                                                static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(slave_az_count, 0) << "Slave azimuth results do not match. Mismatches: " << slave_az_count << std::endl;

    size_t slave_rg_count = alus::EqualsArraysd(tester.results_rg_array_.data(), tester.rg_array_.data(),
                                                static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(slave_rg_count, 0) << "Slave range results do not match. Mismatches: " << slave_rg_count << std::endl;

    size_t lats_count = alus::EqualsArraysd(tester.results_lat_array_.data(), tester.lat_array_.data(),
                                            static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(lats_count, 0) << "Latitude results do not match. Mismatches: " << lats_count << std::endl;

    size_t lons_count = alus::EqualsArraysd(tester.results_lon_array_.data(), tester.lon_array_.data(),
                                            static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(lons_count, 0) << "Longitude results do not match. Mismatches: " << lons_count << std::endl;

    CHECK_CUDA_ERR(cudaFree(device_zdata));
}

TEST(DelaunayTest, BigCPUTriangulationTest) {
    TriangularInterpolationTester tester;

    alus::delaunay::DelaunayTriangulator triangulator;
    triangulator.TriangulateCPU2(tester.master_az_.data(), 1.0, tester.master_rg_.data(),
                                 TriangularInterpolationTester::RG_AZ_RATIO,
                                 static_cast<int>(tester.az_rg_width_) * static_cast<int>(tester.az_rg_height_),
                                 alus::backgeocoding::INVALID_INDEX);

    size_t count = alus::EqualsTriangles(triangulator.host_triangles_.data(), tester.triangles_.data(),
                                         triangulator.triangle_count_, 0.00001);  // NOLINT
    EXPECT_EQ(count, 0) << "Triangle results do not match. Mismatches: " << count << std::endl;
}

TEST(TriangularInterpolation, InterpolationAndTriangulation) {
    TriangularInterpolationTester tester;

    alus::delaunay::DelaunayTriangulator triangulator;

    alus::snapengine::triangularinterpolation::TriangleInterpolationParams params;
    alus::snapengine::triangularinterpolation::Window window;
    alus::snapengine::triangularinterpolation::Zdata zdata[alus::backgeocoding::Z_DATA_SIZE];
    alus::snapengine::triangularinterpolation::Zdata* device_zdata;

    triangulator.TriangulateCPU2(tester.master_az_.data(), 1.0, tester.master_rg_.data(),
                                 TriangularInterpolationTester::RG_AZ_RATIO,
                                 static_cast<int>(tester.az_rg_width_) * static_cast<int>(tester.az_rg_height_),
                                 alus::backgeocoding::INVALID_INDEX);
    tester.triangle_size_ = triangulator.triangle_count_;
    tester.triangles_ = triangulator.host_triangles_;

    tester.HostToDevice();
    PrepareParams(&tester, &params, &window, zdata);
    params.window = window;

    CHECK_CUDA_ERR(cudaMalloc((void**)&device_zdata, alus::backgeocoding::Z_DATA_SIZE *
                                                         sizeof(alus::snapengine::triangularinterpolation::Zdata)));
    CHECK_CUDA_ERR(
        cudaMemcpy(device_zdata, zdata,
                   alus::backgeocoding::Z_DATA_SIZE * sizeof(alus::snapengine::triangularinterpolation::Zdata),
                   cudaMemcpyHostToDevice));

    alus::snapengine::triangularinterpolation::LaunchInterpolation(tester.device_triangles_, device_zdata, params);

    tester.DeviceToHost();

    const size_t size = static_cast<size_t>(window.lines) * window.pixels;
    size_t slave_az_count = alus::EqualsArraysd(tester.results_az_array_.data(), tester.az_array_.data(),
                                                static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(slave_az_count, 0) << "Slave azimuth results do not match. Mismatches: " << slave_az_count << '\n';

    size_t slave_rg_count = alus::EqualsArraysd(tester.results_rg_array_.data(), tester.rg_array_.data(),
                                                static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(slave_rg_count, 0) << "Slave range results do not match. Mismatches: " << slave_rg_count << '\n';

    size_t lats_count = alus::EqualsArraysd(tester.results_lat_array_.data(), tester.lat_array_.data(),
                                            static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(lats_count, 0) << "Latitude results do not match. Mismatches: " << lats_count << '\n';

    size_t lons_count = alus::EqualsArraysd(tester.results_lon_array_.data(), tester.lon_array_.data(),
                                            static_cast<int>(size), 0.00001);  // NOLINT
    EXPECT_EQ(lons_count, 0) << "Longitude results do not match. Mismatches: " << lons_count << '\n';

    CHECK_CUDA_ERR(cudaFree(device_zdata));
}

}  // namespace
