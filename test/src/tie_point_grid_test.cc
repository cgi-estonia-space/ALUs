#include "gmock/gmock.h"
#include "tests_common.hpp"

#include "CudaFriendlyObject.hpp"
#include "cuda_util.hpp"
#include "tie_point_grid.h"
#include "tie_point_grid_test.cuh"

using namespace alus;
using namespace alus::snapengine;
using namespace alus::tests;

namespace {

class TiePointGridTester : public cuda::CudaFriendlyObject {
   public:
    // Array values are received by running terrain correction on
    // S1A_IW_SLC__1SDV_20190715T160437_20190715T160504_028130_032D5B_58D6_Orb_Stack_coh_deb.dim data file.
    float tie_points_array_[126]{
        58.213177, 58.22355,  58.23374,  58.243763, 58.25362,  58.263313, 58.272854, 58.28224,  58.29149,  58.3006,
        58.30957,  58.318413, 58.32713,  58.33572,  58.344196, 58.352554, 58.3608,   58.36894,  58.37697,  58.384895,
        58.39272,  58.2498,   58.26018,  58.270382, 58.280407, 58.290268, 58.29997,  58.309513, 58.31891,  58.32816,
        58.337273, 58.34625,  58.355095, 58.363815, 58.372414, 58.38089,  58.389256, 58.397503, 58.405643, 58.413677,
        58.421608, 58.42944,  58.28643,  58.296814, 58.307022, 58.31705,  58.32692,  58.336624, 58.346172, 58.35557,
        58.36483,  58.373947, 58.382927, 58.39178,  58.400505, 58.409103, 58.417587, 58.425953, 58.434208, 58.442352,
        58.45039,  58.458324, 58.466156, 58.323055, 58.33345,  58.34366,  58.3537,   58.363567, 58.37328,  58.38283,
        58.39224,  58.4015,   58.41062,  58.419605, 58.428463, 58.43719,  58.445797, 58.454285, 58.462654, 58.470913,
        58.47906,  58.487103, 58.49504,  58.502876, 58.359684, 58.370083, 58.3803,   58.390343, 58.40022,  58.40993,
        58.419495, 58.4289,   58.438168, 58.447296, 58.456287, 58.465145, 58.47388,  58.48249,  58.49098,  58.499355,
        58.507614, 58.515766, 58.52381,  58.531754, 58.539593, 58.39631,  58.40671,  58.416935, 58.426983, 58.436863,
        58.446587, 58.45615,  58.465565, 58.474834, 58.483967, 58.49296,  58.501827, 58.510563, 58.51918,  58.52767,
        58.536053, 58.544315, 58.55247,  58.56052,  58.568466, 58.57631};
    cudautil::KernelArray<float> tie_points_ = {tie_points_array_, 126};

    const double EXPECTED_RESULT_ = 58.21324222804141;
    double end_result_;
    double *device_result_ {};
    float *device_tie_points_ {};

    ~TiePointGridTester() { this->deviceFree(); }

    void hostToDevice() {
        CHECK_CUDA_ERR(cudaMalloc(&device_tie_points_, sizeof(float) * tie_points_.size));
        CHECK_CUDA_ERR(cudaMalloc(&device_result_, sizeof(double)));
        CHECK_CUDA_ERR(
            cudaMemcpy(device_tie_points_, tie_points_.array, sizeof(float) * tie_points_.size, cudaMemcpyHostToDevice));
    }

    void deviceToHost() {
        CHECK_CUDA_ERR(cudaMemcpy(&end_result_, device_result_, sizeof(double), cudaMemcpyDeviceToHost));
    }

    void deviceFree() {
        cudaFree(device_tie_points_);
        cudaFree(device_result_);
    }
};

TEST(getPixelDouble, tie_point_grid) {
    TiePointGridTester tester;

    tester.hostToDevice();
    CHECK_CUDA_ERR(LaunchGetPixelDouble(
        1,
        1,
        0.5,
        0.5,
        tester.device_result_,
        tiepointgrid::TiePointGrid{0, 0, 1163, 300, 21, 6, {tester.device_tie_points_, tester.tie_points_.size}}));

    tester.deviceToHost();
    EXPECT_DOUBLE_EQ(tester.end_result_, tester.EXPECTED_RESULT_);
}

}  // namespace