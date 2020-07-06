#include "terrain_correction_test.cuh"

#include <thrust/equal.h>
#include <thrust/device_vector.h>

bool alus::integrationtests::AreVectorsEqual(const std::vector<double>& control, const std::vector<double>& test) {
    thrust::device_vector<double> d_control(control);
    thrust::device_vector<double> d_test(test);
    return thrust::equal(thrust::device, d_control.begin(), d_control.end(), d_test.begin());
}