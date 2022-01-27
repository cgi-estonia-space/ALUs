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
#include "terrain_correction_test.cuh"

#include <thrust/device_vector.h>
#include <thrust/equal.h>

bool alus::integrationtests::AreVectorsEqual(const std::vector<double>& control, const std::vector<double>& test) {
    thrust::device_vector<double> d_control(control);
    thrust::device_vector<double> d_test(test);
    return thrust::equal(thrust::device, d_control.begin(), d_control.end(), d_test.begin());
}