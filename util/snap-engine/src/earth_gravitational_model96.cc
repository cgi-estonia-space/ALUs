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
#include "earth_gravitational_model96.h"

#include <sstream>

#include "allocators.h"
#include "cuda_util.hpp"
#include "earth_gravitational_model96_computation.h"
#include "ww15mgh_b_grd.h"

namespace alus::snapengine {

EarthGravitationalModel96::EarthGravitationalModel96() { this->FetchGridValues(); }

EarthGravitationalModel96::~EarthGravitationalModel96() {
    if (this->egm_ != nullptr) {
        delete[] this->egm_;
    }
    this->DeviceFree();
}

void EarthGravitationalModel96::FetchGridValues() {
    std::stringstream grid_reader(WW15MGH_B_GRID);
    this->egm_ = Allocate2DArray<float>(earthgravitationalmodel96computation::NUM_LATS,
                                        earthgravitationalmodel96computation::NUM_LONS);

    int num_char_in_header = earthgravitationalmodel96computation::NUM_CHAR_PER_NORMAL_LINE +
                             earthgravitationalmodel96computation::NUM_CHAR_PER_EMPTY_LINE;

    grid_reader.seekg(num_char_in_header, grid_reader.beg);

    for (int row_idx = 0; row_idx < earthgravitationalmodel96computation::NUM_LATS; row_idx++) {
        for (int col_idx = 0; col_idx < earthgravitationalmodel96computation::NUM_LONS; col_idx++) {
            grid_reader >> this->egm_[row_idx][col_idx];
        }
    }
}

void EarthGravitationalModel96::HostToDevice() {
    CHECK_CUDA_ERR(cudaMalloc((void**)&device_egm_, earthgravitationalmodel96computation::NUM_LATS *
                                                        earthgravitationalmodel96computation::NUM_LONS *
                                                        sizeof(float)));

    CHECK_CUDA_ERR(cudaMemcpy(
        this->device_egm_, this->egm_[0],
        earthgravitationalmodel96computation::NUM_LATS * earthgravitationalmodel96computation::NUM_LONS * sizeof(float),
        cudaMemcpyHostToDevice));
}

void EarthGravitationalModel96::DeviceToHost() { CHECK_CUDA_ERR(cudaErrorNotYetImplemented); }

void EarthGravitationalModel96::DeviceFree() {
    if (this->device_egm_ != nullptr) {
        cudaFree(this->device_egm_);
        this->device_egm_ = nullptr;
    }
}

}  // namespace alus::snapengine
