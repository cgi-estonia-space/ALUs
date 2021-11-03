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
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"

#include <mutex>
#include <sstream>
#include <thread>

#include "allocators.h"
#include "cuda_util.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96_computation.h"
#include "ww15mgh_b_grd.h"

namespace alus::snapengine {

EarthGravitationalModel96::EarthGravitationalModel96() { this->FetchGridValues(); }

EarthGravitationalModel96::~EarthGravitationalModel96() {
    if (init_thread_.joinable()) {
        init_thread_.join();
    }
    if (copy_thread_.joinable()) {
        copy_thread_.join();
    }
    Deallocate2DArray(egm_);
    DeviceFree();
}

void EarthGravitationalModel96::FetchGridValuesThread() {
    try {
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
    } catch (const std::exception&) {
        egm_exception_ = std::current_exception();
    }
    std::unique_lock init_lock(init_mutex_);
    std::unique_lock host_lock(host_mutex_);
    is_inited_ = true;
    init_var_.notify_all();
}

void EarthGravitationalModel96::FetchGridValues() {
    if (!init_thread_.joinable()) {
        init_thread_ = std::thread(&EarthGravitationalModel96::FetchGridValuesThread, this);
    }
}

void EarthGravitationalModel96::HostToDeviceThread() {
    std::unique_lock copy_lock(init_mutex_);
    init_var_.wait(copy_lock, [this]() { return this->is_inited_; });

    if (egm_exception_ == nullptr) {
        try {
            CHECK_CUDA_ERR(cudaMalloc((void**)&device_egm_, earthgravitationalmodel96computation::NUM_LATS *
                                                                earthgravitationalmodel96computation::NUM_LONS *
                                                                sizeof(float)));

            CHECK_CUDA_ERR(cudaMemcpy(this->device_egm_, this->egm_[0],
                                      earthgravitationalmodel96computation::NUM_LATS *
                                          earthgravitationalmodel96computation::NUM_LONS * sizeof(float),
                                      cudaMemcpyHostToDevice));
        } catch (const std::exception&) {
            egm_exception_ = std::current_exception();
        }
    }
    std::unique_lock device_lock(device_mutex_);
    is_on_device_ = true;
    copy_var_.notify_all();
}

void EarthGravitationalModel96::HostToDevice() {
    if (!copy_thread_.joinable()) {
        copy_thread_ = std::thread(&EarthGravitationalModel96::HostToDeviceThread, this);
    }
}

float** EarthGravitationalModel96::GetHostValues() {
    std::unique_lock host_lock(host_mutex_);
    init_var_.wait(host_lock, [this]() { return this->is_inited_; });

    if (egm_exception_ != nullptr) {
        std::rethrow_exception(egm_exception_);
    }
    return egm_;
}

const float* EarthGravitationalModel96::GetDeviceValues() {
    std::unique_lock device_lock(device_mutex_);
    copy_var_.wait(device_lock, [this]() { return this->is_on_device_; });

    if (egm_exception_ != nullptr) {
        std::rethrow_exception(egm_exception_);
    }
    return device_egm_;
}

void EarthGravitationalModel96::DeviceToHost() { CHECK_CUDA_ERR(cudaErrorNotYetImplemented); }

void EarthGravitationalModel96::DeviceFree() {
    if (this->device_egm_ != nullptr) {
        cudaFree(this->device_egm_);
        this->device_egm_ = nullptr;
    }
}

}  // namespace alus::snapengine
