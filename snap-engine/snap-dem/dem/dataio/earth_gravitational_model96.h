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
#pragma once

#include <condition_variable>
#include <mutex>
#include <thread>

#include "cuda_friendly_object.h"

/** EXPLANATION FROM ESA SNAP
 * "WW15MGH.GRD"
 * <p>
 * This file contains 1038961 point values in grid form.  The first row of the file is the "header" of the file
 * and shows the south, north, west, and east limits of the file followed by the grid spacing in n-s and e-w.
 * All values in the "header" are in DECIMAL DEGREES.
 * <p>
 * The geoid undulation grid is computed at 15 arc minute spacings in north/south and east/west with the new
 * "EGM96" spherical harmonic potential coefficient set complete to degree and order 360 and a geoid height
 * correction value computed from a set of spherical harmonic coefficients ("CORRCOEF"), also to degree and
 * order 360.  The file is arranged from north to south, west to east (i.e., the data after the header is
 * the north most latitude band and is ordered from west to east).
 * <p>
 * The coverage of this file is:
 * <p>
 * 90.00 N  +------------------+
 * |                  |
 * | 15' spacing N/S  |
 * |                  |
 * |                  |
 * | 15' spacing E/W  |
 * |                  |
 * -90.00 N  +------------------+
 * 0.00 E           360.00 E
 */
namespace alus::snapengine {
/**
 * This class refers to EarthGravitationalModel96 class from snap-engine module.
 */
class EarthGravitationalModel96 : public cuda::CudaFriendlyObject {
private:
    void FetchGridValuesThread();
    void HostToDeviceThread();
    void FetchGridValues();

    float** egm_{nullptr};
    float* device_egm_{nullptr};
    bool is_inited_{false};
    bool is_on_device_{false};
    std::mutex init_mutex_;
    std::mutex host_mutex_;
    std::mutex device_mutex_;
    std::condition_variable init_var_;
    std::condition_variable copy_var_;

    std::exception_ptr egm_exception_{nullptr};
    std::thread init_thread_;
    std::thread copy_thread_;

public:
    EarthGravitationalModel96();

    [[nodiscard]] float** GetHostValues();
    [[nodiscard]] const float* GetDeviceValues();

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    virtual ~EarthGravitationalModel96();
    EarthGravitationalModel96(const EarthGravitationalModel96&) = delete;  // class does not support copying(and moving)
    EarthGravitationalModel96& operator=(const EarthGravitationalModel96&) = delete;
};

}  // namespace alus::snapengine
