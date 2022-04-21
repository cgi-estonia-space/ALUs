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

#include <future>
#include <thread>
#include <vector>

#include "cuda_device.h"

namespace alus::cuda {
class CudaInit final {
public:
    CudaInit();

    [[nodiscard]] bool IsFinished() const;
    void CheckErrors();

    [[nodiscard]] const std::vector<CudaDevice>& GetDevices() const { return devices_; }

    ~CudaInit();

private:
    void QueryDevices();

    std::vector<CudaDevice> devices_;
    std::future<void> init_future_;
    std::vector<std::thread> device_warmups_;
};
}  // namespace alus::cuda