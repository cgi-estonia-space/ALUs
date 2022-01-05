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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace alus {  // NOLINT
namespace cuda {
/*
    A cuda friendly object sends some or all of it's internal pointers to the gpu and in some cases
    creates a special struct with its values and makes that available on the gpu too.
*/
class CudaFriendlyObject {
public:
    virtual void HostToDevice() = 0;
    virtual void DeviceToHost() = 0;
    virtual void DeviceFree() = 0;
};

}  // namespace cuda
}  // namespace alus