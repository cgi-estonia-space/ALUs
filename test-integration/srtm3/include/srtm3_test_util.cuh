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

#include "pointer_holders.h"

namespace alus {
namespace tests {

struct SRTM3TestData {
    int size;
    PointerArray tiles;
};

cudaError_t LaunchSRTM3AltitudeTester(dim3 grid_size, dim3 block_size, double* lats, double* lons, double* results,
                                      SRTM3TestData data);

}  // namespace tests
}  // namespace alus
