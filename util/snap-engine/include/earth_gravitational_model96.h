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

#include <string>
#include <fstream>

#include "CudaFriendlyObject.h"
#include "allocators.h"
#include "cuda_util.hpp"


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
namespace alus {
namespace snapengine {
namespace earthgravitationalmodel96{

constexpr int NUM_LATS = 721; // 180*4 + 1  (cover 90 degree to -90 degree)
constexpr int NUM_LONS = 1441; // 360*4 + 1 (cover 0 degree to 360 degree)
constexpr int NUM_CHAR_PER_NORMAL_LINE = 74;
constexpr int NUM_CHAR_PER_SHORT_LINE = 11;
constexpr int NUM_CHAR_PER_EMPTY_LINE = 1;
constexpr int BLOCK_HEIGHT = 20;
constexpr int NUM_OF_BLOCKS_PER_LAT = 9;

constexpr int MAX_LATS = NUM_LATS - 1;
constexpr int MAX_LONS = NUM_LONS - 1;

} //namespace earthgravitationalmodel96

/**
 * This class refers to EarthGravitationalModel96 class from snap-engine module.
 */
class EarthGravitationalModel96: public cuda::CudaFriendlyObject {
private:
    std::string grid_file_ = "../test/goods/ww15mgh_b.grd";

    void ReadGridFile();
public:
    float **egm_{nullptr};
    float *device_egm_{nullptr};

    EarthGravitationalModel96(std::string grid_file);
    EarthGravitationalModel96();
    ~EarthGravitationalModel96();

    void HostToDevice() override ;
    void DeviceToHost() override ;
    void DeviceFree() override ;

};

}//namespace
}//namespace
