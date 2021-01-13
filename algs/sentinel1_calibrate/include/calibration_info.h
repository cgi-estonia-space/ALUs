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

#include <vector>
#include <string>

#include "calibration_vector.h"

namespace alus::sentinel1calibrate{

/**
 * Port of SNAP's CalibrationInfo class.
 *
 * Original implementation found in Sentinel1Calibrator.java
 */
struct CalibrationInfo {
    std::string sub_swath;
    std::string polarisation;
    double first_line_time;
    double last_line_time;
    double line_time_interval;
    int num_of_lines;
    int count;
    std::vector<s1tbx::CalibrationVector> calibration_vectors;
};

}