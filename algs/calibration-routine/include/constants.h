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

#include <array>
#include <string_view>

namespace alus::calibrationroutine {

constexpr std::string_view ALG_NAME{"Calibration"};
constexpr std::array<std::string_view, 3> SUBSWATHS{"IW1", "IW2", "IW3"};
constexpr std::array<std::string_view, 2> POLARISATIONS{"VV", "VH"};
constexpr size_t INVALID_BURST_INDEX{0};
constexpr std::string_view CALIBRATION_TYPE_BETA{"beta"};
constexpr std::string_view CALIBRATION_TYPE_SIGMA{"sigma"};
constexpr std::string_view CALIBRATION_TYPE_GAMMA{"gamma"};
constexpr std::string_view CALIBRATION_TYPE_DN{"dn"};
}  // namespace alus::calibrationroutine