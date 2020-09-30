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

namespace alus {
namespace terraincorrection {
constexpr double SEMI_MAJOR_AXIS{6378137.0};
constexpr double RTOD{57.29577951308232};
constexpr int INVALID_SUB_SWATH_INDEX{-1};
constexpr int BILINEAR_INTERPOLATION_MARGIN{1};
}  // namespace terraincorrection
}  // namespace alus