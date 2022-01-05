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

namespace alus {        // NOLINT
namespace snapengine {  // NOLINT
namespace earthgravitationalmodel96computation {

constexpr int NUM_LATS = 721;   // 180*4 + 1  (cover 90 degree to -90 degree)
constexpr int NUM_LONS = 1441;  // 360*4 + 1 (cover 0 degree to 360 degree)
constexpr int NUM_CHAR_PER_NORMAL_LINE = 74;
constexpr int NUM_CHAR_PER_SHORT_LINE = 11;
constexpr int NUM_CHAR_PER_EMPTY_LINE = 1;
constexpr int BLOCK_HEIGHT = 20;
constexpr int NUM_OF_BLOCKS_PER_LAT = 9;

constexpr int MAX_LATS = NUM_LATS - 1;
constexpr int MAX_LONS = NUM_LONS - 1;

}  // namespace earthgravitationalmodel96computation
}  // namespace snapengine
}  // namespace alus