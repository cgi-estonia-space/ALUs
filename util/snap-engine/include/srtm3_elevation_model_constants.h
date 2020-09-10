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
namespace snapengine{
namespace srtm3elevationmodel{


constexpr int NUM_X_TILES {72};
constexpr int NUM_Y_TILES {24};
constexpr int DEGREE_RES {5};
constexpr int NUM_PIXELS_PER_TILE {6000};
constexpr double NO_DATA_VALUE {-32768};
constexpr int RASTER_WIDTH {NUM_X_TILES * NUM_PIXELS_PER_TILE};
constexpr int RASTER_HEIGHT {NUM_Y_TILES * NUM_PIXELS_PER_TILE};

constexpr double NUM_PIXELS_PER_TILEinv {1.0 / (double) NUM_PIXELS_PER_TILE};
constexpr double DEGREE_RES_BY_NUM_PIXELS_PER_TILE {DEGREE_RES / (double) NUM_PIXELS_PER_TILE};
constexpr double DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv {1.0 / DEGREE_RES_BY_NUM_PIXELS_PER_TILE};

} //namespace
} //namespace
} //namespace