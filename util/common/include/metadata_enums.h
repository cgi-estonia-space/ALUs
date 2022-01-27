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

namespace alus::metadata {
enum class ProductType { SLC };

enum class AcquisitionMode { IW };

enum class AntennaDirection { RIGHT, LEFT };

enum class Swath { IW1 };

enum class Pass { ASCENDING, DESCENDING };

enum class SampleType { COMPLEX };

enum class Polarisation { VH, VV };

enum class Algorithm { RANGE_DOPPLER };
}  // namespace alus::metadata