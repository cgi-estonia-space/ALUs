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

#include <cmath>
#include <string_view>

#include <gdal_priv.h>

namespace alus::resample {

void Reprojection(GDALDataset* from, GDALDataset* reprojected, std::string_view projection,
                  double longitude_factor = NAN, double latitude_factor = NAN);

}