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

namespace alus::jlinda {
// private final static double ell_a = Constants.WGS84_A;
// wgs84
constexpr double ELL_A{6378137.000};
// private final static double ell_b = Constants.WGS84_B;
// wgs84
// also found from some other code: 6356752.314245179 (jlinda used shorter)
constexpr double ELL_B{6356752.314245};

//constexpr double MASTER_CONTAINER_META_DATA_GET_RADAR_WAVELENGTH{0.05546576};
//constexpr double SLAVE_CONTAINER_META_DATA_GET_RADAR_WAVELENGTH{0.05546576};

// this has also multiple values, need ot check which was used for what
constexpr double SNAP_PI{3.141592653589793};

//JLINDA CONSTANTS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//    public static final double MeanEarthRadius = 6371008.7714;

constexpr double GRS80_A{6378137.000};    // GRS80 semi-major axis
constexpr double GRS80_B{6356752.314140}; // GRS80 semi-minor axis

constexpr double SOL{2.99792458E8}; // speed-of-light
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!package org.esa.snap.engine_utilities.eo;

}  // namespace alus::kcoh
