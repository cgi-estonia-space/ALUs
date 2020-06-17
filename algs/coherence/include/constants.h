#pragma once

namespace alus::kcoh {
constexpr double C{299792458.0};
// private final static double ell_a = Constants.WGS84_A;
// wgs84
constexpr double ELL_A{6378137.000};
// private final static double ell_b = Constants.WGS84_B;
// wgs84
// also found from some other code: 6356752.314245179 (jlinda used shorter)
constexpr double ELL_B{6356752.314245};

constexpr double MASTER_CONTAINER_META_DATA_GET_RADAR_WAVELENGTH{0.05546576};
constexpr double SLAVE_CONTAINER_META_DATA_GET_RADAR_WAVELENGTH{0.05546576};

// this has also multiple values, need ot check which was used for what
constexpr double SNAP_PI{3.141592653589793};
}  // namespace alus::kcoh
