#pragma once

namespace alus::jlinda {
constexpr double C{299792458.0};
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
static constexpr double EPS{1e-13};

static constexpr double WGS84_A{6378137.000};    // WGS84 semi-major axis
static constexpr double WGS84_B{6356752.314245}; // WGS84 semi-minor axis
//    public static final double MeanEarthRadius = 6371008.7714;

static constexpr double GRS80_A{6378137.000};    // GRS80 semi-major axis
static constexpr double GRS80_B{6356752.314140}; // GRS80 semi-minor axis

static  constexpr double SOL{2.99792458E8}; // speed-of-light
static  constexpr double LIGHT_SPEED{299792458.0}; //  m / s

static constexpr long MEGA{1000000}; // Math.pow(10,6)
static constexpr long GIGA{1000000000}; // Math.pow(10,9)

static constexpr double PI{3.14159265358979323846264338327950288};
static constexpr double _PI{3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348};
static constexpr double _TWO_PI{6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696};

static constexpr double RTOD = 180.0 / _PI;
static constexpr double DTOR = _PI / 180.0;

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//!!!package org.esa.snap.engine_utilities.eo;
static constexpr double SECONDS_IN_DAY{86400.0};

}  // namespace alus::kcoh
