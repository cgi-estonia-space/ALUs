/**
This file is a filtered duplicate of a SNAP's  org.jlinda.core.Orbit.java ported for native code
*/
#pragma once

#include <string>
#include <vector>

#include "geopoint.h"
#include "point.h"

namespace alus {
namespace jlinda {

class Ellipsoid {
   private:
    static double e2_;   // squared first  eccentricity (derived)
    static double e2b_;  // squared second eccentricity (derived)
    static double ComputeEllipsoidNormal(const double phi);
    double ComputeCurvatureRadiusInMeridianPlane(const double phi);
    // first ecc.
    static void SetEcc1stSqr();
    // second ecc.
    static void SetEcc2ndSqr();

   public:
    static double a_;  // semi major
    static double b_;  // semi minor
    static std::string name_;

    Ellipsoid();

    Ellipsoid(const double semi_major, const double semi_minor);

    Ellipsoid(const Ellipsoid& ell);

    void ShowData();

    /**
     *  Convert xyz cartesian coordinates to
     *  Geodetic ellipsoid coordinates latlonh
     *    xyz2ell
     *
     * Converts geocentric cartesian coordinates in the XXXX
     *  reference frame to geodetic coordinates.
     *  method of bowring see globale en locale geodetische systemen
     * input:
     *  - ellipsinfo, xyz, (phi,lam,hei)
     * output:
     *  - void (returned double[] lam<-pi,pi>, phi<-pi,pi>, hei)
     *
     */
    static std::vector<double> Xyz2Ell(const s1tbx::Point& xyz);

    /**
     * ell2xyz
     * Converts wgs84 ellipsoid cn to geocentric cartesian coord.
     * input:
     * - phi,lam,hei (geodetic co-latitude, longitude, [rad] h [m]
     * output:
     * - cn XYZ
     */
    static s1tbx::Point Ell2Xyz(const double phi, const double lambda, const double height);

    static s1tbx::Point Ell2Xyz(const std::vector<double> phi_lambda_height);

    static s1tbx::Point Ell2Xyz(const GeoPoint& geo_point, const double height);

    static s1tbx::Point Ell2Xyz(const GeoPoint& geo_point);

    static void Ell2Xyz(const GeoPoint& geo_point, std::vector<double>& xyz);

    static void Ell2Xyz(const GeoPoint& geo_point, const double height, std::vector<double>& xyz);
};

}  // namespace jlinda
}  // namespace alus
