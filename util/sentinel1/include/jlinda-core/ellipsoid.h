/**
 * This file is a filtered duplicate of a SNAP's
 * org.jlinda.core.Orbit.java
 * ported for native code
 *
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
    static double ComputeEllipsoidNormal(double phi);
    double ComputeCurvatureRadiusInMeridianPlane(double phi);
    // first ecc.
    static void SetEcc1stSqr();
    // second ecc.
    static void SetEcc2ndSqr();

   public:
    static double a_;  // semi major
    static double b_;  // semi minor
    static std::string name_;

    Ellipsoid();

    Ellipsoid(double semi_major, double semi_minor);

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
    static s1tbx::Point Ell2Xyz(double phi, double lambda, double height);

    static s1tbx::Point Ell2Xyz(const std::vector<double>& phi_lambda_height);

    static s1tbx::Point Ell2Xyz(const GeoPoint& geo_point, double height);

    static s1tbx::Point Ell2Xyz(const GeoPoint& geo_point);

    static void Ell2Xyz(const GeoPoint& geo_point, std::vector<double>& xyz);

    static void Ell2Xyz(const GeoPoint& geo_point, double height, std::vector<double>& xyz);
};

}  // namespace jlinda
}  // namespace alus
