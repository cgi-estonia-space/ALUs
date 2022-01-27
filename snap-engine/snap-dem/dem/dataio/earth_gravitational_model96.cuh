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

#include <cuda_runtime.h>

namespace alus {
namespace snapengine {
namespace earthgravitationalmodel96computation {

/**
    What you see here are the extracts from EarthGravitationalModel96 class in snap
    and transform.AffineTransform2D from geotools. The getEGM96 has also suffered some unwrapping.
*/

inline __device__ double InterpolationCubicS(double y0, double y1, double y2, double y3, double mu) {
    return (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * mu * mu +
           (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * (mu * mu) + (-0.5 * y0 + 0.5 * y2) * mu + y1;
}

inline __device__ double InterpolationCubicB(double y0, double y1, double y2, double y3, double mu, double mu2,
                                             double mu3) {
    return (-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu3 + (y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3) * mu2 +
           (-0.5 * y0 + 0.5 * y2) * mu + y1;
}

inline __device__ float GetEGM96(double lat, double lon, int max_lats, int max_lons, float* egm) {
    const double r = (90 - lat) / 0.25;
    const double c = ((lon < 0) * (lon + 360) + (lon >= 0) * lon) / 0.25;
    int ri, i;
    double temp1, temp2, temp3, temp4;
    int ySize = max_lons + 1;

    int r0 = (int)r - 1;
    r0 = (r0 > 0) * r0;
    int c0 = (int)c - 1;
    c0 = (c0 > 0) * c0;

    int ci1 = c0 + 1;
    int ci2 = c0 + 2;
    int ci3 = c0 + 3;
    if (ci3 > max_lons) {
        c0 = min(c0, max_lons);
        ci1 = min(ci1, max_lons);
        ci2 = min(ci2, max_lons);
        ci3 = min(ci3, max_lons);
    }

    double mu_x = c - (double)(c0 + 1);
    double mu_y = r - (double)(r0 + 1);
    double mu_x2 = mu_x * mu_x;
    double mu_x3 = mu_x * mu_x2;

    i = 0;
    ri = (r0 + i > max_lats) * max_lats + (r0 + i <= max_lats) * (r0 + i);
    temp1 = InterpolationCubicB((double)egm[ri * ySize + c0], (double)egm[ri * ySize + ci1],
                                (double)egm[ri * ySize + ci2], (double)egm[ri * ySize + ci3], mu_x, mu_x2, mu_x3);

    i = 1;
    ri = (r0 + i > max_lats) * max_lats + (r0 + i <= max_lats) * (r0 + i);
    temp2 = InterpolationCubicB((double)egm[ri * ySize + c0], (double)egm[ri * ySize + ci1],
                                (double)egm[ri * ySize + ci2], (double)egm[ri * ySize + ci3], mu_x, mu_x2, mu_x3);

    i = 2;
    ri = (r0 + i > max_lats) * max_lats + (r0 + i <= max_lats) * (r0 + i);
    temp3 = InterpolationCubicB((double)egm[ri * ySize + c0], (double)egm[ri * ySize + ci1],
                                (double)egm[ri * ySize + ci2], (double)egm[ri * ySize + ci3], mu_x, mu_x2, mu_x3);

    i = 3;
    ri = (r0 + i > max_lats) * max_lats + (r0 + i <= max_lats) * (r0 + i);
    temp4 = InterpolationCubicB((double)egm[ri * ySize + c0], (double)egm[ri * ySize + ci1],
                                (double)egm[ri * ySize + ci2], (double)egm[ri * ySize + ci3], mu_x, mu_x2, mu_x3);

    return (float)InterpolationCubicS(temp1, temp2, temp3, temp4, mu_y);
}

}  // namespace earthgravitationalmodel96computation
}  // namespace snapengine
}  // namespace alus
