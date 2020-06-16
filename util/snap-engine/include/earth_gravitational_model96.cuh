#pragma once

#include <cuda_runtime.h>

namespace alus {
namespace snapengine {
namespace earthgravitationalmodel96 {

/**
    What you see here are the extracts from EarthGravitationalModel96 class in snap
    and transform.AffineTransform2D from geotools. The getEGM96 has also suffered some unwrapping.
*/

inline __device__ double interpolationCubicS(double y0, double y1, double y2, double y3, double mu) {

    return ((-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu * (mu * mu) + (y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3) * (mu * mu) + (-0.5 * y0 + 0.5 * y2) * mu + y1);
}

inline __device__ double interpolationCubicB(double y0, double y1, double y2, double y3, double mu, double mu2, double mu3) {

    return ((-0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3) * mu3 + (y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3) * mu2 + (-0.5 * y0 + 0.5 * y2) * mu + y1);
}

inline __device__ float getEGM96(double lat, double lon, int maxLats, int maxLons, double* egm){
    const double r = (90 - lat) / 0.25;
    const double c = ((lon < 0)*(lon + 360) + (lon >= 0)*lon) / 0.25;
    int ri, i;
    double temp1, temp2, temp3, temp4;
    int ySize = maxLons + 1;

    int r0 = (int)r - 1;
    r0 = (r0 > 0) * r0;
    int c0 = (int) c - 1;
    c0 = (c0 > 0) * c0;

    int ci1 = c0 + 1;
    int ci2 = c0 + 2;
    int ci3 = c0 + 3;
    if (ci3 > maxLons) {
        c0 = min(c0, maxLons);
        ci1 = min(ci1, maxLons);
        ci2 = min(ci2, maxLons);
        ci3 = min(ci3, maxLons);
    }

    double muX = c - (c0 + 1);
    double muY = r - (r0 + 1);
    double muX2 = muX*muX;
    double muX3 = muX*muX2;

    i=0;
    ri = (r0 + i > maxLats)*maxLats + (r0 + i <= maxLats)*(r0 + i);
    temp1 = interpolationCubicB(egm[ri*ySize + c0], egm[ri*ySize + ci1], egm[ri*ySize + ci2], egm[ri*ySize + ci3], muX, muX2, muX3);
    i=1;
    ri = (r0 + i > maxLats)*maxLats + (r0 + i <= maxLats)*(r0 + i);
    temp2 = interpolationCubicB(egm[ri*ySize + c0], egm[ri*ySize + ci1], egm[ri*ySize + ci2], egm[ri*ySize + ci3], muX, muX2, muX3);
    i=2;
    ri = (r0 + i > maxLats)*maxLats + (r0 + i <= maxLats)*(r0 + i);
    temp3 = interpolationCubicB(egm[ri*ySize + c0], egm[ri*ySize + ci1], egm[ri*ySize + ci2], egm[ri*ySize + ci3], muX, muX2, muX3);
    i=3;
    ri = (r0 + i > maxLats)*maxLats + (r0 + i <= maxLats)*(r0 + i);
    temp4 = interpolationCubicB(egm[ri*ySize + c0], egm[ri*ySize + ci1], egm[ri*ySize + ci2], egm[ri*ySize + ci3], muX, muX2, muX3);

    return (float)interpolationCubicS(temp1, temp2, temp3, temp4, muY);
}

}//namespace
}//namespace
}//namespace
