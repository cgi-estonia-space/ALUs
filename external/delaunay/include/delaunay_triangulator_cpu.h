/**
 * Original implementation by Gilles Dumoulin. Original code was without a licence,
 * so no licence is enforced on this specific part.
 * The code is taken from http://paulbourke.net/papers/triangulate/
 */
#pragma once

#include <cstdlib> // for C qsort
#include <cmath>

#include "shapes.h"

namespace external {
namespace delaunay {

const double EPSILON = 0.000001;

struct ITRIANGLE {
    int p1, p2, p3;
};

struct IEDGE {
    int p1, p2;
};

int XYZCompare(const void *v1, const void *v2);
int Triangulate(int nv, alus::PointDouble pxyz[], ITRIANGLE v[], int &ntri);
int CircumCircle(double, double, double, double, double, double, double, double, double &, double &, double &);

}//namespace
}//namespace