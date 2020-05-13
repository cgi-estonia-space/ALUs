#pragma once
#include <iostream>

#define ERROR_RANGE 0.0000001

namespace alus {

int equalsArrays(const float *a, const float *b, int elems);
int equalsArrays(const float *a, const float *b, int elems, float delta);
int equalsArraysd(const double *a, const double *b, int elems);
int equalsArrays2Dd(const double* const* a, const double* const* b, int x, int y);

}//namespace
