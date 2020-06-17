#pragma once
#include <iostream>

#define ERROR_RANGE 0.0000001

namespace alus {

int EqualsArrays(const float *a, const float *b, int elems);
int EqualsArrays(const float *a, const float *b, int elems, float delta);
int EqualsArraysd(const double *a, const double *b, int elems);
int EqualsArraysd(const double *a, const double *b, int elems, double delta);
int EqualsArrays2Dd(const double *const *a, const double *const *b, int x, int y);

}//namespace
