#pragma once
#include <iostream>

#define ERROR_RANGE 0.0000001

namespace slap {

int equalsArrays(const float *a, const float *b, int elems);
int equalsArrays(const float *a, const float *b, int elems, float delta);
int equalsArraysd(double *a, double *b, int elems);
int equalsArrays2Dd(double **a, double **b, int x, int y);

}//namespace
