#pragma once
#include <iostream>

#define ERROR_RANGE 0.0000001

namespace slap {

int equalsArrays(float *a, float *b, int elems);
int equalsArraysd(double *a, double *b, int elems);
int equalsArrays2Dd(double **a, double **b, int x, int y);

}//namespace
