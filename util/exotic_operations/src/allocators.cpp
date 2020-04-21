#include "allocators.hpp"

namespace slap {

double ** allocate2DDoubleArray(int x, int y){
    int i=0, size=x*y, countX=0;
    double ** result = new double*[x];
    double * inside = new double[size];

    for(i=0; i<size; i+=y){
        result[countX] = &inside[i];
        countX++;
    }
    return result;
}

}//namespace
