#include "cuda_util.hpp"

namespace alus {
namespace cuda {

//DO NOT USE math::ceil here. it was removed because of its inaccuracy.
int GetGridDim(int blockDim, int dataDim){
    double temp = dataDim /blockDim;
    int tempInt;
    if(temp < 1){
        return 1;
    }
    tempInt = (int)temp;
    if(tempInt*blockDim < dataDim){
        tempInt++;
    }
    return tempInt;
}

}//namespace
}//namespace
