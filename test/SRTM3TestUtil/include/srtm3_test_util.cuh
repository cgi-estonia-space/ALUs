#pragma once

#include "pointer_holders.h"

namespace alus {
namespace tests{

struct SRTM3TestData{
    int size;
    PointerArray tiles;
};

cudaError_t launchSRTM3AltitudeTester(dim3 gridSize, dim3 blockSize, double *lats, double *lons, double *results, SRTM3TestData data);

}//namespace
}//namespace
