#pragma once

#include "pointer_holders.h"

namespace alus {
namespace tests{

struct SRTM3TestData{
    int size;
    PointerArray tiles;
};

cudaError_t LaunchSRTM3AltitudeTester(dim3 grid_size, dim3 block_size, double *lats, double *lons, double *results, SRTM3TestData data);

}//namespace
}//namespace
