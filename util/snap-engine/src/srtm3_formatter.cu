#include "srtm3_formatter.cuh"
#include "earth_gravitational_model96.cuh"

#include <cstdio>

namespace alus {
namespace snapengine{

__global__ void formatSRTM3dem(double *target, double *source, DemFormatterData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geoPosLon, geoPosLat, sourceValue;


    //possible bug:it is possible that this is a snap bug, as snap reads line 2501 when index is 2500.
    if(idx < data.xSize && idy < (data.ySize-1)){
        sourceValue = source[idx + data.xSize*(idy+1)];
        if(sourceValue != data.noDataValue){
            //everything that TileGeoReferencing.getGeoPos does.
            geoPosLon = data.m00*(idx + 0.5) + data.m01*(idy + 0.5) + data.m02;
            geoPosLat = data.m10*(idx + 0.5) + data.m11*(idy + 0.5) + data.m12;
            target[idx + data.xSize*idy] = sourceValue + snapengine::earthgravitationalmodel96::getEGM96(geoPosLat,geoPosLon, data.maxLats, data.maxLons, data.egm);
        }else{
            target[idx + data.xSize*idy] = sourceValue;
        }

    }
}


cudaError_t launchDemFormatter(dim3 gridSize, dim3 blockSize, double *target, double *source, DemFormatterData data){

    formatSRTM3dem<<<gridSize, blockSize>>>(
        target,
        source,
        data
    );
    return cudaGetLastError();
}

}//namespace
}//namespace
