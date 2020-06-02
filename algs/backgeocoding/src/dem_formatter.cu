#include "dem_formatter.cuh"


namespace alus {

__global__ void formatSRTM3dem(double *target, double *source, DemFormatterData data){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geoPosLon, geoPosLat;
    double sourceValue = source[idx + data.xSize*idy];


    if(idx < data.xSize && idy < data.ySize){
        if(sourceValue != data.noDataValue){
            //everything that TileGeoReferencing.getGeoPos does.
            geoPosLon = data.m00*(idx + 0.5) + data.m01*(idy + 0.5) + data.m02;
            geoPosLat = data.m10*(idx + 0.5) + data.m11*(idy + 0.5) + data.m12;
            geoPosLon = geoPosLon;
            geoPosLat = geoPosLat;

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
