#include "slave_pixpos.cuh"

#include "srtm3_elevation_calc.cuh"

namespace alus {

//exclusively supports SRTM3 digital elevation map and none other
__global__ void slavePixPos(SlavePixPosData calcData){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
	const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geoPosLat;
    double geoPosLon;
    double alt;


    if(idx < calcData.numPixels && idy < calcData.numLines){
        geoPosLat = (snapengine::srtm3elevationmodel::RASTER_HEIGHT - calcData.latMaxIdx + idy) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 60.0;
        geoPosLon = (calcData.lonMinIdx + idx) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 180.0;
        
        alt= snapengine::srtm3elevationmodel::getElevation(
            geoPosLat,
            geoPosLon,
            &calcData.tiles
        );
        if(idx==0 && idy == 0){
            printf("altitude number: %f\n", alt); //just getting rid of warnings. Continue working from here.
        }

    }
}

cudaError_t launchSlavePixPos(dim3 gridSize, dim3 blockSize, SlavePixPosData calcData){

    slavePixPos<<<gridSize, blockSize>>>(
        calcData
    );
    return cudaGetLastError();
}

} //namespace
