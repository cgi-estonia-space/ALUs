#include "slave_pixpos.cuh"

namespace alus {

inline __device__ double getElevation(double geoPosLat, double geoPosLon, double NO_DATA_VALUE, double DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv){
    if (geoPosLon > 180) {
        geoPosLat -= 360;
    }

    double pixelY = (60.0 - geoPosLat) * DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    if (pixelY < 0 || isnan(pixelY)) {
        return NO_DATA_VALUE;
    }

    double elevation = 0.0;

    /*Resampling.Index newIndex = resampling.createIndex();
    resampling.computeCornerBasedIndex(getIndexX(geoPos), pixelY, RASTER_WIDTH, RASTER_HEIGHT, newIndex);
    elevation = resampling.resample(resamplingRaster, newIndex);*/

    return isnan(elevation) ? NO_DATA_VALUE : elevation;
}

//exclusively supports SRTM3 digital elevation map and none other
__global__ void slavePixPos(SlavePixPosData calcData){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
	const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    double geoPosLat;
    double geoPosLon;
    double alt;


    if(idx < calcData.numLines && idy < calcData.numPixels){
        geoPosLat = (snapengine::srtm3elevationmodel::RASTER_HEIGHT - calcData.lonMinIdx + idy) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 60.0;
        geoPosLon = (calcData.latMaxIdx + idx) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE - 180.0;

        alt= getElevation(geoPosLat, geoPosLon, snapengine::srtm3elevationmodel::NO_DATA_VALUE, snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv);
        if(idx==0 && idy == 0){
            printf("%f\n", alt); //just getting rid of warnings. Continue working from here.
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
