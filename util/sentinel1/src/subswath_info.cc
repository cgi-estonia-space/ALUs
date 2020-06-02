#include "subswath_info.h"

namespace alus {

SubSwathInfo::SubSwathInfo(){

}
SubSwathInfo::~SubSwathInfo(){
    if(dopplerRate != nullptr){
        delete[] dopplerRate;
    }
    if(dopplerCentroid != nullptr){
        delete[] dopplerCentroid;
    }
    if(rangeDependDopplerRate != nullptr){
        delete[] rangeDependDopplerRate;
    }
    if(referenceTime != nullptr){
        delete[] referenceTime;
    }

    if(azimuthTime != nullptr){
        delete[] azimuthTime;
    }
    if(slantRangeTime != nullptr){
        delete[] slantRangeTime;
    }
    if(latitude != nullptr){
        delete[] latitude;
    }
    if(longitude != nullptr){
        delete[] longitude;
    }
    if(incidenceAngle != nullptr){
        delete[] incidenceAngle;
    }
    if(burstFirstLineTime != nullptr){
        delete[] burstFirstLineTime;
    }
    if(burstLastLineTime != nullptr){
        delete[] burstLastLineTime;
    }
    deviceFree();
}

void SubSwathInfo::hostToDevice(){
    int dopplerSizeX = this->numOfBursts;
    int dopplerSizeY = this->samplesPerBurst;
    int elems = dopplerSizeX * dopplerSizeY;
    DeviceSubswathInfo tempPack;

    CHECK_CUDA_ERR(cudaMalloc((void**)&tempPack.deviceDopplerRate, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&tempPack.deviceDopplerCentroid, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&tempPack.deviceReferenceTime, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&tempPack.deviceRangeDependDopplerRate, elems*sizeof(double)));


    CHECK_CUDA_ERR(cudaMemcpy(tempPack.deviceDopplerRate, this->dopplerRate[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(tempPack.deviceDopplerCentroid, this->dopplerCentroid[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(tempPack.deviceReferenceTime, this->referenceTime[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(tempPack.deviceRangeDependDopplerRate, this->rangeDependDopplerRate[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    tempPack.firstValidPixel = this->firstValidPixel;
    tempPack.lastValidPixel = this->lastValidPixel;
    tempPack.firstLineTime = this->firstLineTime;
    tempPack.lastLineTime = this->lastLineTime;
    tempPack.slrTimeToFirstPixel = this->slrTimeToFirstPixel;
    tempPack.slrTimeToLastPixel = this->slrTimeToLastPixel;
    tempPack.rangePixelSpacing = this->rangePixelSpacing;
    tempPack.azimuthTimeInterval = this->azimuthTimeInterval;
    tempPack.radarFrequency = this->radarFrequency;
    tempPack.azimuthSteeringRate = this->azimuthSteeringRate;

    tempPack.dopplerSizeX = dopplerSizeX;
    tempPack.dopplerSizeY = dopplerSizeY;

    tempPack.linesPerBurst = this->linesPerBurst;
    tempPack.numOfBursts = this->numOfBursts;
    tempPack.samplesPerBurst = this->samplesPerBurst;

    tempPack.numOfGeoLines = this->numOfGeoLines;
    tempPack.numOfGeoPointsPerLine = this->numOfGeoPointsPerLine;

    this->devicePointersHolder = tempPack;

    CHECK_CUDA_ERR(cudaMalloc((void**)&this->deviceSubswathInfo, sizeof(DeviceSubswathInfo)));
    CHECK_CUDA_ERR(cudaMemcpy(this->deviceSubswathInfo, &tempPack, sizeof(DeviceSubswathInfo),cudaMemcpyHostToDevice));
}
void SubSwathInfo::deviceToHost(){

    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);

}
void SubSwathInfo::deviceFree(){
    if(this->devicePointersHolder.deviceBurstFirstLineTime != nullptr){
        cudaFree(this->devicePointersHolder.deviceBurstFirstLineTime);
        this->devicePointersHolder.deviceBurstFirstLineTime = nullptr;
    }
    if(this->devicePointersHolder.deviceBurstLastLineTime != nullptr){
        cudaFree(this->devicePointersHolder.deviceBurstLastLineTime);
        this->devicePointersHolder.deviceBurstLastLineTime = nullptr;
    }

    if(this->devicePointersHolder.deviceDopplerRate != nullptr){
        cudaFree(this->devicePointersHolder.deviceDopplerRate);
        this->devicePointersHolder.deviceDopplerRate = nullptr;
    }
    if(this->devicePointersHolder.deviceDopplerCentroid != nullptr){
        cudaFree(this->devicePointersHolder.deviceDopplerCentroid);
        this->devicePointersHolder.deviceDopplerCentroid = nullptr;
    }
    if(this->devicePointersHolder.deviceReferenceTime != nullptr){
        cudaFree(this->devicePointersHolder.deviceReferenceTime);
        this->devicePointersHolder.deviceReferenceTime = nullptr;
    }
    if(this->deviceSubswathInfo != nullptr){
        cudaFree(this->deviceSubswathInfo);
        this->deviceSubswathInfo = nullptr;
    }


}

}//namespace
