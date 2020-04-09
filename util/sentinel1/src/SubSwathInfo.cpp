#include "Sentinel1Utils.hpp"

namespace slap {

SubSwathInfo::SubSwathInfo(){

}
SubSwathInfo::~SubSwathInfo(){
    if(dopplerRate != NULL){
        delete[] dopplerRate;
    }
    if(dopplerCentroid != NULL){
        delete[] dopplerCentroid;
    }
    if(rangeDependDopplerRate != NULL){
        delete[] rangeDependDopplerRate;
    }
    if(referenceTime != NULL){
        delete[] referenceTime;
    }
    deviceFree();
}

void SubSwathInfo::hostToDevice(){
    this->dopplerSizeX = this->numOfBursts;
    this->dopplerSizeY = this->samplesPerBurst;
    int elems = this->dopplerSizeX * this->dopplerSizeY;

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceDopplerRate, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceDopplerCentroid, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceReferenceTime, elems*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceRangeDependDopplerRate, elems*sizeof(double)));


    CHECK_CUDA_ERR(cudaMemcpy(this->deviceDopplerRate, this->dopplerRate[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceDopplerCentroid, this->dopplerCentroid[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceReferenceTime, this->referenceTime[0], elems*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceRangeDependDopplerRate, this->rangeDependDopplerRate[0], elems*sizeof(double),cudaMemcpyHostToDevice));

}
void SubSwathInfo::deviceToHost(){

    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);

}
void SubSwathInfo::deviceFree(){
    if(deviceBurstFirstLineTime != NULL){
        cudaFree(deviceBurstFirstLineTime);
    }
    if(deviceBurstLastLineTime != NULL){
        cudaFree(deviceBurstLastLineTime);
    }

    if(deviceDopplerRate != NULL){
        cudaFree(deviceDopplerRate);
    }
    if(deviceDopplerCentroid != NULL){
        cudaFree(deviceDopplerCentroid);
    }
    if(deviceReferenceTime != NULL){
        cudaFree(deviceReferenceTime);
    }
    if(deviceRangeDependDopplerRate != NULL){
        cudaFree(deviceRangeDependDopplerRate);
    }
}

}//namespace
