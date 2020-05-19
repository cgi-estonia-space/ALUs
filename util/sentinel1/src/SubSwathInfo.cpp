#include "Sentinel1Utils.hpp"

namespace slap {

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
    if(deviceBurstFirstLineTime != nullptr){
        cudaFree(deviceBurstFirstLineTime);
    }
    if(deviceBurstLastLineTime != nullptr){
        cudaFree(deviceBurstLastLineTime);
    }

    if(deviceDopplerRate != nullptr){
        cudaFree(deviceDopplerRate);
    }
    if(deviceDopplerCentroid != nullptr){
        cudaFree(deviceDopplerCentroid);
    }
    if(deviceReferenceTime != nullptr){
        cudaFree(deviceReferenceTime);
    }
    if(deviceRangeDependDopplerRate != nullptr){
        cudaFree(deviceRangeDependDopplerRate);
    }
}

}//namespace
