#include "Backgeocoding.cuh"
#include "bilinear.cu"
#include "derampDemod.cu"

namespace slap{

cudaError_t Backgeocoding::launchBilinear(){
    cudaError_t status;
    dim3 gridSize(5,5);
    dim3 blockSize(20,20);

    bilinearInterpolation<<<gridSize, blockSize>>>(
        this->deviceXPoints,
        this->deviceYPoints,
        this->deviceDemodPhase,
        this->deviceDemodI,
        this->deviceDemodQ,
        this->deviceParams,
        0.0,
        this->deviceIResults,
        this->deviceQResults
    );
    status = cudaGetLastError();

    return status;
}

//TODO: using placeholder as number 11
cudaError_t Backgeocoding::launchDerampDemod(Rectangle slaveRect){
    cudaError_t status;
    dim3 gridSize(6,6);
    dim3 blockSize(20,20);

    derampDemod<<<gridSize, blockSize>>>(
        slaveRect,
        this->deviceSlaveI,
        this->deviceSlaveQ,
        this->deviceDemodPhase,
        this->deviceDemodI,
        this->deviceDemodQ,
        this->slaveUtils->subSwath[0],
        11
    );
    status = cudaGetLastError();

    return status;
}

}//namespace
