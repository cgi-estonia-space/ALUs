#include "deramp_demod.cuh"

namespace alus {
__global__ void derampDemod(alus::Rectangle rectangle, double *slaveI, double *slaveQ, double* demodPhase,
                            double *demodI, double *demodQ,
                            alus::DeviceSubswathInfo *subSwath, int sBurstIndex){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    const int globalIndex = rectangle.width * idy + idx;
    const int firstLineInBurst = sBurstIndex * subSwath->linesPerBurst;
    const int y = rectangle.y + idy;
    const int x = rectangle.x + idx;
    double ta, kt, deramp, demod;
    double valueI, valueQ, valuePhase, cosPhase, sinPhase;

    if(idx < rectangle.width && idy < rectangle.height){

        ta = (y - firstLineInBurst)* subSwath->azimuthTimeInterval;
        kt = subSwath->deviceDopplerRate[sBurstIndex*subSwath->dopplerSizeY + x];
        deramp = -alus::snapengine::constants::PI * kt * pow(ta -
                                                           subSwath->deviceReferenceTime[sBurstIndex*subSwath->dopplerSizeY + x],2);
        demod = -alus::snapengine::constants::TWO_PI *
                subSwath->deviceDopplerCentroid[sBurstIndex*subSwath->dopplerSizeY +
                                                                                 x] * ta;
        valuePhase = deramp + demod;

        demodPhase[globalIndex] = valuePhase;

        valueI = slaveI[globalIndex];
        valueQ = slaveQ[globalIndex];

        cosPhase = cos(valuePhase);
        sinPhase = sin(valuePhase);
        demodI[globalIndex] = valueI*cosPhase - valueQ*sinPhase;
        demodQ[globalIndex] = valueI*sinPhase + valueQ*cosPhase;

    }
}

cudaError_t launchDerampDemod(dim3 gridSize,
    dim3 blockSize,
                              alus::Rectangle rectangle,
    double *slaveI,
    double *slaveQ,
    double *demodPhase,
    double *demodI,
    double *demodQ,
                              alus::DeviceSubswathInfo *subSwath,
    int sBurstIndex){

    derampDemod<<<gridSize, blockSize>>>(
        rectangle,
        slaveI,
        slaveQ,
        demodPhase,
        demodI,
        demodQ,
        subSwath,
        sBurstIndex
    );
    return cudaGetLastError();
}

}//namespace
