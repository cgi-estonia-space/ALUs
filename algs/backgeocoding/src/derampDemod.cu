#pragma once

#include "Constants.hpp"

namespace slap {
__global__ void derampDemod(Rectangle rectangle, double *slaveI, double*slaveQ, double* demodPhase, double *demodI, double *demodQ, SubSwathInfo subSwath, int sBurstIndex){
    const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
	const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
    const int globalIndex = rectangle.width * idy + idx;
    const int firstLineInBurst = sBurstIndex* subSwath.linesPerBurst;
    const int y = rectangle.y + idy;
    const int x = rectangle.x + idx;
    double ta, kt, deramp, demod;
    double valueI, valueQ, valuePhase, cosPhase, sinPhase;

    if(idx < rectangle.width && idy < rectangle.height){

        ta = (y - firstLineInBurst)* subSwath.azimuthTimeInterval;
        kt = subSwath.deviceDopplerRate[sBurstIndex*subSwath.dopplerSizeY + x];
        deramp = -snapEngine::constants::PI * kt * pow(ta - subSwath.deviceReferenceTime[sBurstIndex*subSwath.dopplerSizeY + x],2);
        demod = -snapEngine::constants::TWO_PI * subSwath.deviceDopplerCentroid[sBurstIndex*subSwath.dopplerSizeY + x] * ta;
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

}//namespace
