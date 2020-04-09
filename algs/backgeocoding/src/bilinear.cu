#pragma once
#include <stdio.h>

namespace slap{

inline __device__ void computeIndex(const double x,const double y, const int width,const int height, double *indexI, double *indexJ, double *indexKi, double *indexKj){
	int i0 = (int)x;
	int j0 = (int)y;
	double di = x - (i0 + 0.5);
	double dj = y - (j0 + 0.5);

	int iMax = width - 1;
	int jMax = 0;

	if (di >= 0.0) {
		jMax = i0 + 1;
		indexI[0] = i0 < 0 ? 0.0 : (i0 > iMax ? iMax : i0);
		indexI[1] = jMax < 0 ? 0.0 : (jMax > iMax ? iMax : jMax);
		indexKi[0] = di;
	}else {
		jMax = i0 - 1;
		indexI[0] = jMax < 0 ? 0.0 : (jMax > iMax ? iMax : jMax);
		indexI[1] = i0 < 0 ? 0.0 : (i0 > iMax ? iMax : i0);
		indexKi[0] = di + 1.0;
	}

	jMax = height - 1;
	int j1 = 0;

	if (dj >= 0.0) {
		j1 = j0 + 1;
		indexJ[0] = j0 < 0 ? 0.0 : (j0 > jMax ? jMax : j0);
		indexJ[1] = j1 < 0 ? 0.0 : (j1 > jMax ? jMax : j1);
		indexKj[0] = dj;
	}else {
		j1 = j0 - 1;
		indexJ[0] = j1 < 0 ? 0.0 : (j1 > jMax ? jMax : j1);
		indexJ[1] = j0 < 0 ? 0.0 : (j0 > jMax ? jMax : j0);
		indexKj[0] = dj + 1.0;
	}
}

inline __device__ int getSamples(double *values,int *x, int * y, double *samples, int width, int height, int valueWidth, int valueHeight, double noValue, int useNoData)
{
	int i = 0, j=0, isValid = 1;
	while (i < height) {
		j = 0;
		while (j < width) {
			samples[i*width + j] = values[valueWidth*y[i] + x[j]];
			if(useNoData){
				if(noValue == samples[i*width + j]){
				    isValid=0;
				}
			}
			++j;
		}
		++i;
	}
	return isValid;
}

inline __device__ double resample(double * values, const int valueWidth, const int valueHeight, double * indexI, double * indexJ, double * indexKi, double * indexKj, double noValue, int useNoData)
{
	int x[2] = { (int)indexI[0], (int)indexI[1] };
	int y[2] = { (int)indexJ[0], (int)indexJ[1] };
	double samples[2][2];
	samples[0][0] = 0.0;
	if(getSamples(values,x, y, samples[0], 2, 2, valueWidth, valueHeight, noValue, useNoData)){
        double ki = indexKi[0];
        double kj = indexKj[0];
        return samples[0][0] * (1.0 - ki) * (1.0 - kj) + samples[0][1] * ki * (1.0 - kj) + samples[1][0] * (1.0 - ki) * kj + samples[1][1] * ki * kj;
	}else{
	    return samples[0][0];
	}
}

__global__ void bilinearInterpolation(double *xPixels,
                                    double *yPixels,
                                    double *demodPhase,
                                    double *demodI,
                                    double *demodQ,
                                    int *intParams,
                                    double doubleParams,
                                    float *resultsI,
                                    float *resultsQ){
	double indexI[2];
	double indexJ[2];
	double indexKi[1];
	double indexKj[1];

	const int pointWidth = intParams[0];
	const int pointHeight = intParams[1];
	const int demodWidth = intParams[2];
	const int demodHeight = intParams[3];
	const int startX = intParams[4];
	const int startY = intParams[5];
	const int rectangleX = intParams[10];
	const int rectangleY = intParams[11];
	const int disableReramp = intParams[12];
	const int subswathStart = intParams[13];
	const int subswathEnd = intParams[14];
	//TODO: this needs to come from tile information
	const int useNoDataPhase = 0;
	const int useNoDataI = 0;
	const int useNoDataQ = 0;

	const double noDataValue = doubleParams;

	const int idx = threadIdx.x + (blockDim.x*blockIdx.x);
	const int idy = threadIdx.y + (blockDim.y*blockIdx.y);
	const double x = xPixels[(idy*pointWidth) + idx];
	const double y = yPixels[(idy*pointWidth) + idx];
	double samplePhase = 0.0;
	double sampleI = 0.0;
	double sampleQ = 0.0;
	double cosPhase = 0.0;
	double sinPhase = 0.0;
	double rerampRemodI = 0.0;
	double rerampRemodQ = 0.0;

    //y stride + x index
    const int targetIndex = (startX+idx) - (intParams[8] - (((startY + idy) - intParams[9]) * intParams[7] + intParams[6]));

	if(idx < pointWidth && idy < pointHeight){

	    if((x==noDataValue && y==noDataValue) || !(y >= subswathStart && y < subswathEnd)){
	        resultsI[(idy*pointWidth) + idx] = noDataValue;
	        resultsQ[(idy*pointWidth) + idx] = noDataValue;
	    }else{
	        computeIndex(x - rectangleX + 0.5, y - rectangleY + 0.5, demodWidth, demodHeight, indexI, indexJ, indexKi, indexKj);
        	samplePhase = resample(demodPhase, demodWidth, demodHeight, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataPhase);
        	sampleI = resample(demodI, demodWidth, demodHeight, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataI);
        	sampleQ = resample(demodQ, demodWidth, demodHeight, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataQ);

        	if(!disableReramp){
                cosPhase = cos(samplePhase);
                sinPhase = sin(samplePhase);
                rerampRemodI = sampleI * cosPhase + sampleQ * sinPhase;
                rerampRemodQ = -sampleI * sinPhase + sampleQ * cosPhase;
                resultsI[targetIndex] = rerampRemodI;
                resultsQ[targetIndex] = rerampRemodQ;
				/*if(targetIndex == 4810){
					printf("current sampleI: %f, sampleQ: %f, rerampI: %f, rerampQ: %f\n", sampleI, sampleQ, rerampRemodI, rerampRemodQ);
					printf("Sample Phase: %f \n", samplePhase);
				}*/
            }else{
                resultsI[targetIndex] = sampleI;
                resultsQ[targetIndex] = sampleQ;
        	}
	    }
	}
}

}//namespace
