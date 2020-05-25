#include "bilinear.cuh"

#include "pointer_holders.h"
#include "bilinear_interpolation.cuh"

namespace alus{

inline __device__ int getSamples(PointerArray *tiles,int *x, int *y, double *samples, int width, int height, double noValue, int useNoData)
{
	double *values = (double*)tiles->array[0].pointer;
	const int valueWidth = tiles->array[0].x;
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

	PointerArray pArray;
	PointerHolder pHolder;
	pArray.array = &pHolder;
	pHolder.x = demodWidth;
	pHolder.y = demodHeight;

    //y stride + x index
    const int targetIndex = (startX+idx) - (intParams[8] - (((startY + idy) - intParams[9]) * intParams[7] + intParams[6]));

	if(idx < pointWidth && idy < pointHeight){

	    if((x==noDataValue && y==noDataValue) || !(y >= subswathStart && y < subswathEnd)){
	        resultsI[(idy*pointWidth) + idx] = noDataValue;
	        resultsQ[(idy*pointWidth) + idx] = noDataValue;
	    }else{
	        snapengine::bilinearinterpolation::computeIndex(x - rectangleX + 0.5, y - rectangleY + 0.5, demodWidth, demodHeight, indexI, indexJ, indexKi, indexKj);
			pHolder.pointer = demodPhase;
        	samplePhase = snapengine::bilinearinterpolation::resample(&pArray, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataPhase, getSamples);
			pHolder.pointer = demodI;
        	sampleI = snapengine::bilinearinterpolation::resample(&pArray, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataI, getSamples);
			pHolder.pointer = demodQ;
        	sampleQ = snapengine::bilinearinterpolation::resample(&pArray, indexI, indexJ, indexKi, indexKj, noDataValue, useNoDataQ, getSamples);

        	if(!disableReramp){
                cosPhase = cos(samplePhase);
                sinPhase = sin(samplePhase);
                rerampRemodI = sampleI * cosPhase + sampleQ * sinPhase;
                rerampRemodQ = -sampleI * sinPhase + sampleQ * cosPhase;
                resultsI[targetIndex] = rerampRemodI;
                resultsQ[targetIndex] = rerampRemodQ;
            }else{
                resultsI[targetIndex] = sampleI;
                resultsQ[targetIndex] = sampleQ;
        	}
	    }
	}
}

cudaError_t launchBilinearInterpolation(dim3 gridSize,
						dim3 blockSize,
						double *xPixels,
                        double *yPixels,
                        double *demodPhase,
                        double *demodI,
                        double *demodQ,
                        int *intParams,
                        double doubleParams,
                        float *resultsI,
                        float *resultsQ){


    bilinearInterpolation<<<gridSize, blockSize>>>(
        xPixels,
        yPixels,
        demodPhase,
        demodI,
        demodQ,
        intParams,
        doubleParams,
        resultsI,
        resultsQ
    );
    return cudaGetLastError();

}

}//namespace
