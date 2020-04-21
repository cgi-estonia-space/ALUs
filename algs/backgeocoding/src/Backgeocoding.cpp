#include "Backgeocoding.cuh"

namespace slap {

Backgeocoding::~Backgeocoding(){
    cudaFree(deviceDemodI);
    cudaFree(deviceDemodQ);
    cudaFree(deviceDemodPhase);
    cudaFree(deviceXPoints);
    cudaFree(deviceYPoints);
    cudaFree(deviceIResults);
    cudaFree(deviceQResults);
    cudaFree(deviceParams);
    cudaFree(deviceSlaveI);
    cudaFree(deviceSlaveQ);

    if(demodI != nullptr){
        delete[] demodI;
    }
    if(demodQ != nullptr){
        delete[] demodQ;
    }
    if(params != nullptr){
        delete[] params;
    }
    if(qResult != nullptr){
        delete[] qResult;
    }
    if(iResult != nullptr){
        delete[] iResult;
    }
    if(xPoints != nullptr){
        delete[] xPoints;
    }
    if(yPoints != nullptr){
        delete[] yPoints;
    }
}

void Backgeocoding::allocateGPUData(){

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceDemodI, this->demodSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceDemodQ, this->demodSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceDemodPhase, this->demodSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceSlaveI, this->demodSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceSlaveQ, this->demodSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceXPoints, this->tileSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceYPoints, this->tileSize*sizeof(double)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceIResults, this->tileSize*sizeof(float)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceQResults, this->tileSize*sizeof(float)));

    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceParams, this->paramSize*sizeof(int)));

}

void Backgeocoding::copySlaveTiles(double *slaveTileI, double *slaveTileQ){

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceSlaveI, slaveTileI, this->demodSize*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceSlaveQ, slaveTileQ, this->demodSize*sizeof(double),cudaMemcpyHostToDevice));

}

void Backgeocoding::copyGPUData(){

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceXPoints, this->xPoints, this->tileSize*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceYPoints, this->yPoints, this->tileSize*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceParams, this->params, this->paramSize*sizeof(int),cudaMemcpyHostToDevice));

}

void Backgeocoding::feedPlaceHolders(){
    std::ifstream xPointsStream(xPointsFile);
    std::ifstream yPointsStream(yPointsFile);
    if(!xPointsStream.is_open()){
        throw std::ios::failure("X Points file not open.");
    }
    if(!yPointsStream.is_open()){
        throw std::ios::failure("Y Points file not open.");
    }
    this->tileX = 100;
    this->tileY = 100;
    this->tileSize = this->tileX * this->tileY;

    this->xPoints = new double[this->tileSize];
    this->yPoints = new double[this->tileSize];

    for(int i=0; i<this->tileSize; i++){
        xPointsStream >> xPoints[i];
        yPointsStream >> yPoints[i];
    }

    xPointsStream.close();
    yPointsStream.close();

    std::ifstream paramStream(paramsFile);
    if(!paramStream.is_open()){
        throw std::ios::failure("Params file not open.");
    }
    this->paramSize = 15;
    this->params = new int[15];

    for(int i=0; i<this->paramSize; i++){
        paramStream >> params[i];
    }

    paramStream.close();

    this->demodSize = 108*108;
}

void Backgeocoding::prepareToCompute(){

    this->allocateGPUData();

    std::cout << "making new results with size:" << this->tileSize << '\n';
    this->qResult = new float[this->tileSize];
    this->iResult = new float[this->tileSize];
    this->slaveUtils = std::make_unique<Sentinel1Utils>();
    this->slaveUtils->setPlaceHolderFiles(this->orbitStateVectorsFile, this->dcEstimateListFile, this->azimuthListFile);
    this->slaveUtils->computeDopplerRate();
    this->slaveUtils->computeReferenceTime();
    this->slaveUtils->subSwath[0].hostToDevice();

}

void Backgeocoding::computeTile(Rectangle slaveRect, double *slaveTileI, double *slaveTileQ){

    this->copySlaveTiles(slaveTileI, slaveTileQ);

    this->copyGPUData();

    CHECK_CUDA_ERR(this->launchDerampDemod(slaveRect));

    CHECK_CUDA_ERR(this->launchBilinear());

    this->getGPUEndResults();

}

void Backgeocoding::getGPUEndResults(){

    CHECK_CUDA_ERR(cudaMemcpy(this->iResult, this->deviceIResults, this->tileSize*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(cudaMemcpy(this->qResult, this->deviceQResults, this->tileSize*sizeof(float), cudaMemcpyDeviceToHost));

}

void Backgeocoding::setPlaceHolderFiles(std::string paramsFile,std::string xPointsFile, std::string yPointsFile){
    this->paramsFile = paramsFile;
    this->xPointsFile = xPointsFile;
    this->yPointsFile = yPointsFile;
}

void Backgeocoding::setSentinel1Placeholders(std::string orbitStateVectorsFile, std::string dcEstimateListFile, std::string azimuthListFile){
    this->orbitStateVectorsFile = orbitStateVectorsFile;
    this->dcEstimateListFile = dcEstimateListFile;
    this->azimuthListFile = azimuthListFile;
}

}//namespace
