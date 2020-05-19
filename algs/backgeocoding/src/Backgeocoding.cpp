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
    cudaFree(deviceSubswath);
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

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceXPoints, this->xPoints.data(), this->tileSize*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceYPoints, this->yPoints.data(), this->tileSize*sizeof(double),cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceParams, this->params.data(), this->paramSize*sizeof(int),cudaMemcpyHostToDevice));

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

    this->xPoints.resize(this->tileSize);
    this->yPoints.resize(this->tileSize);

    for(int i=0; i<this->tileSize; i++){
        xPointsStream >> xPoints.at(i);
        yPointsStream >> yPoints.at(i);
    }

    xPointsStream.close();
    yPointsStream.close();

    std::ifstream paramStream(paramsFile);
    if(!paramStream.is_open()){
        throw std::ios::failure("Params file not open.");
    }
    this->paramSize = 15;
    this->params.resize(this->paramSize);

    for(int i=0; i<this->paramSize; i++){
        paramStream >> params.at(i);
    }

    paramStream.close();

    this->demodSize = 108*108;

    this->demSamplingLat = 8.333333333333334E-4;
    this->demSamplingLon = 8.333333333333334E-4;
}

void Backgeocoding::prepareToCompute(){

    this->allocateGPUData();

    std::cout << "making new results with size:" << this->tileSize << '\n';
    this->qResult.resize(this->tileSize);
    this->iResult.resize(this->tileSize);
    this->slaveUtils = std::make_unique<Sentinel1Utils>(2);
    this->slaveUtils->setPlaceHolderFiles(this->orbitStateVectorsFile, this->dcEstimateListFile, this->azimuthListFile, this->burstLineTimeFile, this->geoLocationFile);
    this->slaveUtils->readPlaceHolderFiles();
    this->slaveUtils->computeDopplerRate();
    this->slaveUtils->computeReferenceTime();
    this->slaveUtils->subSwath[0].hostToDevice();

    this->masterUtils = std::make_unique<Sentinel1Utils>(1);
    this->masterUtils->setPlaceHolderFiles(this->orbitStateVectorsFile, this->dcEstimateListFile, this->azimuthListFile, this->burstLineTimeFile, this->geoLocationFile);
    this->masterUtils->readPlaceHolderFiles();
    this->masterUtils->computeDopplerRate();
    this->masterUtils->computeReferenceTime();

    CHECK_CUDA_ERR(cudaMalloc((void**)&this->deviceSubswath, sizeof(SubSwathInfo)));
    CHECK_CUDA_ERR(cudaMemcpy(this->deviceSubswath, &this->slaveUtils->subSwath[0], sizeof(SubSwathInfo),cudaMemcpyHostToDevice));

    this->prepareSrtm3Data();
}

void Backgeocoding::prepareSrtm3Data(){
    int size;

    this->srtms.push_back(Dataset(this->srtm_41_01File));
    this->srtms.at(0).loadRasterBand(1);
    this->srtms.push_back(Dataset(this->srtm_42_01File));
    this->srtms.at(1).loadRasterBand(1);

    this->deviceSrtms.resize(2);
    dim3 blockSize(20,20);

    std::vector<DemFormatterData> datas;

    //placeholders
    DemFormatterData srtm41_01Data;
    srtm41_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m01 = 0;
    srtm41_01Data.m02 = 20;
    srtm41_01Data.m10 = 0;
    srtm41_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m12 = 60;
    srtm41_01Data.noDataValue = -32768.0;

    DemFormatterData srtm42_01Data;
    srtm42_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m01 = 0;
    srtm42_01Data.m02 = 25;
    srtm42_01Data.m10 = 0;
    srtm42_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m12 = 60;
    srtm42_01Data.noDataValue = -32768.0;

    datas.push_back(srtm41_01Data);
    datas.push_back(srtm42_01Data);

    for(int i=0; i<2; i++){
        double *tempBuffer;

        size = this->srtms.at(i).getXSize() * this->srtms.at(i).getYSize();
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->deviceSrtms.at(i), size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&tempBuffer, size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMemcpy(tempBuffer, this->srtms.at(i).getDataBuffer().data(), size*sizeof(double),cudaMemcpyHostToDevice));
        datas.at(i).xSize = this->srtms.at(i).getXSize();
        datas.at(i).ySize = this->srtms.at(i).getYSize();
        dim3 gridSize(getGridDim(blockSize.x, datas.at(i).xSize),getGridDim(blockSize.y, datas.at(i).ySize));

        CHECK_CUDA_ERR(launchDemFormatter(gridSize, blockSize, this->deviceSrtms.at(i), tempBuffer, datas.at(i) ));
        cudaFree(tempBuffer);
    }

}

void Backgeocoding::computeTile(Rectangle slaveRect, double *slaveTileI, double *slaveTileQ){

    this->copySlaveTiles(slaveTileI, slaveTileQ);

    this->copyGPUData();

    std::vector<double> extendedAmount;
    extendedAmount.push_back(-0.01773467106249882);
    extendedAmount.push_back(0.0);
    extendedAmount.push_back(-3.770974349203243);
    extendedAmount.push_back(3.8862058607542167);


    this->computeSlavePixPos(11,11,4000,17000,100,100,extendedAmount);

    //TODO: using placeholder as number 11
    CHECK_CUDA_ERR(this->launchDerampDemodComp(slaveRect,11));

    CHECK_CUDA_ERR(this->launchBilinearComp());

    this->getGPUEndResults();
    std::cout << "all computations ended." << '\n';
}

void Backgeocoding::getGPUEndResults(){

    CHECK_CUDA_ERR(cudaMemcpy(this->iResult.data(), this->deviceIResults, this->tileSize*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(cudaMemcpy(this->qResult.data(), this->deviceQResults, this->tileSize*sizeof(float), cudaMemcpyDeviceToHost));

}


void Backgeocoding::computeSlavePixPos(
        int mBurstIndex,
        int sBurstIndex,
        int x0,
        int y0,
        int w,
        int h,
        std::vector<double> extendedAmount){
//        double **slavePixelPosAz,
//        double **slavePixelPosRg){ add those later.

    SlavePixPosData calcData;
    calcData.mBurstIndex = mBurstIndex;
    calcData.sBurstIndex = sBurstIndex;
    int xmin = x0 - (int)extendedAmount.at(3);
    int ymin = y0 - (int)extendedAmount.at(1);
    int ymax = y0 + h + (int)abs(extendedAmount.at(0));
    int xmax = x0 + w + (int)abs(extendedAmount.at(2));

    std::vector<double> latLonMinMax = this->computeImageGeoBoundary(&this->masterUtils->subSwath[0],mBurstIndex,xmin, xmax, ymin, ymax);

    double delta = fmax(this->demSamplingLat, this->demSamplingLon);
    double extralat = 20*delta;
    double extralon = 20*delta;

    double latMin = latLonMinMax.at(0) - extralat;
    double latMax = latLonMinMax.at(1) + extralat;
    double lonMin = latLonMinMax.at(2) - extralon;
    double lonMax = latLonMinMax.at(3) + extralon;

    double upperLeftX = (lonMin + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double upperLeftY = (60.0 - latMax) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double lowerRightX = (lonMax + 180.0) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;
    double lowerRightY = (60.0 - latMin) * snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILEinv;

    calcData.latMaxIdx = (int)floor(upperLeftY);
    calcData.latMinIdx = (int)ceil(lowerRightY);
    calcData.lonMinIdx = (int)floor(upperLeftX);
    calcData.lonMaxIdx = (int)ceil(lowerRightX);

    calcData.numLines = calcData.latMinIdx - calcData.latMaxIdx;
    calcData.numPixels = calcData.lonMaxIdx - calcData.lonMinIdx;

    CHECK_CUDA_ERR(this->launchSlavePixPosComp(calcData));
}

//usually we use the subswath from master product.
std::vector<double> Backgeocoding::computeImageGeoBoundary(SubSwathInfo *subSwath, int burstIndex,int xMin, int xMax, int yMin, int yMax){
    std::vector<double> results;
    results.resize(4);

    double azTimeMin = subSwath->burstFirstLineTime[burstIndex] +
            (yMin - burstIndex * subSwath->linesPerBurst) * subSwath->azimuthTimeInterval;

    double azTimeMax = subSwath->burstFirstLineTime[burstIndex] +
            (yMax - burstIndex * subSwath->linesPerBurst) * subSwath->azimuthTimeInterval;

    double rgTimeMin = subSwath->slrTimeToFirstPixel + xMin * masterUtils->rangeSpacing / snapEngine::constants::lightSpeed;

    double rgTimeMax = subSwath->slrTimeToFirstPixel + xMax * masterUtils->rangeSpacing / snapEngine::constants::lightSpeed;

    double latUL = masterUtils->getLatitude(azTimeMin, rgTimeMin, subSwath);
    double lonUL = masterUtils->getLongitude(azTimeMin, rgTimeMin, subSwath);
    double latUR = masterUtils->getLatitude(azTimeMin, rgTimeMax, subSwath);
    double lonUR = masterUtils->getLongitude(azTimeMin, rgTimeMax, subSwath);
    double latLL = masterUtils->getLatitude(azTimeMax, rgTimeMin, subSwath);
    double lonLL = masterUtils->getLongitude(azTimeMax, rgTimeMin, subSwath);
    double latLR = masterUtils->getLatitude(azTimeMax, rgTimeMax, subSwath);
    double lonLR = masterUtils->getLongitude(azTimeMax, rgTimeMax, subSwath);

    double latMin = 90.0;
    double latMax = -90.0;
    double lonMin = 180.0;
    double lonMax = -180.0;

    std::vector<double> lats {latMin, latUL, latUR, latLL, latLR, latMax};
    std::vector<double> lons {lonMin, lonUL, lonUR, lonLL, lonLR, lonMax};

    latMin = *std::min_element(lats.begin(), lats.end()-1);
    latMax = *std::max_element(lats.begin()+1, lats.end());
    lonMin = *std::min_element(lons.begin(), lons.end()-1);
    lonMax = *std::max_element(lons.begin()+1, lons.end());

    results.at(0) = latMin;
    results.at(1) = latMax;
    results.at(2) = lonMin;
    results.at(3) = lonMax;

    return results;
}

void Backgeocoding::setPlaceHolderFiles(std::string paramsFile,std::string xPointsFile, std::string yPointsFile){
    this->paramsFile = paramsFile;
    this->xPointsFile = xPointsFile;
    this->yPointsFile = yPointsFile;
}

void Backgeocoding::setSRTMPlaceholders(std::string srtm_41_01File, std::string srtm_42_01File){
    this->srtm_41_01File = srtm_41_01File;
    this->srtm_42_01File = srtm_42_01File;
}

void Backgeocoding::setSentinel1Placeholders(
        std::string orbitStateVectorsFile,
        std::string dcEstimateListFile,
        std::string azimuthListFile,
        std::string burstLineTimeFile,
        std::string geoLocationFile){


    this->orbitStateVectorsFile = orbitStateVectorsFile;
    this->dcEstimateListFile = dcEstimateListFile;
    this->azimuthListFile = azimuthListFile;
    this->burstLineTimeFile = burstLineTimeFile;
    this->geoLocationFile = geoLocationFile;
}

cudaError_t Backgeocoding::launchBilinearComp(){
    cudaError_t status;
    dim3 gridSize(5,5);
    dim3 blockSize(20,20);

    launchBilinearInterpolation(
        gridSize,
        blockSize,
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

cudaError_t Backgeocoding::launchDerampDemodComp(Rectangle slaveRect, int sBurstIndex){
    cudaError_t status;
    dim3 gridSize(6,6);
    dim3 blockSize(20,20);

    launchDerampDemod(
        gridSize,
        blockSize,
        slaveRect,
        this->deviceSlaveI,
        this->deviceSlaveQ,
        this->deviceDemodPhase,
        this->deviceDemodI,
        this->deviceDemodQ,
        this->deviceSubswath,
        sBurstIndex
    );
    status = cudaGetLastError();
    cudaFree(deviceSubswath);

    return status;
}

cudaError_t Backgeocoding::launchSlavePixPosComp(SlavePixPosData calcData){
    dim3 blockSize(20,20);
    dim3 gridSize(getGridDim(20,calcData.numLines), getGridDim(20, calcData.numPixels));


    launchSlavePixPos(
        gridSize,
        blockSize,
        calcData
    );
    return cudaGetLastError();
}

}//namespace
