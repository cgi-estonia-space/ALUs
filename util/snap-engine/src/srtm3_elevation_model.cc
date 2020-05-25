#include "srtm3_elevation_model.h"

namespace alus{
namespace snapengine{

SRTM3ElevationModel::SRTM3ElevationModel(std::vector<Point> fileIndexes, std::string directory){
    this->fileIndexes_ = fileIndexes;
    this->filesDirectory_ = directory;
}

SRTM3ElevationModel::~SRTM3ElevationModel(){
    this->deviceFree();
}

//use a preconfigured emg96 instance with the tiles already loaded onto the gpu.
void SRTM3ElevationModel::ReadSrtmTiles(EarthGravitationalModel96 *egm96){

    this->ResolveFileNames();

    this->deviceSrtms_.resize(this->nrOfTiles_);

    for(int i=0; i<this->nrOfTiles_; i++){

        std::string imageFile = this->fileTemplates_.at(i);
        imageFile.append(this->tifExtension_);
        std::string tfwFile = this->fileTemplates_.at(i);
        tfwFile.append(this->tfwExtension_);

        this->srtms_.push_back(Dataset(imageFile));
        this->srtms_.at(i).loadRasterBand(1);

        std::ifstream tfwReader(tfwFile);
        if(!tfwReader.is_open()){
            std::stringstream errMsg;
            errMsg << "Can not open TFW file " << tfwFile << '\n';
            throw std::ios::failure(errMsg.str().c_str());
        }

        DemFormatterData srtmData;
        tfwReader >> srtmData.m00 >> srtmData.m10 >> srtmData.m01 >> srtmData.m11 >> srtmData.m02 >> srtmData.m12;
        srtmData.noDataValue = -32768.0;
        srtmData.maxLats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
        srtmData.maxLons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
        srtmData.egm = egm96->deviceEgm;

        tfwReader.close();
        this->datas_.push_back(srtmData);
    }
}
//keep it for accuracy tests. Not sure if first 4 values(by column) are calcuated or read.
//not sure if the last 2 values get their decimals ripped of or not.
//once backgeocoding is completed and SRTM3 tile processing is automated and without placeholders, remove this.
std::vector<DemFormatterData> SRTM3ElevationModel::srtmPlaceholderData(EarthGravitationalModel96 *egm96){

    std::vector<DemFormatterData> result;

    DemFormatterData srtm41_01Data;
    srtm41_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m01 = 0;
    srtm41_01Data.m02 = 20;
    srtm41_01Data.m10 = 0;
    srtm41_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m12 = 60;
    srtm41_01Data.noDataValue = -32768.0;
    srtm41_01Data.maxLats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
    srtm41_01Data.maxLons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
    srtm41_01Data.egm = egm96->deviceEgm;

    DemFormatterData srtm42_01Data;
    srtm42_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m01 = 0;
    srtm42_01Data.m02 = 25;
    srtm42_01Data.m10 = 0;
    srtm42_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m12 = 60;
    srtm42_01Data.noDataValue = -32768.0;
    srtm42_01Data.maxLats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
    srtm42_01Data.maxLons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
    srtm42_01Data.egm = egm96->deviceEgm;

    result.push_back(srtm41_01Data);
    result.push_back(srtm42_01Data);
    return result;
}

void SRTM3ElevationModel::ResolveFileNames(){

    for(unsigned int i=0; i< this->fileIndexes_.size(); i++){
        Point coords = fileIndexes_.at(i);
        if (coords.x <= srtm3elevationmodel::NUM_X_TILES && coords.x > 0 && coords.y <= srtm3elevationmodel::NUM_Y_TILES && coords.y > 0) {
            this->fileTemplates_.push_back(this->FormatName(coords));
        }else{
            std::stringstream errMsg;
            errMsg << "Tile file at indexes " << coords.x << ":" << coords.y << " does not exist. Indexes start with 1." << '\n';
            throw errMsg.str().c_str();
        }
    }
    this->nrOfTiles_ = this->fileTemplates_.size();
}

std::string SRTM3ElevationModel::FormatName(Point coords){
    char buffer[500];
    std::snprintf(buffer, sizeof(buffer), "%ssrtm_%02d_%02d",this->filesDirectory_.c_str(), coords.x, coords.y);
    std::string result(buffer);
    return result;
}

void SRTM3ElevationModel::hostToDevice(){
    int size;

    std::vector<PointerHolder> tempTiles;
    tempTiles.resize(this->nrOfTiles_);
    dim3 blockSize(20,20);

    for(int i=0; i<this->nrOfTiles_; i++){
        double *tempBuffer;

        size = this->srtms_.at(i).getXSize() * this->srtms_.at(i).getYSize();
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->deviceSrtms_.at(i), size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&tempBuffer, size*sizeof(double)));
        CHECK_CUDA_ERR(cudaMemcpy(tempBuffer, this->srtms_.at(i).getDataBuffer().data(), size*sizeof(double),cudaMemcpyHostToDevice));
        this->datas_.at(i).xSize = this->srtms_.at(i).getXSize();
        this->datas_.at(i).ySize = this->srtms_.at(i).getYSize();
        dim3 gridSize(cuda::getGridDim(blockSize.x, this->datas_.at(i).xSize),cuda::getGridDim(blockSize.y, this->datas_.at(i).ySize));

        CHECK_CUDA_ERR(launchDemFormatter(gridSize, blockSize, this->deviceSrtms_.at(i), tempBuffer, this->datas_.at(i) ));
        tempTiles.at(i).pointer = this->deviceSrtms_.at(i);
        tempTiles.at(i).x = this->srtms_.at(i).getXSize();
        tempTiles.at(i).y = this->srtms_.at(i).getYSize();
        tempTiles.at(i).z = 1;
        cudaFree(tempBuffer);
    }
    CHECK_CUDA_ERR(cudaMalloc((void**)&this->deviceSrtm3Tiles_, this->nrOfTiles_*sizeof(PointerHolder)));
    CHECK_CUDA_ERR(cudaMemcpy(this->deviceSrtm3Tiles_, tempTiles.data(), this->nrOfTiles_*sizeof(PointerHolder), cudaMemcpyHostToDevice));

}

void SRTM3ElevationModel::deviceToHost(){
    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);
}

void SRTM3ElevationModel::deviceFree(){
    if(this->deviceSrtm3Tiles_ != nullptr){
        cudaFree(this->deviceSrtm3Tiles_);
        this->deviceSrtm3Tiles_ = nullptr;
    }

    for(unsigned int i=0; i< this->deviceSrtms_.size(); i++){
        if(this->deviceSrtms_.at(0) != nullptr){
            cudaFree(this->deviceSrtms_.at(0));
            this->deviceSrtms_.at(0) = nullptr;
        }
    }
    this->deviceSrtms_.clear();
}


}//namespace
}//namespace
