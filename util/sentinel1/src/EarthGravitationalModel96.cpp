#include "EarthGravitationalModel96.hpp"

namespace slap {

EarthGravitationalModel96::EarthGravitationalModel96(std::string gridFile){
    this->gridFile = gridFile;
    this->readGridFile();

}

EarthGravitationalModel96::EarthGravitationalModel96(){
    this->readGridFile();
}

EarthGravitationalModel96::~EarthGravitationalModel96(){
    if(this->egm != nullptr){
        delete[] this->egm;
    }
    this->deviceFree();
}


void EarthGravitationalModel96::readGridFile(){
    std::ifstream gridReader(this->gridFile);
    if(!gridReader.is_open()){
        throw std::ios::failure("Grid file not open.");
    }
    this->egm = allocate2DDoubleArray(NUM_LATS, NUM_LONS);

    int numCharInHeader = NUM_CHAR_PER_NORMAL_LINE + NUM_CHAR_PER_EMPTY_LINE;

    gridReader.seekg(numCharInHeader, gridReader.beg);

    for(int rowIdx=0; rowIdx<NUM_LATS; rowIdx++){
        for(int colIdx=0; colIdx<NUM_LONS; colIdx++){
            gridReader >> this->egm[rowIdx][colIdx];
        }
    }

    gridReader.close();
}

void EarthGravitationalModel96::hostToDevice(){
    CHECK_CUDA_ERR(cudaMalloc((void**)&deviceEgm, NUM_LATS*NUM_LONS*sizeof(double)));

    CHECK_CUDA_ERR(cudaMemcpy(this->deviceEgm, this->egm[0], NUM_LATS*NUM_LONS*sizeof(double),cudaMemcpyHostToDevice));
}

void EarthGravitationalModel96::deviceToHost(){
    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);
}

void EarthGravitationalModel96::deviceFree(){
    if(this->deviceEgm != nullptr){
        cudaFree(this->deviceEgm);
        this->deviceEgm = nullptr;
    }
}

}//namespace
