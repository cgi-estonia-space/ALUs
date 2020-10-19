/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "earth_gravitational_model96.h"

namespace alus {
namespace snapengine {

EarthGravitationalModel96::EarthGravitationalModel96(std::string grid_file){
    this->grid_file_ = grid_file;
    this->ReadGridFile();

}

EarthGravitationalModel96::EarthGravitationalModel96(){ this->ReadGridFile();
}

EarthGravitationalModel96::~EarthGravitationalModel96(){
    if(this->egm_ != nullptr){
        delete[] this->egm_;
    }
    this->DeviceFree();
}


void EarthGravitationalModel96::ReadGridFile(){
    std::ifstream grid_reader(this->grid_file_);
    if(!grid_reader.is_open()){
        throw std::ios::failure("Grid file not open.");
    }
    this->egm_ = Allocate2DArray<float>(earthgravitationalmodel96::NUM_LATS, earthgravitationalmodel96::NUM_LONS);

    int numCharInHeader = earthgravitationalmodel96::NUM_CHAR_PER_NORMAL_LINE + earthgravitationalmodel96::NUM_CHAR_PER_EMPTY_LINE;

    grid_reader.seekg(numCharInHeader, grid_reader.beg);

    for(int rowIdx=0; rowIdx<earthgravitationalmodel96::NUM_LATS; rowIdx++){
        for(int colIdx=0; colIdx<earthgravitationalmodel96::NUM_LONS; colIdx++){
            grid_reader >> this->egm_[rowIdx][colIdx];
        }
    }

    grid_reader.close();
}

void EarthGravitationalModel96::HostToDevice(){
    CHECK_CUDA_ERR(cudaMalloc((void**)&device_egm_, earthgravitationalmodel96::NUM_LATS*earthgravitationalmodel96::NUM_LONS*sizeof(float)));

    CHECK_CUDA_ERR(cudaMemcpy(this->device_egm_, this->egm_[0], earthgravitationalmodel96::NUM_LATS*earthgravitationalmodel96::NUM_LONS*sizeof(float),cudaMemcpyHostToDevice));
}

void EarthGravitationalModel96::DeviceToHost(){
    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);
}

void EarthGravitationalModel96::DeviceFree(){
    if(this->device_egm_ != nullptr){
        cudaFree(this->device_egm_);
        this->device_egm_ = nullptr;
    }
}

}//namespace
}//namespace
