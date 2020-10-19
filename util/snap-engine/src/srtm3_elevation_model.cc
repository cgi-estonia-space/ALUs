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
#include "srtm3_elevation_model.h"

#include <sstream>

#include "CudaFriendlyObject.h"
#include "shapes.h"
#include "srtm3_elevation_model_constants.h"

namespace alus{
namespace snapengine{

SRTM3ElevationModel::SRTM3ElevationModel(std::vector<Point> file_indexes, std::string directory){
    this->file_indexes_ = file_indexes;
    this->files_directory_ = directory;
}

SRTM3ElevationModel::~SRTM3ElevationModel(){ this->DeviceFree();
}

//use a preconfigured emg96 instance with the tiles already loaded onto the gpu.
void SRTM3ElevationModel::ReadSrtmTiles(EarthGravitationalModel96 *egm96){

    this->ResolveFileNames();

    this->device_srtms_.resize(this->nr_of_tiles_);

    for(int i=0; i<this->nr_of_tiles_; i++){

        std::string image_file = this->file_templates_.at(i);
        image_file.append(this->tif_extension_);
        std::string tfw_file = this->file_templates_.at(i);
        tfw_file.append(this->tfw_extension_);

        this->srtms_.push_back(Dataset(image_file));
        this->srtms_.at(i).LoadRasterBandFloat(1);

        std::ifstream tfw_reader(tfw_file);
        if(!tfw_reader.is_open()){
            std::stringstream err_msg;
            err_msg << "Can not open TFW file " << tfw_file << '\n';
            throw std::ios::failure(err_msg.str().c_str());
        }

        DemFormatterData srtm_data;
        tfw_reader >> srtm_data.m00 >> srtm_data.m10 >> srtm_data.m01 >> srtm_data.m11 >> srtm_data.m02 >>
            srtm_data.m12;
        srtm_data.no_data_value = -32768.0;
        srtm_data.max_lats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
        srtm_data.max_lons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
        srtm_data.egm = egm96->device_egm_;

        tfw_reader.close();
        this->datas_.push_back(srtm_data);
    }
}
//keep it for accuracy tests. Not sure if first 4 values(by column) are calcuated or read.
//not sure if the last 2 values get their decimals ripped of or not.
//once backgeocoding is completed and SRTM3 tile processing is automated and without placeholders, remove this.
std::vector<DemFormatterData> SRTM3ElevationModel::SrtmPlaceholderData(EarthGravitationalModel96 *egm96){

    std::vector<DemFormatterData> result;

    DemFormatterData srtm41_01Data;
    srtm41_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m01 = 0;
    srtm41_01Data.m02 = 20;
    srtm41_01Data.m10 = 0;
    srtm41_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm41_01Data.m12 = 60;
    srtm41_01Data.no_data_value = -32768.0;
    srtm41_01Data.max_lats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
    srtm41_01Data.max_lons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
    srtm41_01Data.egm = egm96->device_egm_;

    DemFormatterData srtm42_01Data;
    srtm42_01Data.m00 = snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m01 = 0;
    srtm42_01Data.m02 = 25;
    srtm42_01Data.m10 = 0;
    srtm42_01Data.m11 = -snapengine::srtm3elevationmodel::DEGREE_RES_BY_NUM_PIXELS_PER_TILE;
    srtm42_01Data.m12 = 60;
    srtm42_01Data.no_data_value = -32768.0;
    srtm42_01Data.max_lats = alus::snapengine::earthgravitationalmodel96::MAX_LATS;
    srtm42_01Data.max_lons = alus::snapengine::earthgravitationalmodel96::MAX_LONS;
    srtm42_01Data.egm = egm96->device_egm_;

    result.push_back(srtm41_01Data);
    result.push_back(srtm42_01Data);
    return result;
}

void SRTM3ElevationModel::ResolveFileNames(){

    for(unsigned int i=0; i< this->file_indexes_.size(); i++){
        Point coords = file_indexes_.at(i);
        if (coords.x <= srtm3elevationmodel::NUM_X_TILES && coords.x > 0 && coords.y <= srtm3elevationmodel::NUM_Y_TILES && coords.y > 0) {
            this->file_templates_.push_back(this->FormatName(coords));
        }else{
            std::stringstream err_msg;
            err_msg << "Tile file at indexes " << coords.x << ":" << coords.y << " does not exist. Indexes start with 1." << '\n';
            throw err_msg.str().c_str();
        }
    }
    this->nr_of_tiles_ = this->file_templates_.size();
}

std::string SRTM3ElevationModel::FormatName(Point coords){
    char buffer[500];
    std::snprintf(buffer, sizeof(buffer), "%ssrtm_%02d_%02d",this->files_directory_.c_str(), coords.x, coords.y);
    std::string result(buffer);
    return result;
}

void SRTM3ElevationModel::HostToDevice(){
    int size;

    std::vector<PointerHolder> temp_tiles;
    temp_tiles.resize(this->nr_of_tiles_);
    dim3 blockSize(20,20);

    for(int i=0; i<this->nr_of_tiles_; i++){
        float *temp_buffer;

        size = this->srtms_.at(i).GetXSize() * this->srtms_.at(i).GetYSize();
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_srtms_.at(i), size*sizeof(float)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&temp_buffer, size*sizeof(float)));
        CHECK_CUDA_ERR(cudaMemcpy(temp_buffer, this->srtms_.at(i).GetFloatDataBuffer().data(), size*sizeof(float),cudaMemcpyHostToDevice));
        this->datas_.at(i).x_size = this->srtms_.at(i).GetXSize();
        this->datas_.at(i).y_size = this->srtms_.at(i).GetYSize();
        dim3 gridSize(cuda::GetGridDim(blockSize.x, this->datas_.at(i).x_size),
                      cuda::GetGridDim(blockSize.y, this->datas_.at(i).y_size));

        CHECK_CUDA_ERR(
            LaunchDemFormatter(gridSize, blockSize, this->device_srtms_.at(i), temp_buffer, this->datas_.at(i)));
        temp_tiles.at(i).pointer = this->device_srtms_.at(i);
        temp_tiles.at(i).x = this->srtms_.at(i).GetXSize();
        temp_tiles.at(i).y = this->srtms_.at(i).GetYSize();
        temp_tiles.at(i).z = 1;
        cudaFree(temp_buffer);
    }
    CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_srtm3_tiles_, this->nr_of_tiles_ *sizeof(PointerHolder)));
    CHECK_CUDA_ERR(cudaMemcpy(this->device_srtm3_tiles_,
                              temp_tiles.data(),
                              this->nr_of_tiles_ *sizeof(PointerHolder),
                              cudaMemcpyHostToDevice));

}

void SRTM3ElevationModel::DeviceToHost(){
    CHECK_CUDA_ERR(cudaErrorNotYetImplemented);
}

void SRTM3ElevationModel::DeviceFree(){
    if(this->device_srtm3_tiles_ != nullptr){
        cudaFree(this->device_srtm3_tiles_);
        this->device_srtm3_tiles_ = nullptr;
    }

    for(unsigned int i=0; i< this->device_srtms_.size(); i++){
        if(this->device_srtms_.at(0) != nullptr){
            cudaFree(this->device_srtms_.at(0));
            this->device_srtms_.at(0) = nullptr;
        }
    }
    this->device_srtms_.clear();
}


}//namespace
}//namespace
