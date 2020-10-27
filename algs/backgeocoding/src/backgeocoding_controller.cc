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
#include "backgeocoding_controller.h"

namespace alus {
namespace backgeocoding{

BackgeocodingController::BackgeocodingController(){
    std::cout << "Controller started." << '\n';
}

BackgeocodingController::~BackgeocodingController(){
    if(this->backgeocoding_ != nullptr){
        delete backgeocoding_;
    }
}

void BackgeocodingController::ComputeImage(){
    std::cout << "compute image started" << '\n';
    this->backgeocoding_ = new Backgeocoding;
    this->backgeocoding_->FeedPlaceHolders();
    this->backgeocoding_->PrepareToCompute();

    //TODO:placeholder
    int slave_burst_offset = 0;
    std::vector<double> extended_amount;
    extended_amount.push_back(-0.01773467106249882);
    extended_amount.push_back(0.0);
    extended_amount.push_back(-3.770974349203243);
    extended_amount.push_back(3.8862058607542167);
    Rectangle target_area = {4000, 17000, 100, 100,};
    Rectangle target_tile = {4000, 17000, 0, 100};

    this->backgeocoding_->ComputeTile(this, 11, 11 + slave_burst_offset, target_area, target_tile, extended_amount); //TODO:placeholders

    std::cout << "compute image ended" << '\n';
}

//TODO: obviously a placeholder, but to test out a pattern.
void BackgeocodingController::ReadTile(Rectangle area, double *tile_i, double *tile_q) {
    std::ifstream slave_i_stream("../test/goods/backgeocoding/slaveTileI.txt");
    std::ifstream slave_q_stream("../test/goods/backgeocoding/slaveTileQ.txt");
    if(!slave_i_stream.is_open()){
        throw std::ios::failure("Error opening slaveTileI.txt");
    }
    if(!slave_q_stream.is_open()){
        throw std::ios::failure("Error opening slaveTileQ.txt");
    }

    size_t size = area.width * area.height;


    for(size_t i=0; i< size; i++){
        slave_i_stream >> tile_i[i];
        slave_q_stream >> tile_q[i];
    }

    slave_i_stream.close();
    slave_q_stream.close();
}

}//namespace
}//namespace
