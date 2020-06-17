#include "backgeocoding_controller.h"

namespace alus {

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

    this->backgeocoding_->ComputeTile(this->slave_rect, this->slave_tile_i_, this->slave_tile_q_);

    std::cout << "compute image ended" << '\n';
}

void BackgeocodingController::ReadPlacehoderData(){
    int i, size;

    std::ifstream rect_stream("../test/goods/backgeocoding/rectangle.txt");
    if(!rect_stream.is_open()){
        throw std::ios::failure("Error opening rectangle.txt");
    }
    rect_stream >> slave_rect.x >> slave_rect.y >> slave_rect.width >> slave_rect.height;
    rect_stream.close();

    std::ifstream slave_i_stream("../test/goods/backgeocoding/slaveTileI.txt");
    std::ifstream slave_q_stream("../test/goods/backgeocoding/slaveTileQ.txt");
    if(!slave_i_stream.is_open()){
        throw std::ios::failure("Error opening slaveTileI.txt");
    }
    if(!slave_q_stream.is_open()){
        throw std::ios::failure("Error opening slaveTileQ.txt");
    }

    size = slave_rect.width * slave_rect.height;

    this->slave_tile_i_ = new double[size];
    this->slave_tile_q_ = new double[size];

    for(i=0; i< size; i++){
        slave_i_stream >> slave_tile_i_[i];
        slave_q_stream >> slave_tile_q_[i];
    }

    slave_i_stream.close();
    slave_q_stream.close();


}

}//namespace
