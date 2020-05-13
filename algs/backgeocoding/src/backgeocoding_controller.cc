#include "backgeocoding_controller.h"

namespace alus {

BackgeocodingController::BackgeocodingController(){
    std::cout << "Controller started." << '\n';
}

BackgeocodingController::~BackgeocodingController(){
    if(this->backgeocoding != nullptr){
        delete backgeocoding;
    }
}

void BackgeocodingController::computeImage(){
    std::cout << "compute image started" << '\n';
    this->backgeocoding = new Backgeocoding;
    this->backgeocoding->feedPlaceHolders();
    this->backgeocoding->prepareToCompute();

    this->backgeocoding->computeTile(this->slaveRect, this->slaveTileI, this->slaveTileQ);

    std::cout << "compute image ended" << '\n';
}

void BackgeocodingController::readPlacehoderData(){
    int i, size;

    std::ifstream rectStream("../test/goods/backgeocoding/rectangle.txt");
    if(!rectStream.is_open()){
        throw std::ios::failure("Error opening rectangle.txt");
    }
    rectStream >> slaveRect.x >> slaveRect.y >>slaveRect.width >> slaveRect.height;
    rectStream.close();

    std::ifstream slaveIStream("../test/goods/backgeocoding/slaveTileI.txt");
    std::ifstream slaveQStream("../test/goods/backgeocoding/slaveTileQ.txt");
    if(!slaveIStream.is_open()){
        throw std::ios::failure("Error opening slaveTileI.txt");
    }
    if(!slaveQStream.is_open()){
        throw std::ios::failure("Error opening slaveTileQ.txt");
    }

    size = slaveRect.width * slaveRect.height;

    this->slaveTileI = new double[size];
    this->slaveTileQ = new double[size];

    for(i=0; i< size; i++){
        slaveIStream >> slaveTileI[i];
        slaveQStream >> slaveTileQ[i];
    }

    slaveIStream.close();
    slaveQStream.close();


}

}//namespace
