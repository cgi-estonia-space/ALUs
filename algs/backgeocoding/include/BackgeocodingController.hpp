#pragma once
#include <iostream>
#include "Backgeocoding.cuh"

namespace slap {

class BackgeocodingController{
private:
    Backgeocoding *backgeocoding = nullptr;
    double *slaveTileI, *slaveTileQ;
    Rectangle slaveRect;

public:

    void readPlacehoderData();
    void computeImage();
    BackgeocodingController();
    ~BackgeocodingController();
};

}//namespace
