#pragma once
#include <iostream>
#include "backgeocoding.h"

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