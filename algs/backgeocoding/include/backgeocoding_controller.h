#pragma once
#include <iostream>
#include "backgeocoding.h"

namespace alus {

class BackgeocodingController{
private:
    Backgeocoding *backgeocoding_{nullptr};
    double *slave_tile_i_, *slave_tile_q_;
    Rectangle slave_rect;

public:

    void ReadPlacehoderData();
    void ComputeImage();
    BackgeocodingController();
    ~BackgeocodingController();
};

}//namespace
