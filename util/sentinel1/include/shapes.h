#pragma once
#include "utc.h"

namespace slap {

struct Rectangle{
    int x,y,width,height;
};

struct OrbitStateVector {
    UTC time;
    double time_mjd;
    double x_pos;
    double y_pos;
    double z_pos;
    double x_vel;
    double y_vel;
    double z_vel;
};

}//namespace
