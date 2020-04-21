#pragma once
#include "UTC.hpp"

namespace slap {

struct Rectangle{
    int x,y,width,height;
};

struct PosVector {
    double x;
    double y;
    double z;
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
