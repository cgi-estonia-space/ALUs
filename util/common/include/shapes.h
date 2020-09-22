#pragma once

namespace alus {

struct Rectangle{
    int x,y,width,height;
};

struct Point{
    int x,y;
};

struct PointDouble{
    double x,y;
    int index; //some points are indexed to trace their origins. Delaunay for example. Also Delauney uses indexes to compare the points.
};

}//namespace
