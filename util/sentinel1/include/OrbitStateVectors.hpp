#pragma once
#include <vector>
#include "Shapes.hpp"
#include <algorithm>

namespace slap {

class OrbitStateVectors{
private:

    std::vector<PosVector> sensorPosition; // sensor position for all range lines
    std::vector<PosVector> sensorVelocity; // sensor velocity for all range lines
    double dt = 0.0;
    const int nv = 8;

    void getMockData();
    std::vector<OrbitStateVector> removeRedundantVectors(std::vector<OrbitStateVector> orbitStateVectors);
public:
    std::vector<OrbitStateVector> orbitStateVectors;
    std::vector<OrbitStateVector> orbitStateVectors2;
    PosVector getVelocity(double time);
    int testVectors();

    OrbitStateVectors();
    OrbitStateVectors(std::vector<OrbitStateVector> orbitStateVectors);
    ~OrbitStateVectors();
};

}//namespace
