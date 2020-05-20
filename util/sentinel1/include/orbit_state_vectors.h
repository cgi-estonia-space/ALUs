#pragma once

#include <algorithm>
#include <vector>

#include "PosVector.hpp"
#include "shapes.h"

namespace slap {

class OrbitStateVectors{
private:

    std::vector<snapEngine::PosVector> sensorPosition; // sensor position for all range lines
    std::vector<snapEngine::PosVector> sensorVelocity; // sensor velocity for all range lines
    double dt = 0.0;
    const int nv = 8;

    void getMockData();
    std::vector<OrbitStateVector> removeRedundantVectors(std::vector<OrbitStateVector> orbitStateVectors);
public:
    std::vector<OrbitStateVector> orbitStateVectors;
    snapEngine::PosVector getVelocity(double time);

    int testVectors();

    OrbitStateVectors();
    OrbitStateVectors(std::vector<OrbitStateVector> orbitStateVectors);
    ~OrbitStateVectors();
};

}//namespace
