#include "orbit_state_vectors.h"

namespace slap {

using namespace snapEngine;

OrbitStateVectors::OrbitStateVectors(){
    getMockData();
}


OrbitStateVectors::OrbitStateVectors(std::vector<OrbitStateVector> orbitStateVectors) {

    this->orbitStateVectors = removeRedundantVectors(orbitStateVectors);

    this->dt = (this->orbitStateVectors[orbitStateVectors.size() - 1].time_mjd -
            this->orbitStateVectors[0].time_mjd) / (this->orbitStateVectors.size() - 1);
}


OrbitStateVectors::~OrbitStateVectors(){

}

//TODO: this is mocked.
void OrbitStateVectors::getMockData(){

}

std::vector<OrbitStateVector> OrbitStateVectors::removeRedundantVectors(std::vector<OrbitStateVector> orbitStateVectors){

    std::vector<OrbitStateVector> vectorList;
    double currentTime = 0.0;
    for (unsigned int i = 0; i < orbitStateVectors.size(); i++) {
        if (i == 0) {
            currentTime = orbitStateVectors[i].time_mjd;
            vectorList.push_back(orbitStateVectors[i]);
        } else if (orbitStateVectors[i].time_mjd > currentTime) {
            currentTime = orbitStateVectors[i].time_mjd;
            vectorList.push_back(orbitStateVectors[i]);
        }
    }

    return vectorList;
}

PosVector OrbitStateVectors::getVelocity(double time) {

    int i0, iN;
    double weight, time2;
    int vectorsSize = orbitStateVectors.size();
    //lagrangeInterpolatingPolynomial
    PosVector velocity{};
    OrbitStateVector orbI;

    if (vectorsSize <= nv) {
        i0 = 0;
        iN = vectorsSize - 1;
    } else {
        i0 = std::max((int) ((time - orbitStateVectors[0].time_mjd) / dt) - nv / 2 + 1, 0);
        iN = std::min(i0 + nv - 1, vectorsSize - 1);
        i0 = (iN < vectorsSize - 1 ? i0 : iN - nv + 1);
    }



    for (int i = i0; i <= iN; ++i) {
        orbI = orbitStateVectors[i];

        weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                time2 = orbitStateVectors[j].time_mjd;
                weight *= (time - time2) / (orbI.time_mjd - time2);
            }
        }
        velocity.x += weight * orbI.x_vel;
        velocity.y += weight * orbI.y_vel;
        velocity.z += weight * orbI.z_vel;
    }
    return velocity;
}

int OrbitStateVectors::testVectors(){

    return 0;
}

} //namespace
