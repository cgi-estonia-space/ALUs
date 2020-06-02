#include "orbit_state_vectors.h"

namespace alus::s1tbx {

OrbitStateVectors::OrbitStateVectors(){
    getMockData();
}


OrbitStateVectors::OrbitStateVectors(std::vector<snapengine::OrbitStateVector> const& orbitStateVectors) {

    this->orbitStateVectors = RemoveRedundantVectors(orbitStateVectors);

    this->dt = (this->orbitStateVectors[orbitStateVectors.size() - 1].timeMjd_ -
            this->orbitStateVectors[0].timeMjd_) / static_cast<double>(this->orbitStateVectors.size() - 1);
}

//TODO: this is mocked.
void OrbitStateVectors::getMockData(){

}

std::vector<snapengine::OrbitStateVector> OrbitStateVectors::RemoveRedundantVectors(std::vector<snapengine::OrbitStateVector>
    orbitStateVectors){

    std::vector<snapengine::OrbitStateVector> vectorList;
    double currentTime = 0.0;
    currentTime = orbitStateVectors.at(0).timeMjd_;
    vectorList.push_back(orbitStateVectors.at(0));
    for (unsigned int i = 1; i < orbitStateVectors.size(); i++) {
        if (orbitStateVectors.at(i).timeMjd_ > currentTime) {
            currentTime = orbitStateVectors.at(i).timeMjd_;
            vectorList.push_back(orbitStateVectors.at(i));
        }
    }

    return vectorList;
}

snapengine::PosVector OrbitStateVectors::GetVelocity(double time) {

    int i0, iN;
    double weight, time2;
    int vectorsSize = orbitStateVectors.size();
    //lagrangeInterpolatingPolynomial
    snapengine::PosVector velocity{};
    snapengine::OrbitStateVector orbI{};

    if (vectorsSize <= nv) {
        i0 = 0;
        iN = vectorsSize - 1;
    } else {
        i0 = std::max((int) ((time - orbitStateVectors[0].timeMjd_) / dt) - nv / 2 + 1, 0);
        iN = std::min(i0 + nv - 1, vectorsSize - 1);
        i0 = (iN < vectorsSize - 1 ? i0 : iN - nv + 1);
    }

    for (int i = i0; i <= iN; ++i) {
        orbI = orbitStateVectors[i];

        weight = 1;
        for (int j = i0; j <= iN; ++j) {
            if (j != i) {
                time2 = orbitStateVectors[j].timeMjd_;
                weight *= (time - time2) / (orbI.timeMjd_ - time2);
            }
        }
        velocity.x += weight * orbI.xVel_;
        velocity.y += weight * orbI.yVel_;
        velocity.z += weight * orbI.zVel_;
    }
    return velocity;
}

int OrbitStateVectors::testVectors(){

    return 0;
}

} //namespace
