#include "Sentinel1Utils.hpp"

namespace slap {

Sentinel1Utils::Sentinel1Utils(){
    writePlaceolderInfo();
}

Sentinel1Utils::~Sentinel1Utils(){
    if(orbit != nullptr){
        delete orbit;
    }
}

//TODO: using placeholder data
void Sentinel1Utils::writePlaceolderInfo(){
    numOfSubSwath = 1;

    SubSwathInfo temp;
    subSwath.push_back(temp);

    subSwath[0].azimuthTimeInterval = 0.002055556299999998;
    subSwath[0].numOfBursts = 19;
    subSwath[0].linesPerBurst = 1503;
    subSwath[0].samplesPerBurst = 21401;
    subSwath[0].firstValidPixel = 267;
    subSwath[0].lastValidPixel = 20431;
    subSwath[0].rangePixelSpacing = 2.329562;
    subSwath[0].slrTimeToFirstPixel = 0.002679737321566982;
    subSwath[0].subSwathName = "IW1";
    subSwath[0].firstLineTime = 5.50770938201763E8;
    subSwath[0].lastLineTime = 5.50770990939114E8;
    subSwath[0].radarFrequency = 5.40500045433435E9;
    subSwath[0].azimuthSteeringRate = 1.590368784;

}

void Sentinel1Utils::setPlaceHolderFiles(std::string orbitStateVectorsFile, std::string dcEstimateListFile, std::string azimuthListFile){
    this->orbitStateVectorsFile = orbitStateVectorsFile;
    this->dcEstimateListFile = dcEstimateListFile;
    this->azimuthListFile = azimuthListFile;
}

double *Sentinel1Utils::computeDerampDemodPhase(int subSwathIndex,int sBurstIndex,Rectangle rectangle){
    const int x0 = rectangle.x;
    const int y0 = rectangle.y;
    const int w = rectangle.width;
    const int h = rectangle.height;

    const int xMax = x0 + w;
    const int yMax = y0 + h;
    const int s = subSwathIndex - 1;
    const int firstLineInBurst = sBurstIndex* subSwath[s].linesPerBurst;

    double *result = new double[h*w*sizeof(double)];
    int yy,xx,x,y;
    double ta,kt,deramp,demod;

    for (y = y0; y < yMax; y++) {
        yy = y - y0;
        ta = (y - firstLineInBurst)* subSwath[s].azimuthTimeInterval;
        for (x = x0; x < xMax; x++) {
            xx = x - x0;
            kt = subSwath[s].dopplerRate[sBurstIndex][x];
            deramp = -EsaConstants_PI * kt * pow(ta - subSwath[s].referenceTime[sBurstIndex][x], 2);
            demod = -EsaConstants_TWO_PI * subSwath[s].dopplerCentroid[sBurstIndex][x] * ta;
            result[yy*w + xx] = deramp + demod;
        }
    }

    return result;
}

//TODO: is using placeholder info
void Sentinel1Utils::getProductOrbit(){
    std::vector<OrbitStateVector> originalVectors;
    int i,count;
    OrbitStateVector tempVector;

    std::ifstream vectorReader(orbitStateVectorsFile);
    if(!vectorReader.is_open()){
        throw std::ios::failure("Vector reader is not open.");
    }
    vectorReader >> count;
    std::cout << "writing original vectors: " << count << '\n';
    for(i=0; i<count; i++){
        vectorReader >> tempVector.time.days >>tempVector.time.seconds >> tempVector.time.microseconds;
        vectorReader >> tempVector.time_mjd;
        vectorReader >> tempVector.x_pos >> tempVector.y_pos >> tempVector.z_pos;
        vectorReader >> tempVector.x_vel >> tempVector.y_vel >> tempVector.z_vel;
        originalVectors.push_back(tempVector);
    }
    vectorReader >> count;
    this->orbit = new OrbitStateVectors(originalVectors);

    std::cout << "writing test vectors: " <<count << '\n';
    for(i=0; i<count; i++){
        vectorReader >> tempVector.time.days >>tempVector.time.seconds >> tempVector.time.microseconds;
        vectorReader >> tempVector.time_mjd;
        vectorReader >> tempVector.x_pos >> tempVector.y_pos >> tempVector.z_pos;
        vectorReader >> tempVector.x_vel >> tempVector.y_vel >> tempVector.z_vel;
        this->orbit->orbitStateVectors2.push_back(tempVector);
    }


    vectorReader.close();
    isOrbitAvailable = 1;
}

double Sentinel1Utils::getVelocity(double time){
    PosVector velocity = orbit->getVelocity(time);
    return sqrt(velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z);
}

void Sentinel1Utils::computeDopplerRate(){
    double waveLength,azTime, v, steeringRate, krot;

    if (!isOrbitAvailable) {
        getProductOrbit();
    }

    if (!isRangeDependDopplerRateAvailable) {
        computeRangeDependentDopplerRate();
    }

    waveLength = EsaConstants_lightSpeed / subSwath[0].radarFrequency;
    for (int s = 0; s < numOfSubSwath; s++) {
        azTime = (subSwath[s].firstLineTime + subSwath[s].lastLineTime)/2.0;
        subSwath[s].dopplerRate = allocate2DDoubleArray(subSwath[s].numOfBursts, subSwath[s].samplesPerBurst);
        v = getVelocity(azTime/EsaConstants_secondsInDay); // DLR: 7594.0232
        steeringRate = subSwath[s].azimuthSteeringRate * EsaConstants_DTOR;
        krot = 2*v*steeringRate/waveLength; // doppler rate by antenna steering

        for (int b = 0; b < subSwath[s].numOfBursts; b++) {
            for (int x = 0; x < subSwath[s].samplesPerBurst; x++) {
                subSwath[s].dopplerRate[b][x] = subSwath[s].rangeDependDopplerRate[b][x] * krot
                        / (subSwath[s].rangeDependDopplerRate[b][x] - krot);
            }
        }
    }
}

double Sentinel1Utils::getSlantRangeTime(int x, int subSwathIndex) {
    return subSwath[subSwathIndex - 1].slrTimeToFirstPixel +
            x * subSwath[subSwathIndex - 1].rangePixelSpacing / EsaConstants_lightSpeed;
}

//TODO: using mock data
std::vector<DCPolynomial> Sentinel1Utils::getDCEstimateList(std::string subSwathName){
    std::vector<DCPolynomial> result;
    int count,i,j, tempCount;
    double temp;
    std::cout << "Mocking for subswath: " << subSwathName << '\n';
    std::ifstream dcLister(dcEstimateListFile);
    if(!dcLister.is_open()){
        throw std::ios::failure("Azimuth list reader is not open.");
    }

    dcLister >> count;
    result.reserve(count);
    for(i=0; i<count; i++){
        DCPolynomial tempPoly;
        dcLister >> tempPoly.time >> tempPoly.t0 >> tempCount;
        tempPoly.dataDcPolynomial.reserve(tempCount);
        for(j=0; j<tempCount; j++){
            dcLister >> temp;
            tempPoly.dataDcPolynomial.push_back(temp);
        }
        result.push_back(tempPoly);
    }

    dcLister.close();

    return result;
}

DCPolynomial Sentinel1Utils::computeDC(double centerTime, std::vector<DCPolynomial> dcEstimateList) {
    DCPolynomial dcPolynomial;
    double mu;
    int i0 = 0, i1 = 0;
    if (centerTime < dcEstimateList[0].time) {
        i0 = 0;
        i1 = 1;
    } else if (centerTime > dcEstimateList[dcEstimateList.size() - 1].time) {
        i0 = dcEstimateList.size() - 2;
        i1 = dcEstimateList.size() - 1;
    } else {
        for (unsigned int i = 0; i < dcEstimateList.size() - 1; i++) {
            if (centerTime >= dcEstimateList[i].time && centerTime < dcEstimateList[i+1].time) {
                i0 = i;
                i1 = i + 1;
                break;
            }
        }
    }


    dcPolynomial.time = centerTime;
    dcPolynomial.t0 = dcEstimateList[i0].t0;
    dcPolynomial.dataDcPolynomial.reserve(dcEstimateList[i0].dataDcPolynomial.size());
    mu = (centerTime - dcEstimateList[i0].time) / (dcEstimateList[i1].time - dcEstimateList[i0].time);
    for (unsigned int j = 0; j < dcEstimateList[i0].dataDcPolynomial.size(); j++) {
        dcPolynomial.dataDcPolynomial[j] = (1 - mu)*dcEstimateList[i0].dataDcPolynomial[j] +
                mu*dcEstimateList[i1].dataDcPolynomial[j];
    }

    return dcPolynomial;
}

//TODO: Half of this function will not work due to missing data. We just got lucky atm.
std::vector<DCPolynomial> Sentinel1Utils::computeDCForBurstCenters(std::vector<DCPolynomial> dcEstimateList,int subSwathIndex){
    double centerTime;
    if ((int)dcEstimateList.size() >= subSwath[subSwathIndex - 1].numOfBursts) {
        std::cout << "used the fast lane" << '\n';
        return dcEstimateList;
    }

    std::vector<DCPolynomial> dcBurstList(subSwath[subSwathIndex - 1].numOfBursts);
    for (int b = 0; b < subSwath[subSwathIndex - 1].numOfBursts; b++) {
        if (b < (int)dcEstimateList.size()) {
            dcBurstList[b] = dcEstimateList[b];
            std::cout << "using less list" << '\n';
        } else {
            std::cout << "using more list" << '\n';
            centerTime = 0.5*(subSwath[subSwathIndex - 1].burstFirstLineTime[b] +
                    subSwath[subSwathIndex - 1].burstLastLineTime[b]);

            dcBurstList[b] = computeDC(centerTime, dcEstimateList);
        }
    }

    return dcBurstList;
}

void Sentinel1Utils::computeDopplerCentroid(){
    double slrt, dt, dcValue;
    for (int s = 0; s < numOfSubSwath; s++) {
        std::vector<DCPolynomial> dcEstimateList = getDCEstimateList(subSwath[s].subSwathName);
        std::vector<DCPolynomial> dcBurstList = computeDCForBurstCenters(dcEstimateList, s+1);
        subSwath[s].dopplerCentroid = allocate2DDoubleArray(subSwath[s].numOfBursts, subSwath[s].samplesPerBurst);
        for (int b = 0; b < subSwath[s].numOfBursts; b++) {
            for (int x = 0; x < subSwath[s].samplesPerBurst; x++) {
                slrt = getSlantRangeTime(x, s+1)*2;
                dt = slrt - dcBurstList[b].t0;

                dcValue = 0.0;
                for (unsigned int i = 0; i < dcBurstList[b].dataDcPolynomial.size(); i++) {
                    dcValue += dcBurstList[b].dataDcPolynomial[i] * pow(dt, i);
                }
                subSwath[s].dopplerCentroid[b][x] = dcValue;
            }
        }
    }

    isDopplerCentroidAvailable = 1;
}

//TODO: useing mock data
std::vector<AzimuthFmRate> Sentinel1Utils::getAzimuthFmRateList(std::string subSwathName){
    std::vector<AzimuthFmRate> result;
    int count, i;
    std::cout << "Getting azimuth FM list for subswath: " << subSwathName << '\n';
    std::ifstream azimuthListReader(azimuthListFile);
    if(!azimuthListReader.is_open()){
        throw std::ios::failure("Azimuth list reader is not open.");
    }

    azimuthListReader >> count;
    result.reserve(count);
    for(i=0; i<count; i++){
        AzimuthFmRate temp;
        azimuthListReader >> temp.time >> temp.t0 >> temp.c0 >> temp.c1 >>temp.c2;
        result.push_back(temp);
    }

    azimuthListReader.close();

    return result;
}

void Sentinel1Utils::computeRangeDependentDopplerRate(){
    double slrt,dt;

    for (int s = 0; s < numOfSubSwath; s++) {
        std::vector<AzimuthFmRate> azFmRateList = getAzimuthFmRateList(subSwath[s].subSwathName);
        subSwath[s].rangeDependDopplerRate = allocate2DDoubleArray(subSwath[s].numOfBursts,subSwath[s].samplesPerBurst);
        for (int b = 0; b < subSwath[s].numOfBursts; b++) {
            for (int x = 0; x < subSwath[s].samplesPerBurst; x++) {
                slrt = getSlantRangeTime(x, s+1)*2; // 1-way to 2-way
                dt = slrt - azFmRateList[b].t0;
                subSwath[s].rangeDependDopplerRate[b][x] =
                        azFmRateList[b].c0 + azFmRateList[b].c1*dt + azFmRateList[b].c2*dt*dt;
            }
        }
    }
    isRangeDependDopplerRateAvailable = true;
}

void Sentinel1Utils::computeReferenceTime(){
    double tmp1, tmp2;
    if (!isDopplerCentroidAvailable) {
        computeDopplerCentroid();
    }

    if (!isRangeDependDopplerRateAvailable) {
        computeRangeDependentDopplerRate();
    }

    for (int s = 0; s < numOfSubSwath; s++) {
        subSwath[s].referenceTime = allocate2DDoubleArray(subSwath[s].numOfBursts,subSwath[s].samplesPerBurst);
        tmp1 = subSwath[s].linesPerBurst * subSwath[s].azimuthTimeInterval / 2.0;

        for (int b = 0; b < subSwath[s].numOfBursts; b++) {

            tmp2 = tmp1 + subSwath[s].dopplerCentroid[b][subSwath[s].firstValidPixel] /
                    subSwath[s].rangeDependDopplerRate[b][subSwath[s].firstValidPixel];

            for (int x = 0; x < subSwath[s].samplesPerBurst; x++) {
                subSwath[s].referenceTime[b][x] = tmp2 -
                        subSwath[s].dopplerCentroid[b][x] / subSwath[s].rangeDependDopplerRate[b][x];
            }
        }
    }
}

}//namespace
