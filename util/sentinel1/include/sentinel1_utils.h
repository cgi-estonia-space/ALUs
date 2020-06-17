#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <fstream>

#include "allocators.h"
#include "comparators.h"
#include "orbit_state_vectors.h"
#include "shapes.h"
#include "subswath_info.h"

namespace alus {

struct AzimuthFmRate {
    double time;
    double t0;
    double c0;
    double c1;
    double c2;
};

struct DCPolynomial {
    double time;
    double t0;
    std::vector<double> data_dc_polynomial;
};

struct Sentinel1Index {
    int i0;
    int i1;
    int j0;
    int j1;
    double mu_x;
    double mu_y;
};

class Sentinel1Utils{
private:
    int num_of_sub_swath_;

    int is_doppler_centroid_available_ = 0;
    int is_range_depend_doppler_rate_available_ = 0;
    int is_orbit_available = 0;
    alus::s1tbx::OrbitStateVectors *orbit{nullptr};


    std::vector<DCPolynomial> GetDCEstimateList(std::string subswath_name);
    std::vector<DCPolynomial> ComputeDCForBurstCenters(std::vector<DCPolynomial> dc_estimate_list,int subswath_index);
    std::vector<AzimuthFmRate> GetAzimuthFmRateList(std::string subswath_name);
    DCPolynomial ComputeDC(double center_time, std::vector<DCPolynomial> dc_estimate_list);
    void WritePlaceolderInfo(int placeholder_type);
    void GetProductOrbit();
    double GetVelocity(double time);
    double GetLatitudeValue(Sentinel1Index index, SubSwathInfo *subswath);
    double GetLongitudeValue(Sentinel1Index index, SubSwathInfo *subswath);

    //files for placeholder data
    std::string orbit_state_vectors_file_ = "";
    std::string dc_estimate_list_file_ = "";
    std::string azimuth_list_file_ = "";
    std::string burst_line_time_file_ = "";
    std::string geo_location_file_ = "";

public:
    std::vector<SubSwathInfo> subswath_;
    double range_spacing;

    double *ComputeDerampDemodPhase(int subswath_index,int s_burst_index,Rectangle rectangle);
    Sentinel1Index ComputeIndex(double azimuth_time,double slant_range_time, SubSwathInfo *subswath);

    void ComputeReferenceTime();
    void ComputeDopplerCentroid();
    void ComputeRangeDependentDopplerRate();
    void ComputeDopplerRate();
    double GetSlantRangeTime(int x, int subswath_index);
    double GetLatitude(double azimuth_time, double slant_range_time, SubSwathInfo *subswath);
    double GetLongitude(double azimuth_time, double slant_range_time, SubSwathInfo *subswath);

    void SetPlaceHolderFiles(
        std::string orbit_state_vectors_file,
        std::string dc_estimate_list_file,
        std::string azimuth_list_file,
        std::string burst_line_time_file,
        std::string geo_location_file);
    void ReadPlaceHolderFiles();

    Sentinel1Utils();
    Sentinel1Utils(int placeholderType);
    ~Sentinel1Utils();
};

}//namespace
