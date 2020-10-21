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

#include "sentinel1_utils.cuh"

namespace alus {
namespace s1tbx {

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

/**
 * This class refers to Sentinel1Utils class from s1tbx module.
 */
class Sentinel1Utils: public cuda::CudaFriendlyObject{
private:
    int num_of_sub_swath_;

    bool is_doppler_centroid_available_ = false;
    bool is_range_depend_doppler_rate_available_ = false;
    bool is_orbit_available_ = false;
    std::unique_ptr<s1tbx::OrbitStateVectors> orbit;


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

    double first_line_utc_{0.0};
    double last_line_utc_{0.0};
    double line_time_interval_{0.0};
    double near_edge_slant_range_{0.0};
    double wavelength_{0.0};
    double range_spacing_{0.0};
    double azimuth_spacing_{0.0};

    int source_image_width_{0};
    int source_image_height_{0};
    int near_range_on_left_{1};
    int srgr_flag_{0};

    DeviceSentinel1Utils *device_sentinel_1_utils_{nullptr};

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

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    alus::s1tbx::OrbitStateVectors * GetOrbitStateVectors(){
        if(this->is_orbit_available_){
            return this->orbit.get();
        }
        return nullptr;
    }

    Sentinel1Utils();
    Sentinel1Utils(int placeholderType);
    ~Sentinel1Utils();
};

}//namespace
}//namespace
