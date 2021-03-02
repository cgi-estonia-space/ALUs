/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "allocators.h"
#include "calibration_vector.h"
#include "comparators.h"
#include "metadata_element.h"
#include "orbit_state_vectors.h"
#include "product_data_utc.h"
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
class Sentinel1Utils : public cuda::CudaFriendlyObject {
private:
    int num_of_sub_swath_;

    bool is_doppler_centroid_available_ = false;
    bool is_range_depend_doppler_rate_available_ = false;
    bool is_orbit_available_ = false;
    std::unique_ptr<s1tbx::OrbitStateVectors> orbit_;

    std::vector<DCPolynomial> GetDCEstimateList(std::string subswath_name);
    std::vector<DCPolynomial> ComputeDCForBurstCenters(std::vector<DCPolynomial> dc_estimate_list, int subswath_index);
    std::vector<AzimuthFmRate> GetAzimuthFmRateList(std::string subswath_name);
    DCPolynomial ComputeDC(double center_time, std::vector<DCPolynomial> dc_estimate_list);
    void WritePlaceolderInfo(int placeholder_type);
    void GetProductOrbit();
    double GetVelocity(double time);
    double GetLatitudeValue(Sentinel1Index index, SubSwathInfo* subswath);
    double GetLongitudeValue(Sentinel1Index index, SubSwathInfo* subswath);

    // files for placeholder data
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

    DeviceSentinel1Utils* device_sentinel_1_utils_{nullptr};

    double* ComputeDerampDemodPhase(int subswath_index, int s_burst_index, Rectangle rectangle);
    Sentinel1Index ComputeIndex(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);

    void ComputeReferenceTime();
    void ComputeDopplerCentroid();
    void ComputeRangeDependentDopplerRate();
    void ComputeDopplerRate();
    double GetSlantRangeTime(int x, int subswath_index);
    double GetLatitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);
    double GetLongitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);

    void SetPlaceHolderFiles(std::string_view orbit_state_vectors_file, std::string_view dc_estimate_list_file,
                             std::string_view azimuth_list_file, std::string_view burst_line_time_file,
                             std::string_view geo_location_file);
    void ReadPlaceHolderFiles();

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    alus::s1tbx::OrbitStateVectors* GetOrbitStateVectors() {
        if (is_orbit_available_) {
            return orbit_.get();
        }
        return nullptr;
    }

    Sentinel1Utils();
    Sentinel1Utils(int placeholderType);
    ~Sentinel1Utils();

    static std::shared_ptr<snapengine::Utc> GetTime(std::shared_ptr<snapengine::MetadataElement> element,
                                                    std::string_view tag);

    /**
     * Port of SNAP's Sentinel1Utils.getCalibrationVector(). Name was changed a bit in order to better represent return
     * type.
     *
     * @param calibration_vector_list_element MetadataElement containing CalibrationVector metadata elements.
     * @param output_sigma_band Whether to read sigma nought values.
     * @param output_beta_band Whether to read beta nought values.
     * @param output_gamma_band Whether to read gamma values.
     * @param output_dn_band Whether to read dn values.
     * @return List of CalibrationVector struct elements contained in the given metadata element.
     */
    static std::vector<CalibrationVector> GetCalibrationVectors(
        const std::shared_ptr<snapengine::MetadataElement>& calibration_vector_list_element, bool output_sigma_band,
        bool output_beta_band, bool output_gamma_band, bool output_dn_band);
};
}  // namespace s1tbx
}  // namespace alus
