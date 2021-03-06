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
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "allocators.h"
#include "comparators.h"
#include "s1tbx-commons/calibration_vector.h"
#include "s1tbx-commons/orbit_state_vectors.h"
#include "shapes.h"
#include "snap-core/core/datamodel/i_meta_data_reader.h"
#include "snap-core/core/datamodel/metadata_attribute.h"
#include "snap-core/core/datamodel/metadata_element.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-core/core/datamodel/product_data_utc.h"
#include "subswath_info.h"

#include "sentinel1_utils.cuh"

namespace alus::s1tbx {

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
public:
    //    todo: why public?
    std::vector<std::shared_ptr<SubSwathInfo>> subswath_;

    double first_line_utc_{0.0};
    double last_line_utc_{0.0};
    double line_time_interval_{0.0};
    double near_edge_slant_range_{0.0};
    double wavelength_{0.0};
    double range_spacing_{0.0};
    double azimuth_spacing_{0.0};

    int source_image_width_{0};
    int source_image_height_{0};
    bool near_range_on_left_{true};
    bool srgr_flag_{false};

    DeviceSentinel1Utils* device_sentinel_1_utils_{nullptr};

    explicit Sentinel1Utils(std::string_view metadata_file_name);
    explicit Sentinel1Utils(std::shared_ptr<snapengine::Product> product);
    Sentinel1Utils(const Sentinel1Utils&) = delete;  // class does not support copying(and moving)
    Sentinel1Utils& operator=(const Sentinel1Utils&) = delete;
    virtual ~Sentinel1Utils();

    double* ComputeDerampDemodPhase(int subswath_index, int s_burst_index, Rectangle rectangle);
    [[nodiscard]] Sentinel1Index ComputeIndex(double azimuth_time, double slant_range_time, int sub_swath_index) const;

    void ComputeReferenceTime();
    void ComputeDopplerCentroid();
    void ComputeRangeDependentDopplerRate();
    void ComputeDopplerRate();
    [[nodiscard]] double GetSlantRangeTime(int x, int subswath_index) const;

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    [[nodiscard]] double GetLatitude(double azimuth_time, double slant_range_time) const;
    [[nodiscard]] double GetLongitude(double azimuth_time, double slant_range_time) const;
    [[nodiscard]] double GetSlantRangeTime(double azimuth_time, double slant_range_time) const;
    [[nodiscard]] double GetIncidenceAngle(double azimuth_time, double slant_range_time) const;

    alus::s1tbx::OrbitStateVectors* GetOrbitStateVectors() {
        if (is_orbit_available_) {
            return orbit_.get();
        }
        return nullptr;
    }

    static std::shared_ptr<snapengine::Utc> GetTime(const std::shared_ptr<snapengine::MetadataElement>& element,
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

    static void UpdateBandNames(std::shared_ptr<snapengine::MetadataElement>& abs_root,
                                const std::set<std::string, std::less<>>& selected_pol_list,
                                const std::vector<std::string>& band_names);

    /**
     * Get source product subSwath names.
     *
     * @return The subSwath name array.
     */
    [[nodiscard]] const std::vector<std::string>& GetSubSwathNames() const;

    /**
     * Get source product polarizations.
     *
     * @return The polarization array.
     */
    [[nodiscard]] const std::vector<std::string>& GetPolarizations() const;
    [[nodiscard]] const std::vector<std::shared_ptr<SubSwathInfo>>& GetSubSwath() const;
    [[nodiscard]] int GetNumOfSubSwath() const;

    /**
     * Get sub-swath index for given slant range time.
     *
     * @param slant_range_time The given slant range time.
     * @return The sub-swath index (start from 1).
     */
    [[nodiscard]] int GetSubswathIndex(double slant_range_time) const;

    std::vector<float> GetCalibrationVector(int sub_swath_index, std::string_view polarization, int vector_index,
                                            std::string_view vector_name);

    std::vector<int> GetCalibrationPixel(int sub_swath_index, std::string_view polarization, int vector_index);

    static int AddToArray(std::vector<int>& array, int index, std::string_view csv_string, std::string_view delim);

    static int AddToArray(std::vector<float>& array, int index, std::string_view csv_string, std::string_view delim);

    static Sentinel1Index ComputeIndex(double azimuth_time, double slant_range_time, const SubSwathInfo* subswath);

    static double GetLatitude(double azimuth_time, double slant_range_time, const SubSwathInfo* subswath);

    static double GetLongitude(double azimuth_time, double slant_range_time, const SubSwathInfo* subswath);

    static double GetSlantRangeTime(double azimuth_time, double slant_range_time, const SubSwathInfo* subswath);

    static double GetIncidenceAngle(double azimuth_time, double slant_range_time, const SubSwathInfo* subswath);

private:
    std::shared_ptr<snapengine::Product> source_product_;
    std::shared_ptr<snapengine::MetadataElement> abs_root_;
    std::shared_ptr<snapengine::MetadataElement> orig_prod_root_;
    int num_of_sub_swath_;
    std::string acquisition_mode_;
    std::vector<std::string> polarizations_;
    std::vector<std::string> sub_swath_names_;
    bool is_doppler_centroid_available_ = false;
    bool is_range_depend_doppler_rate_available_ = false;
    bool is_orbit_available_ = false;
    bool legacy_init_ = false;

    std::unique_ptr<s1tbx::OrbitStateVectors> orbit_;
    std::shared_ptr<snapengine::IMetaDataReader> metadata_reader_;

    std::vector<DCPolynomial> GetDCEstimateList(std::string_view subswath_name);
    std::vector<DCPolynomial> ComputeDCForBurstCenters(std::vector<DCPolynomial> dc_estimate_list, int subswath_index);
    std::vector<AzimuthFmRate> GetAzimuthFmRateList(std::string_view subswath_name);
    void GetProductOrbit();
    double GetVelocity(double time);

    [[nodiscard]] double GetLatitudeValue(const Sentinel1Index& index, int sub_swath_index) const;
    [[nodiscard]] double GetLongitudeValue(const Sentinel1Index& index, int sub_swath_index) const;
    [[nodiscard]] double GetSlantRangeTimeValue(const Sentinel1Index& index, int sub_swath_index) const;
    [[nodiscard]] double GetIncidenceAngleValue(const Sentinel1Index& index, int sub_swath_index) const;
    void FillSubswathMetaData(const std::shared_ptr<snapengine::MetadataElement>& subswath_metadata,
                              SubSwathInfo* subswath);
    void FillUtilsMetadata();
    void GetMetadataRoot();
    void GetAbstractedMetadata();
    /**
     * Get source product polarizations.
     */
    void GetProductPolarizations();
    /**
     * Get acquisition mode from abstracted metadata.
     */
    void GetProductAcquisitionMode();

    /**
     * Get source product subSwath names.
     */
    void GetProductSubSwathNames();

    /**
     * Get parameters for all sub-swaths.
     */
    void GetSubSwathParameters();

    /**
     * Get root metadata element of given sub-swath.
     *
     * @param subSwathName Sub-swath name string.
     * @return The root metadata element.
     */
    [[nodiscard]] std::shared_ptr<snapengine::MetadataElement> GetSubSwathMetadata(
        std::string_view sub_swath_name) const;

    [[nodiscard]] static std::vector<double> GetDoubleVector(
        const std::shared_ptr<snapengine::MetadataAttribute>& attribute, std::string_view delimiter);

    [[nodiscard]] static std::vector<int> GetIntVector(const std::shared_ptr<snapengine::MetadataAttribute>& attribute,
                                                       std::string_view delimiter);

    static DCPolynomial ComputeDC(double center_time, std::vector<DCPolynomial> dc_estimate_list);

    static double GetLatitudeValue(const Sentinel1Index& index, const SubSwathInfo* subswath);

    static double GetLongitudeValue(const Sentinel1Index& index, const SubSwathInfo* subswath);

    static double GetSlantRangeTimeValue(const Sentinel1Index& index, const SubSwathInfo* subswath);

    static double GetIncidenceAngleValue(const Sentinel1Index& index, const SubSwathInfo* subswath);

    std::shared_ptr<snapengine::MetadataElement> GetCalibrationVectorList(int sub_swath_index,
                                                                          std::string_view polarization);

    [[nodiscard]] std::shared_ptr<snapengine::Band> GetSourceBand(std::string_view sub_swath_name,
                                                                  std::string_view polarization) const;
};
}  // namespace alus::s1tbx