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
#include "i_meta_data_reader.h"
#include "metadata_attribute.h"
#include "orbit_state_vectors.h"
#include "product_data_utc.h"
#include "shapes.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
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
public:
    //    todo: why public?
    std::vector<std::unique_ptr<SubSwathInfo>> subswath_;

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
    explicit Sentinel1Utils(const std::shared_ptr<snapengine::Product>& product);
    Sentinel1Utils(const Sentinel1Utils&) = delete;  // class does not support copying(and moving)
    Sentinel1Utils& operator=(const Sentinel1Utils&) = delete;
    ~Sentinel1Utils();

    double* ComputeDerampDemodPhase(int subswath_index, int s_burst_index, Rectangle rectangle);
    Sentinel1Index ComputeIndex(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);

    void ComputeReferenceTime();
    void ComputeDopplerCentroid();
    void ComputeRangeDependentDopplerRate();
    void ComputeDopplerRate();
    double GetSlantRangeTime(int x, int subswath_index);

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    double GetLatitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);
    double GetLongitude(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);
    double GetSlantRangeTime(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);
    double GetIncidenceAngle(double azimuth_time, double slant_range_time, SubSwathInfo* subswath);

    alus::s1tbx::OrbitStateVectors* GetOrbitStateVectors() {
        if (is_orbit_available_) {
            return orbit_.get();
        }
        return nullptr;
    }

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

    /**
     * Get source product subSwath names.
     *
     * @return The subSwath name array.
     */
    const std::vector<std::string>& GetSubSwathNames() const;

    /**
     * Get source product polarizations.
     *
     * @return The polarization array.
     */
    const std::vector<std::string>& GetPolarizations() const;
    const std::vector<std::unique_ptr<SubSwathInfo>>& GetSubSwath() const;
    int GetNumOfSubSwath() const;

    std::vector<float> GetCalibrationVector(int sub_swath_index, std::string_view polarization, int vector_index,
                                            std::string_view vector_name);

    std::vector<int> GetCalibrationPixel(int sub_swath_index, std::string_view polarization, int vector_index);

    static int AddToArray(std::vector<int>& array, int index, std::string_view csv_string, std::string_view delim);

    static int AddToArray(std::vector<float>& array, int index, std::string_view csv_string, std::string_view delim);

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

    std::unique_ptr<s1tbx::OrbitStateVectors> orbit_;
    std::shared_ptr<snapengine::IMetaDataReader> metadata_reader_;

    std::vector<DCPolynomial> GetDCEstimateList(std::string subswath_name);
    std::vector<DCPolynomial> ComputeDCForBurstCenters(std::vector<DCPolynomial> dc_estimate_list, int subswath_index);
    std::vector<AzimuthFmRate> GetAzimuthFmRateList(std::string subswath_name);
    DCPolynomial ComputeDC(double center_time, std::vector<DCPolynomial> dc_estimate_list);
    void GetProductOrbit();
    double GetVelocity(double time);

    double GetLatitudeValue(Sentinel1Index index, SubSwathInfo* subswath);
    double GetLongitudeValue(Sentinel1Index index, SubSwathInfo* subswath);
    double GetSlantRangeTimeValue(Sentinel1Index index, SubSwathInfo* subswath);
    double GetIncidenceAngleValue(Sentinel1Index index, SubSwathInfo* subswath);
    void FillSubswathMetaData(SubSwathInfo *subswath);
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
    std::shared_ptr<snapengine::MetadataElement> GetSubSwathMetadata(std::string_view sub_swath_name) const;

    /**
     * Get sub-swath parameters and save them in SubSwathInfo object.
     *
     * @param subSwathMetadata The root metadata element of a given sub-swath.
     * @param subSwath         The SubSwathInfo object.
     */
    static void GetSubSwathParameters(const std::shared_ptr<snapengine::MetadataElement>& sub_swath_metadata,
                                      SubSwathInfo& sub_swath);

    [[nodiscard]] std::vector<int> GetIntVector(const std::shared_ptr<snapengine::MetadataAttribute>& attribute,
                                                std::string_view delimiter) const;
    [[nodiscard]] std::vector<double> GetDoubleVector(const std::shared_ptr<snapengine::MetadataAttribute>& attribute,
                                                      std::string_view delimiter) const;

    std::shared_ptr<snapengine::MetadataElement> GetCalibrationVectorList(int sub_swath_index,
                                                                          std::string_view polarization);

    std::shared_ptr<snapengine::Band> GetSourceBand(std::string_view sub_swath_name, std::string_view polarization) const;
};
}  // namespace s1tbx
}  // namespace alus
