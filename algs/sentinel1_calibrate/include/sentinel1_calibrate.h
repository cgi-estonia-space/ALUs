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

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "calibration_info.h"
#include "calibration_info_computation.h"
#include "calibration_type.h"
#include "calibration_vector_computation.h"
#include "dataset.h"
#include "metadata_element.h"
#include "sentinel1_calibrate_kernel.h"
#include "shapes.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/datamodel/raster_data_node.h"
#include "sentinel1_calibrate_safe_helper.h"

namespace alus::sentinel1calibrate {

struct SelectedCalibrationBands {
    bool get_sigma_lut{false};
    bool get_beta_lut{false};
    bool get_gamma_lut{false};
    bool get_dn_lut{false};
};

struct CInt16 {
    int16_t i;
    int16_t q;
};

class Sentinel1Calibrator {
public:
    Sentinel1Calibrator(
        std::shared_ptr<snapengine::Product> source_product, const std::string& src_path,
        std::vector<std::string> selected_sub_swaths, std::set<std::string, std::less<>> selected_polarisations,
        SelectedCalibrationBands selected_calibration_bands,
        std::string_view output_path, bool output_image_in_complex = false, int tile_width = 2000, int tile_height = 2000);

    Sentinel1Calibrator(const Sentinel1Calibrator&) = delete;
    Sentinel1Calibrator(Sentinel1Calibrator&&) = delete;
    Sentinel1Calibrator& operator=(const Sentinel1Calibrator&) = delete;
    Sentinel1Calibrator& operator=(Sentinel1Calibrator&&) = delete;

    ~Sentinel1Calibrator();

    std::shared_ptr<snapengine::Product> GetTargetProduct() { return target_product_; }
    std::string GetTargetPath(const std::string& swath) { return target_paths_.at(swath); }
    void Execute();


private:
    std::shared_ptr<snapengine::Product> source_product_;
    std::map<std::string, std::vector<std::string>, std::less<>> target_band_name_to_source_band_name_;
    std::map<std::string, CalibrationInfo, std::less<>> target_band_to_calibration_info_;
    std::vector<std::string> selected_sub_swaths_;
    std::set<std::string, std::less<>> selected_polarisations_;
    SelectedCalibrationBands selected_calibration_bands_;
    std::shared_ptr<snapengine::Product> target_product_;
    bool is_complex_;
    std::shared_ptr<snapengine::MetadataElement> abstract_metadata_root_;
    CAL_TYPE data_type_{CAL_TYPE::NONE};
    int subset_offset_x_;
    int subset_offset_y_;
    std::vector<CalibrationInfo> calibration_info_list_;
    bool is_multi_swath_;
    std::vector<std::string> source_band_names_;
    bool output_image_in_complex_;
    int tile_width_;
    int tile_height_;
    std::map<std::string, std::shared_ptr<Dataset<float>>, std::less<>> target_datasets_;
    std::map<std::string, std::string, std::less<>> target_paths_;
    std::string output_path_;
    std::vector<void*> cuda_arrays_to_clean_;
    Sentinel1CalibrateSafeHelper safe_helper_;

    // Device variables
    std::map<std::string, CalibrationInfoComputation, std::less<>> target_band_to_d_calibration_info_;

    void ComputeTile(std::shared_ptr<snapengine::Band> target_band, Rectangle target_rectangle, int band_index);
    void CreateTargetProduct();
    void GetSampleType();
    void GetSubsetOffset();
    void GetVectors();  // Very misleading name as it actually fetches calibration info
    void CreateTargetBandToCalibrationInfoMap();
    void Validate();
    void Initialise();
    void UpdateTargetProductMetadata() const;
    void SetTargetImages();
    void AddSelectedBands(std::vector<std::string>& source_band_names);
    void OutputInComplex(std::vector<std::string>& source_band_names);
    void OutputInIntensity(const std::vector<std::string>& source_band_names);
    [[nodiscard]] std::vector<std::string> CreateTargetBandNames(std::string_view source_band_name) const;
    [[nodiscard]] std::vector<Rectangle> CalculateTiles(std::shared_ptr<snapengine::Band> target_band) const;
    void CopyAllCalibrationInfoToDevice();

    static CAL_TYPE GetCalibrationType(std::string_view band_name);

    static constexpr std::string_view PRODUCT_SUFFIX{"_Cal"};

    void CreateDatasetsFromProduct(std::shared_ptr<snapengine::Product> product,
                                   std::string_view output_path);  // TODO: should it be moved to some other space?

    int GetCalibrationCount() const;
};

/**
 * This is a port of SNAP's Sentinel1Calibrator.getCalibrationVectors(). The name was changed in order to better
 * represent the functions return type;
 *
 * @param original_product_metadata "Original_Product_Metadata" MetadataElement
 * @param selected_polarisations All polarisations for which calibration info should be returned.
 * @param selected_calibration_bands All calibration bands for which calibration info should be returned.
 * @return List of CalibrationInfo structs contained in the given original_product_metadata element.
 */
std::vector<CalibrationInfo> GetCalibrationInfoList(
    const std::shared_ptr<snapengine::MetadataElement>& original_product_metadata,
    std::set<std::string, std::less<>> selected_polarisations, SelectedCalibrationBands selected_calibration_bands);

int GetNumOfLines(const std::shared_ptr<snapengine::MetadataElement>& original_product_root,
                  std::string_view polarisation, std::string_view swath);

/**
 * Custom wrapper for MetadataElement::GetElement() function. It checks whether the returned element exists and throws
 * an std::runtime_error if not.
 *
 * @param parent_element Element, whose child should be found.
 * @param element_name Name of the child to be found.
 * @throws std::runtime_element Throws this error if the element is not found.
 * @return
 */
std::shared_ptr<snapengine::MetadataElement> GetElement(
    const std::shared_ptr<snapengine::MetadataElement>& parent_element, std::string_view element_name);
}  // namespace alus::sentinel1calibrate