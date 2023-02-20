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
#include "coregistration_controller.h"

#include <memory>

#include "apply_orbit_file_op.h"
#include "backgeocoding_controller.h"
#include "general_constants.h"
#include "snap-core/core/util/alus_utils.h"
#include "snap-core/core/util/system_utils.h"
#include "target_dataset.h"
#include "topsar_split.h"

namespace {
constexpr size_t FULL_SUBSWATH_BURST_INDEX_START{alus::topsarsplit::TopsarSplit::BURST_INDEX_OFFSET};
constexpr size_t FULL_SUBSWATH_BURST_INDEX_END{9999};
}  // namespace

namespace alus::coregistration {

Coregistration::Coregistration(const std::string& aux_data_path) {
    alus::snapengine::SystemUtils::SetAuxDataPath(aux_data_path + "/");
}

void Coregistration::Initialize(std::string_view reference_file, std::string_view secondary_file,
                                std::string_view output_file, std::string_view subswath_name,
                                std::string_view polarisation, size_t first_burst_index, size_t last_burst_index) {
    Parameters params;
    params.main_scene_file = reference_file;
    params.secondary_scene_file = secondary_file;
    params.subswath = subswath_name;
    params.output_file = output_file;
    params.polarisation = polarisation;
    params.main_scene_first_burst_index = params.secondary_scene_first_burst_index = first_burst_index;
    params.main_scene_last_burst_index = params.secondary_scene_last_burst_index = last_burst_index;

    Initialize(params);
}

void Coregistration::Initialize(std::string_view reference_file, std::string_view secondary_file,
                                std::string_view output_file, std::string_view subswath_name,
                                std::string_view polarisation) {
    Parameters params;
    params.main_scene_file = reference_file;
    params.secondary_scene_file = secondary_file;
    params.subswath = subswath_name;
    params.output_file = output_file;
    params.polarisation = polarisation;
    params.main_scene_first_burst_index = params.secondary_scene_first_burst_index = FULL_SUBSWATH_BURST_INDEX_START;
    params.main_scene_last_burst_index = params.secondary_scene_last_burst_index = FULL_SUBSWATH_BURST_INDEX_END;

    Initialize(params);
}

void Coregistration::Initialize(const Coregistration::Parameters& params) {
    if (!params.aoi.empty()) {
        split_reference_ = std::make_unique<topsarsplit::TopsarSplit>(params.main_scene_file, params.subswath,
                                                                      params.polarisation, params.aoi);
        split_secondary_ = std::make_unique<topsarsplit::TopsarSplit>(params.secondary_scene_file, params.subswath,
                                                                      params.polarisation, params.aoi);
    } else {
        size_t main_burst_start = params.main_scene_first_burst_index;
        size_t main_burst_end = params.main_scene_last_burst_index;
        if (main_burst_start == main_burst_end && main_burst_start < FULL_SUBSWATH_BURST_INDEX_START) {
            main_burst_start = FULL_SUBSWATH_BURST_INDEX_START;
            main_burst_end = FULL_SUBSWATH_BURST_INDEX_END;
        }
        split_reference_ = std::make_unique<topsarsplit::TopsarSplit>(
            params.main_scene_file, params.subswath, params.polarisation, main_burst_start, main_burst_end);

        size_t sec_burst_start = params.secondary_scene_first_burst_index;
        size_t sec_burst_end = params.secondary_scene_last_burst_index;
        if (sec_burst_start == sec_burst_end && sec_burst_start < FULL_SUBSWATH_BURST_INDEX_START) {
            sec_burst_start = FULL_SUBSWATH_BURST_INDEX_START;
            sec_burst_end = FULL_SUBSWATH_BURST_INDEX_END;
        }
        split_secondary_ = std::make_unique<topsarsplit::TopsarSplit>(
            params.secondary_scene_file, params.subswath, params.polarisation, sec_burst_start, sec_burst_end);
    }

    split_reference_->Initialize();
    split_reference_->OpenPixelReader(params.main_scene_file);
    std::shared_ptr<C16Dataset<int16_t>> reference_reader = split_reference_->GetPixelReader();
    if (!params.main_orbit_file.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(params.main_orbit_file);
    }
    auto orbit_file_master = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_reference_->GetTargetProduct(), true);
    orbit_file_master->Initialize();

    split_secondary_->Initialize();
    split_secondary_->OpenPixelReader(params.secondary_scene_file);
    if (!params.secondary_orbit_file.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(params.secondary_orbit_file);
    }
    auto orbit_file_slave = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_secondary_->GetTargetProduct(), true);
    orbit_file_slave->Initialize();

    auto* reference_temp = reference_reader->GetDataset();

    alus::TargetDatasetParams out_ds_params = {};
    out_ds_params.filename = params.output_file;
    out_ds_params.band_count = 4;
    out_ds_params.dataset_per_band = true;
    out_ds_params.driver = GetGdalMemDriver();
    out_ds_params.dimension = reference_temp->GetRasterDimensions();
    out_ds_params.transform = reference_temp->GetTransform();
    out_ds_params.projectionRef = reference_temp->GetGdalDataset()->GetProjectionRef();

    target_dataset_ = std::make_shared<alus::TargetDataset<float>>(out_ds_params);

    backgeocoding_ = std::make_unique<backgeocoding::BackgeocodingController>(
        reference_reader, split_secondary_->GetPixelReader(), target_dataset_, split_reference_->GetTargetProduct(),
        split_secondary_->GetTargetProduct());
}

void Coregistration::Initialize(std::shared_ptr<topsarsplit::TopsarSplit> split_reference,
                                std::shared_ptr<topsarsplit::TopsarSplit> split_secondary) {
    split_reference_ = std::move(split_reference);
    split_secondary_ = std::move(split_secondary);
    std::shared_ptr<C16Dataset<int16_t>> reference_reader = split_reference_->GetPixelReader();
    auto* reference_temp = reference_reader->GetDataset();

    alus::TargetDatasetParams out_ds_params = {};
    out_ds_params.filename = "";
    out_ds_params.band_count = 4;
    out_ds_params.dataset_per_band = true;
    out_ds_params.driver = GetGdalMemDriver();
    out_ds_params.dimension = reference_temp->GetRasterDimensions();
    out_ds_params.transform = reference_temp->GetTransform();
    out_ds_params.projectionRef = reference_temp->GetGdalDataset()->GetProjectionRef();

    target_dataset_ = std::make_shared<alus::TargetDataset<float>>(out_ds_params);

    backgeocoding_ = std::make_unique<backgeocoding::BackgeocodingController>(
        reference_reader, split_secondary_->GetPixelReader(), target_dataset_, split_reference_->GetTargetProduct(),
        split_secondary_->GetTargetProduct());
}

void Coregistration::DoWork(const float* egm96_device_array, PointerArray srtm3_tiles,
                            bool mask_out_area_without_elevation, const dem::Property* device_dem_properties,
                            const std::vector<dem::Property> dem_properties, dem::Type dem_type) const {
    backgeocoding_->PrepareToCompute(egm96_device_array, srtm3_tiles, mask_out_area_without_elevation,
                                     device_dem_properties, dem_properties, dem_type);
    backgeocoding_->DoWork();
}

}  // namespace alus::coregistration
