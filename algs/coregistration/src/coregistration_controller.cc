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

#include "gdal_priv.h"

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

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation, size_t first_burst_index,
                                size_t last_burst_index) {
    Parameters params;
    params.main_scene_file = master_file;
    params.secondary_scene_file = slave_file;
    params.subswath = subswath_name;
    params.output_file = output_file;
    params.polarisation = polarisation;
    params.main_scene_first_burst_index = params.secondary_scene_first_burst_index = first_burst_index;
    params.main_scene_last_burst_index = params.secondary_scene_last_burst_index = last_burst_index;

    Initialize(params);
}

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation) {
    Parameters params;
    params.main_scene_file = master_file;
    params.secondary_scene_file = slave_file;
    params.subswath = subswath_name;
    params.output_file = output_file;
    params.polarisation = polarisation;
    params.main_scene_first_burst_index = params.secondary_scene_first_burst_index = FULL_SUBSWATH_BURST_INDEX_START;
    params.main_scene_last_burst_index = params.secondary_scene_last_burst_index = FULL_SUBSWATH_BURST_INDEX_END;

    Initialize(params);
}

void Coregistration::Initialize(const Coregistration::Parameters& params) {
    if (!params.aoi.empty()) {
        split_master_ = std::make_unique<topsarsplit::TopsarSplit>(params.main_scene_file, params.subswath,
                                                                   params.polarisation, params.aoi);
        split_slave_ = std::make_unique<topsarsplit::TopsarSplit>(params.secondary_scene_file, params.subswath,
                                                                  params.polarisation, params.aoi);
    } else {
        size_t main_burst_start = params.main_scene_first_burst_index;
        size_t main_burst_end = params.main_scene_last_burst_index;
        if (main_burst_start == main_burst_end && main_burst_start < FULL_SUBSWATH_BURST_INDEX_START) {
            main_burst_start = FULL_SUBSWATH_BURST_INDEX_START;
            main_burst_end = FULL_SUBSWATH_BURST_INDEX_END;
        }
        split_master_ = std::make_unique<topsarsplit::TopsarSplit>(
            params.main_scene_file, params.subswath, params.polarisation, main_burst_start, main_burst_end);

        size_t sec_burst_start = params.secondary_scene_first_burst_index;
        size_t sec_burst_end = params.secondary_scene_last_burst_index;
        if (sec_burst_start == sec_burst_end && sec_burst_start < FULL_SUBSWATH_BURST_INDEX_START) {
            sec_burst_start = FULL_SUBSWATH_BURST_INDEX_START;
            sec_burst_end = FULL_SUBSWATH_BURST_INDEX_END;
        }
        split_slave_ = std::make_unique<topsarsplit::TopsarSplit>(params.secondary_scene_file, params.subswath,
                                                                  params.polarisation, sec_burst_start, sec_burst_end);
    }

    split_master_->initialize();
    std::shared_ptr<C16Dataset<int16_t>> master_reader = split_master_->GetPixelReader();
    if (!params.main_orbit_file.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(params.main_orbit_file);
    }
    orbit_file_master_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_master_->GetTargetProduct(), true);
    orbit_file_master_->Initialize();

    split_slave_->initialize();
    if (!params.secondary_orbit_file.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(params.secondary_orbit_file);
    }
    orbit_file_slave_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_slave_->GetTargetProduct(), true);
    orbit_file_slave_->Initialize();

    auto* master_temp = master_reader->GetDataset();

    alus::TargetDatasetParams out_ds_params = {};
    out_ds_params.filename = params.output_file;
    out_ds_params.band_count = 4;
    out_ds_params.dataset_per_band = true;
    out_ds_params.driver = GetGdalMemDriver();
    out_ds_params.dimension = master_temp->GetRasterDimensions();
    out_ds_params.transform = master_temp->GetTransform();
    out_ds_params.projectionRef = master_temp->GetGdalDataset()->GetProjectionRef();

    target_dataset_ = std::make_shared<alus::TargetDataset<float>>(out_ds_params);

    backgeocoding_ = std::make_unique<backgeocoding::BackgeocodingController>(
        master_reader, split_slave_->GetPixelReader(), target_dataset_, split_master_->GetTargetProduct(),
        split_slave_->GetTargetProduct());
}

void Coregistration::DoWork(const float* egm96_device_array, PointerArray srtm3_tiles,
                            bool mask_out_area_without_elevation) const {
    backgeocoding_->PrepareToCompute(egm96_device_array, srtm3_tiles, mask_out_area_without_elevation);
    backgeocoding_->DoWork();
}

}  // namespace alus::coregistration