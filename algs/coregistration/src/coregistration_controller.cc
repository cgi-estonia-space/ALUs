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
#include "snap-core/util/alus_utils.h"
#include "snap-core/util/system_utils.h"
#include "target_dataset.h"
#include "topsar_split.h"

namespace alus::coregistration {

Coregistration::Coregistration(std::string aux_data_path) {
    alus::snapengine::SystemUtils::SetAuxDataPath(aux_data_path);
}

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation,
                                size_t master_first_burst_index, size_t master_last_burst_index,
                                size_t slave_first_burst_index, size_t slave_last_burst_index) {

    split_master_ = std::make_unique<topsarsplit::TopsarSplit>(master_file, subswath_name, polarisation,
                                                               master_first_burst_index, master_last_burst_index);
    split_master_->initialize();
    std::shared_ptr<C16Dataset<double>> master_reader = split_master_->GetPixelReader();
    if (!main_orbit_file_.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(main_orbit_file_);
    }
    orbit_file_master_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_master_->GetTargetProduct(), true);
    orbit_file_master_->Initialize();

    split_slave_ = std::make_unique<topsarsplit::TopsarSplit>(slave_file, subswath_name, polarisation,
                                                              slave_first_burst_index, slave_last_burst_index);
    split_slave_->initialize();
    if (!secondary_orbit_file_.empty()) {
        snapengine::AlusUtils::SetOrbitFilePath(secondary_orbit_file_);
    }
    orbit_file_slave_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_slave_->GetTargetProduct(), true);
    orbit_file_slave_->Initialize();

    auto* master_temp = master_reader->GetDataset();

    alus::TargetDatasetParams params;
    params.filename = output_file;
    params.band_count = 4;
    params.driver = GetGDALDriverManager()->GetDriverByName(utils::constants::GDAL_MEM_DRIVER);
    params.dimension = master_temp->GetRasterDimensions();
    params.transform = master_temp->GetTransform();
    params.projectionRef = master_temp->GetGdalDataset()->GetProjectionRef();

    target_dataset_ = std::make_shared<alus::TargetDataset<float>>(params);

    backgeocoding_ = std::make_unique<backgeocoding::BackgeocodingController>(
        master_reader, split_slave_->GetPixelReader(), target_dataset_, split_master_->GetTargetProduct(),
        split_slave_->GetTargetProduct());
}

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation, size_t first_burst_index,
                                size_t last_burst_index) {
    Initialize(master_file, slave_file, output_file, subswath_name, polarisation, first_burst_index, last_burst_index,
               first_burst_index, last_burst_index);
}

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation) {
    Initialize(master_file, slave_file, output_file, subswath_name, polarisation, 1, 9999);
}

void Coregistration::Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                                std::string_view subswath_name, std::string_view polarisation,
                                std::string_view main_orbit_file, std::string_view secondary_orbit_file) {
    main_orbit_file_ = main_orbit_file;
    secondary_orbit_file_ = secondary_orbit_file;
    Initialize(master_file, slave_file, output_file, subswath_name, polarisation, 1, 9999);
}

void Coregistration::DoWork(const float* egm96_device_array, PointerArray srtm3_tiles) {
    backgeocoding_->PrepareToCompute(egm96_device_array, srtm3_tiles);
    backgeocoding_->DoWork();
}

}  // namespace alus::coregistration