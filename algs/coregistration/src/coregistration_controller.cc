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
#include "snap-core/util/system_utils.h"
#include "target_dataset.h"
#include "topsar_split.h"

namespace alus::coregistration {

Coregistration::Coregistration(std::string aux_data_path) {
    alus::snapengine::SystemUtils::SetAuxDataPath(aux_data_path);
}

void Coregistration::Initialize(std::string master_file, std::string slave_file, std::string output_file,
                                std::string subswath_name, std::string polarisation) {
    split_master_ = std::make_unique<topsarsplit::TopsarSplit>(master_file, subswath_name, polarisation);
    split_master_->initialize();
    std::shared_ptr<C16Dataset<double>> master_reader = split_master_->GetPixelReader();
    orbit_file_master_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_master_->GetTargetProduct(), true);
    orbit_file_master_->Initialize();

    split_slave_ = std::make_unique<topsarsplit::TopsarSplit>(slave_file, subswath_name, polarisation);
    split_slave_->initialize();
    orbit_file_slave_ = std::make_unique<s1tbx::ApplyOrbitFileOp>(split_slave_->GetTargetProduct(), true);
    orbit_file_slave_->Initialize();

    auto* master_temp = master_reader->GetDataset();

    alus::TargetDatasetParams params;
    params.filename = output_file;
    params.band_count = 4;
    params.driver = master_temp->GetGdalDataset()->GetDriver();
    params.dimension = master_temp->GetRasterDimensions();
    params.transform = master_temp->GetTransform();
    params.projectionRef = master_temp->GetGdalDataset()->GetProjectionRef();

    std::shared_ptr<AlusFileWriter<float>> output_writer = std::make_shared<alus::TargetDataset<float>>(params);

    backgeocoding_ = std::make_unique<backgeocoding::BackgeocodingController>(
        master_reader, split_slave_->GetPixelReader(), output_writer, split_master_->GetTargetProduct(),
        split_slave_->GetTargetProduct());
}

void Coregistration::DoWork(const float* egm96_device_array, PointerArray srtm3_tiles) {
    backgeocoding_->PrepareToCompute(egm96_device_array, srtm3_tiles);
    backgeocoding_->DoWork();
}

}  // namespace alus::coregistration