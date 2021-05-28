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

#include <memory>

#include "apply_orbit_file_op.h"
#include "backgeocoding_controller.h"
#include "topsar_split.h"

namespace alus::coregistration {

class Coregistration {
public:
    Coregistration(std::string aux_data_path);
    Coregistration() = default;

    void Initialize(std::string master_file, std::string slave_file, std::string output_file, std::string subswath_name,
                    std::string polarisation);
    void Initialize(std::string master_file, std::string slave_file, std::string output_file, std::string subswath_name,
                    std::string polarisation, const std::string& main_orbit_file,
                    const std::string& secondary_orbit_file);
    void DoWork(const float* egm96_device_array, PointerArray srtm3_tiles);

    std::shared_ptr<snapengine::Product> GetMasterProduct() { return split_master_->GetTargetProduct(); }
    std::shared_ptr<snapengine::Product> GetSlaveProduct() { return split_slave_->GetTargetProduct(); }

    ~Coregistration() = default;

private:
    std::unique_ptr<backgeocoding::BackgeocodingController> backgeocoding_;
    std::unique_ptr<topsarsplit::TopsarSplit> split_master_;
    std::unique_ptr<topsarsplit::TopsarSplit> split_slave_;
    std::unique_ptr<s1tbx::ApplyOrbitFileOp> orbit_file_master_;
    std::unique_ptr<s1tbx::ApplyOrbitFileOp> orbit_file_slave_;
    std::string main_orbit_file_{};
    std::string secondary_orbit_file_{};
};

}  // namespace alus::coregistration