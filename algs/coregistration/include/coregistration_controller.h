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
#include "dem_property.h"
#include "dem_type.h"
#include "target_dataset.h"
#include "topsar_split.h"

namespace alus::coregistration {

class Coregistration {
public:
    struct Parameters {
        std::string_view main_scene_file{};
        std::string_view secondary_scene_file{};
        std::string_view output_file{};
        std::string_view subswath{};
        std::string_view polarisation{};
        std::string_view main_orbit_file{};
        std::string_view secondary_orbit_file{};
        size_t main_scene_first_burst_index{};
        size_t main_scene_last_burst_index{};
        size_t secondary_scene_first_burst_index{};
        size_t secondary_scene_last_burst_index{};
        std::string_view aoi{};
    };

    explicit Coregistration(const std::string& aux_data_path);

    Coregistration() = default;

    void Initialize(const Parameters&);

    void Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                    std::string_view subswath_name, std::string_view polarisation, size_t first_burst_index,
                    size_t last_burst_index);

    void Initialize(std::string_view master_file, std::string_view slave_file, std::string_view output_file,
                    std::string_view subswath_name, std::string_view polarisation);

    void Initialize(std::shared_ptr<topsarsplit::TopsarSplit> split_reference,
                    std::shared_ptr<topsarsplit::TopsarSplit> split_secondary);

    void DoWork(const float* egm96_device_array, PointerArray srtm3_tiles, bool mask_out_area_without_elevation,
                const dem::Property* device_dem_properties, const std::vector<dem::Property> dem_properties,
                dem::Type dem_type) const;

    std::shared_ptr<snapengine::Product> GetReferenceProduct() { return split_reference_->GetTargetProduct(); }

    std::shared_ptr<snapengine::Product> GetSecondaryProduct() { return split_secondary_->GetTargetProduct(); }

    std::shared_ptr<alus::TargetDataset<float>> GetTargetDataset() { return target_dataset_; }

    ~Coregistration() = default;

private:
    std::unique_ptr<backgeocoding::BackgeocodingController> backgeocoding_;
    std::shared_ptr<topsarsplit::TopsarSplit> split_reference_;
    std::shared_ptr<topsarsplit::TopsarSplit> split_secondary_;
    std::shared_ptr<alus::TargetDataset<float>> target_dataset_;
};

}  // namespace alus::coregistration
