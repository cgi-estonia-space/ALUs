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

#include <string_view>
#include <string>
#include <memory>

#include "alg_bond.h"
#include "backgeocoding_controller.h"
#include "dataset.h"
#include "target_dataset.h"
#include "algorithm_parameters.h"
#include "pointer_holders.h"
#include "raster_properties.hpp"

namespace alus::backgeocoding {

constexpr std::string_view PARAMETER_MASTER_PATH{"master"};
constexpr std::string_view PARAMETER_SLAVE_PATH{"slave"};
constexpr std::string_view PARAMETER_MASTER_METADATA{"master_metadata"};
constexpr std::string_view PARAMETER_SLAVE_METADATA{"slave_metadata"};
constexpr std::string_view PARAMETER_OUTPUT_PATH{"output"};
constexpr std::string_view PARAMETER_MASK_ELEVATION{"mask_elevation"};  // TODO: this will be used some day

class BackgeocodingBond : public AlgBond {
public:
    BackgeocodingBond() = default;

    virtual void SetInputFilenames([[maybe_unused]] const std::string& input_dataset,
                                   [[maybe_unused]] const std::string&
                                   metadata_path) override;
    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;
    void SetSrtm3Buffers(const PointerHolder* buffers, size_t length) override;
    void SetEgm96Buffers(const float* egm96_device_array) override;
    void SetTileSize(size_t width, size_t height) override;
    void SetOutputFilename([[maybe_unused]] const std::string& output_name) override;
    int Execute() override;

    [[nodiscard]] std::string GetArgumentsHelp() const override;

    ~BackgeocodingBond() override;

private:
    std::shared_ptr<alus::Dataset<double>> master_input_dataset_;
    std::shared_ptr<alus::Dataset<double>> slave_input_dataset_;
    std::shared_ptr<alus::TargetDataset<float>> output_dataset_;

    std::unique_ptr<BackgeocodingController> controller_;

    std::string master_metadata_path_;
    std::string slave_metadata_path_;
    bool use_elevation_mask_;
    PointerArray srtm3_tiles_;
    const float* egm96_device_array_;

    bool srtm3_set_ = false;
    bool egm96_set_ = false;
    bool inputs_set_ = false;
    bool outputs_set_ = false;
};

}  // namespace alus::backgeocoding
