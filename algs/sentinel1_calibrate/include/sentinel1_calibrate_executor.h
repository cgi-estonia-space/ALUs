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

#include <cstddef>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "alg_bond.h"
#include "sentinel1_calibrate.h"

namespace alus::sentinel1calibrate {
class Sentinel1CalibrateExecutor : public AlgBond {
public:
    void SetParameters(const app::AlgorithmParameters::Table& param_values) override;
    void SetTileSize(size_t width, size_t height) override;
    int Execute() override;
    [[nodiscard]] std::string GetArgumentsHelp() const override;
    void SetInputFilenames(const std::vector<std::string>& input_datasets,
                           const std::vector<std::string>& metadata_paths) override;
    void SetOutputFilename(const std::string& output_name) override;

private:
    size_t tile_width_{};
    size_t tile_height_{};
    std::string input_file_{};
    std::vector<std::string> input_dataset_filenames_;
    std::vector<std::string> metadata_paths_;
    std::string output_dir_{};
    std::vector<std::string> sub_swaths_{};
    std::set<std::string, std::less<>> polarisations_{};
    sentinel1calibrate::SelectedCalibrationBands calibration_bands_{};

    void ParseCalibrationType(std::string_view calibration_string);
    void ValidateSubSwath();
    void ValidatePolarisation();
    void ValidateCalibrationType();
    void ValidateParameters();

    static bool DoesStringEqualAnyOf(std::string_view comparand, const std::vector<std::string>& string_list);
};
}  // namespace alus::sentinel1calibrate