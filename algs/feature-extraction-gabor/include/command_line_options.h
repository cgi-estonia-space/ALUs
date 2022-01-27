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
#include <string_view>
#include <vector>

#include <boost/program_options.hpp>

namespace alus::featurextractiongabor {

class Arguments final {
public:
    Arguments();
    explicit Arguments(const std::vector<char*>& args);

    void ParseArgs(const std::vector<char*>& args);

    bool IsHelpRequested() const;
    std::string GetHelp() const;
    std::string_view GetInput() const { return input_dataset_path_; }
    std::string_view GetOutputPath() const { return results_output_path_; }
    size_t GetOrientationCount() const { return orientation_count_; }
    size_t GetFrequencyCount() const { return frequency_count_; }
    size_t GetPatchSize() const { return patch_edge_size_; }
    bool IsConvolutionInputsRequested() const;
    std::string_view GetConvolutionInputsStorePath() const { return convolution_inputs_path_; }

    ~Arguments() = default;

private:
    void ConstructCliArgs();

    boost::program_options::variables_map vm_;
    boost::program_options::options_description app_args_{""};
    boost::program_options::options_description alus_args_{""};
    boost::program_options::options_description combined_args_{"Arguments"};

    std::string input_dataset_path_{};
    std::string results_output_path_{};
    size_t orientation_count_{};
    size_t frequency_count_{};
    size_t patch_edge_size_{};
    std::string convolution_inputs_path_{};
};

}  // namespace alus::featurextractiongabor