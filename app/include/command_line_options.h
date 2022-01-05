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

#include <boost/program_options.hpp>

namespace alus::app {

class CommandLineOptions final {
public:
    CommandLineOptions();
    CommandLineOptions(int argc, const char* argv[]);

    [[nodiscard]] std::string GetHelp() const;

    void ParseArgs(int argc, const char* argv[]);
    [[nodiscard]] bool DoRequireHelp() const;
    [[nodiscard]] bool DoRequireAlgorithmHelp() const;
    [[nodiscard]] const auto& GetSelectedAlgorithm() const { return alg_to_run_; }
    [[nodiscard]] const auto& GetInputDatasets() const { return input_files_; }
    [[nodiscard]] const auto& GetOutputStem() const { return output_path_; }
    [[nodiscard]] const auto& GetAux() const { return aux_locations_; }
    [[nodiscard]] auto GetTileWidth() const { return tile_width_; }
    [[nodiscard]] auto GetTileHeight() const { return tile_height_; }
    [[nodiscard]] const auto& GetAlgorithmParameters() const { return alg_params_; }
    [[nodiscard]] const auto& GetAlgorithmParametersFile() const { return params_file_path_; }
    [[nodiscard]] const auto& GetDemFiles() const { return dem_files_param_; }

    ~CommandLineOptions() = default;

private:
    void ConstructCommandLineOptions();

    boost::program_options::variables_map vm_;
    boost::program_options::options_description visible_options_{"Arguments"};
    boost::program_options::options_description hidden_options_{"Hidden arguments"};
    boost::program_options::options_description all_options_{"All arguments"};

    std::string alg_to_run_{};
    std::vector<std::string> input_files_{};
    std::string output_path_{};
    std::vector<std::string> aux_locations_{};
    size_t tile_width_{};
    size_t tile_height_{};
    std::string alg_params_{};
    std::string params_file_path_{};
    std::vector<std::string> dem_files_param_{};
};
}  // namespace alus::app