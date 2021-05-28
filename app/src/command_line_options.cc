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

#include "command_line_options.h"

#include <sstream>

#include <boost/program_options.hpp>

namespace alus::app {

namespace po = boost::program_options;

CommandLineOptions::CommandLineOptions() { ConstructCommandLineOptions(); }

CommandLineOptions::CommandLineOptions(int argc, const char* argv[]) : CommandLineOptions() { ParseArgs(argc, argv); }

void CommandLineOptions::ParseArgs(int argc, const char* argv[]) {
    po::store(po::parse_command_line(argc, argv, all_options_), vm_);
    po::notify(vm_);
}

bool CommandLineOptions::DoRequireHelp() const {
    if (vm_.count("help") || vm_.count("alg_name") == 0) {
        return true;
    }

    return false;
}

std::string CommandLineOptions::GetHelp() const {
    std::stringstream help;
    help << visible_options_;
    return help.str();
}

bool CommandLineOptions::DoRequireAlgorithmHelp() const { return vm_.count("alg_help"); }

void CommandLineOptions::ConstructCommandLineOptions() {
    // clang-format off
    visible_options_.add_options()("help,h", "Print help")(
        "alg_name", po::value<std::string>(&alg_to_run_), "Specify algorithm to run")(
        "alg_help", "Print algorithm configurable parameters")(
        "input,i", po::value<std::vector<std::string>>(&input_files_), "Input dataset path.")(
        "output,o", po::value<std::string>(&output_path_), "Output dataset location or name")(
        "tile_width,x", po::value<size_t>(&tile_width_)->default_value(1000), "Tile width.")(
        "tile_height,y", po::value<size_t>(&tile_height_)->default_value(1000), "Tile height.")
        ("parameters,p", po::value<std::string>(&alg_params_)->default_value(""),
         "Algorithm specific configuration. Must be supplied as key=value "
         "pairs "
         "separated by comma','.\n"
         "Example 1: 'orbit_degree=3,rg_window=15'\n"
         "Example 2: 'coherence:az_window=3;backgeocoding:master=...'\nParameters specified "
         "here will overrule ones in the configuration file supplied as '--parameters_file' option.")
        ("parameters_file", po::value<std::string>(&params_file_path_)->default_value(""), "Algorithm specific "
         "configurations file path. File contents shall follow same syntax as '--parameters' option.")(
        "dem", po::value<std::vector<std::string>>(&dem_files_param_), "Dem files with full path. Can be specified "
         "multiple times for multiple DEM files or a space separated files in a single argument. Only SRTM3 is "
                                                                                                   "supported.")(
        "list_algs,l", "Print available algorithms");

    hidden_options_.add_options()(
        "aux", po::value<std::vector<std::string>>(&aux_locations_), "Auxiliary/metadata .dim file"
                                                                    "(metadata, incident angle, etc).");
    // clang-format on
    all_options_.add(visible_options_).add(hidden_options_);
}

}  // namespace alus::app