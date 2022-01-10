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

namespace alus::featurextractiongabor {

namespace po = boost::program_options;

Arguments::Arguments() { ConstructCliArgs(); }

Arguments::Arguments(const std::vector<char*>& args) : Arguments() { ParseArgs(args); }

void Arguments::ConstructCliArgs() {
    // clang-format off
    app_args_.add_options()("help, h", "Print help")
        ("input,i", po::value<std::string>(&input_dataset_path_)->required(), "Input dataset path")
        ("destination,d", po::value<std::string>(&results_output_path_)->required(), "Results output path")
        ("frequency,f", po::value<size_t>(&frequency_count_)->required(), "Frequency count")
        ("patch,p", po::value<size_t>(&patch_edge_size_)->required(),
                    "Patch edge dimension in pixels (patches are squares)")
        ("orientation,o", po::value<size_t>(&orientation_count_)->required(), "Orientation count")
        ("conv_destination", po::value<std::string>(&convolution_inputs_path_), "Path to save convolution inputs");
    // clang-format on

    combined_args_.add(app_args_);
}

void Arguments::ParseArgs(const std::vector<char*>& args) {
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), app_args_), vm_);
    po::notify(vm_);
}

bool Arguments::IsHelpRequested() const { return vm_.count("help"); }

std::string Arguments::GetHelp() const {
    std::stringstream help;
    help << combined_args_;
    return help.str();
}

bool Arguments::IsConvolutionInputsRequested() const { return vm_.count("conv_destination"); }

}  // namespace alus::featurextractiongabor