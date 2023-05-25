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

#include "cli_args.h"

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "constants.h"

namespace alus::sarsegment {

namespace po = boost::program_options;

Arguments::Arguments() { Construct(); }

Arguments::Arguments(const std::vector<char*>& args) : Arguments() { Parse(args); }

void Arguments::Parse(const std::vector<char*>& args) {
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), combined_args_), vm_);
    if (vm_.count("wif")) {
        wif_ = true;
    }
}

void Arguments::Check() {
    boost::program_options::notify(vm_);

    alus_args_.Check();
}

bool Arguments::IsHelpRequested() const { return vm_.count("help"); }

std::string Arguments::GetHelp() const {
    std::stringstream help;
    help << combined_args_;
    return help.str();
}

void Arguments::Construct() {
    std::string calibration_type_help = "Type of calibration to be performed, one of the following - ";
    calibration_type_help.append(CALIBRATION_TYPE_SIGMA).append(";");
    calibration_type_help.append(CALIBRATION_TYPE_BETA).append(";");
    calibration_type_help.append(CALIBRATION_TYPE_GAMMA).append(";");
    calibration_type_help.append(CALIBRATION_TYPE_DN);
    // clang-format off
    alg_args_.add_options()
        ("help,h", "Print help")
        ("input,i", po::value<std::string>(&input_)->required(), "Input SAFE dataset (zipped or unpacked)")
        ("output,o", po::value<std::string>(&output_)->required(), "Output folder or filename")
        ("wif,w",
         "Write intermediate results (will be saved in the same folder as final outcome)."
         " NOTE - this may decrease performance. Default OFF.")
        ("type,t", po::value<std::string>(&calibration_type_)->required(), calibration_type_help.c_str())
        ("dem", po::value<std::vector<std::string>>(&dem_files_)->required(),
         "DEM file(s). SRTM3 and Copernicus DEM 30m COG are currently supported.");
    // clang-format on

    combined_args_.add(alg_args_).add(alus_args_.Get());
}
}  // namespace alus::sarsegment
