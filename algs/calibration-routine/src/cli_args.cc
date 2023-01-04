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

#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

#include "constants.h"

namespace alus::calibrationroutine {

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
    if (vm_.count("bi1") != vm_.count("bi2")) {
        throw std::invalid_argument("Burst indexes both must be either supplied or left undefined.");
    }

    alus_args_.Check();
}

bool Arguments::IsHelpRequested() const { return vm_.count("help"); }

std::string Arguments::GetHelp() const {
    std::stringstream help;
    help << combined_args_;
    return help.str();
}

std::optional<std::string> Arguments::GetAoi() const {
    if (!vm_.count("aoi")) {
        return std::nullopt;
    }

    return std::make_optional(aoi_);
}

std::optional<std::tuple<size_t, size_t>> Arguments::GetBurstIndexes() const {
    if (!vm_.count("bi1") || !vm_.count("bi2")) {
        return std::nullopt;
    }

    return std::make_tuple(burst_start_index_, burst_last_index_);
}

void Arguments::Construct() {
    std::string subswath_help = "Subswath for which the calibration will be performed, one of the following - ";
    for (const auto& s : SUBSWATHS) {
        subswath_help.append(s).append(";");
    }
    subswath_help.pop_back();

    std::string polarisation_help = "Polarisation for which the calibration will be performed - ";
    for (const auto& s : POLARISATIONS) {
        polarisation_help.append(s).append(";");
    }
    polarisation_help.pop_back();

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
        ("sw", po::value<std::string>(&subswath_), subswath_help.c_str())
        ("polarisation,p", po::value<std::string>(&polarisation_)->required(), polarisation_help.c_str())
        ("bi1", po::value<size_t>(&burst_start_index_),
            "First burst index - starting at '1', leave unspecified for whole subswath")
        ("bi2", po::value<size_t>(&burst_last_index_),
            "Last burst index - starting at '1', leave unspecified for whole subswath")
        ("aoi,a", po::value<std::string>(&aoi_),
            "Area Of Interest WKT polygon, overrules first and last burst indexes")
        ("type,t", po::value<std::string>(&calibration_type_)->required(), calibration_type_help.c_str())
        ("dem", po::value<std::vector<std::string>>(&dem_files_)->required(),
            "DEM file(s). Only SRTM3 is currently supported.");
    // clang-format on

    combined_args_.add(alg_args_).add(alus_args_.Get());
}

}  // namespace alus::calibrationroutine