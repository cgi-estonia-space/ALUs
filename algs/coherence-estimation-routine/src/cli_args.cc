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

#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "constants.h"

namespace alus::coherenceestimationroutine {

namespace po = boost::program_options;

Arguments::Arguments() { Construct(); }

Arguments::Arguments(const std::vector<char*>& args) : Arguments() { Parse(args); }

void Arguments::Construct() {
    std::string polarisation_help = "Polarisation for which coherence estimation will be performed - ";
    for (const auto& s : POLARISATIONS) {
        polarisation_help.append(s).append(";");
    }
    polarisation_help.pop_back();

    // clang-format off
    alg_args_.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Print help")
        ("in_ref,r", po::value<std::string>(&input_reference_)->required(),
        "Reference scene's input SAFE dataset (zipped or unpacked)")
        ("in_sec,s", po::value<std::string>(&input_secondary_)->required(),
         "Secondary scene's input SAFE dataset (zipped or unpacked)")
        ("output,o", po::value<std::string>(&output_)->required(), "Output folder or filename")
        ("polarisation,p", po::value<std::string>(&polarisation_)->required(), polarisation_help.c_str())
        ("sw", po::value<std::string>(&subswath_)->required(), "Reference scene's subswath")
        ("b_ref1", po::value<size_t>(&burst_start_index_reference_),
         "Reference scene's first burst index - starting at '1', leave unspecified for whole subswath")
        ("b_ref2", po::value<size_t>(&burst_last_index_reference_),
         "Reference scene's last burst index - starting at '1', leave unspecified for whole subswath")
        ("b_sec1", po::value<size_t>(&burst_start_index_secondary_),
         "Secondary scene's first burst index - starting at '1', leave unspecified for whole subswath")
        ("b_sec2", po::value<size_t>(&burst_last_index_secondary_),
         "Secondary scene's last burst index - starting at '1', leave unspecified for whole subswath")
        ("aoi,a", po::value<std::string>(&aoi_),
         "Area Of Interest WKT polygon, overrules first and last burst indexes")
        ("dem", po::value<std::vector<std::string>>(&dem_files_)->required(),
         "DEM file(s). Only SRTM3 is currently supported.")
        ("orbit_ref", po::value<std::string>(&orbit_file_reference_),"Reference scene's POEORB file")
        ("orbit_sec", po::value<std::string>(&orbit_file_secondary_), "Secondary scenes's POEORB file")
        ("orbit_dir", po::value<std::string>(&orbit_file_dir_),
        "ESA SNAP compatible root folder of orbit files. Can be used to find correct one during processing. "
        "For example: /home/<user>/.snap/auxData/Orbits/Sentinel-1/POEORB/")
        ("srp_number_points", po::value<size_t>(&srp_number_points_)->default_value(501), "")
        ("srp_polynomial_degree", po::value<size_t>(&srp_polynomial_degree_)->default_value(5), "")
        ("subtract_flat_earth_phase", po::bool_switch(&subtract_flat_earth_phase_)->default_value(true),
        "Compute flat earth phase subtraction during coherence operation. By default on.")
        ("rg_win", po::value<size_t>(&range_window_)->default_value(15u),
        "range window size in pixels.")
        ("az_win", po::value<size_t>(&az_window_)->default_value(0u),
        "azimuth window size in pixels, if zero derived from range window.")
        ("orbit_degree", po::value<size_t>(&orbit_degree_)->default_value(3u), "")
        ("wif,w", po::bool_switch(&wif_)->default_value(false),
         "Write intermediate results (will be saved in the same folder as final outcome)."
        " NOTE - this may decrease performance. By default off.");
    // clang-format on

    combined_args_.add(alg_args_).add(alus_args_.Get());
}

void Arguments::Parse(const std::vector<char*>& args) {
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), combined_args_), vm_);
}

bool Arguments::IsHelpRequested() const { return vm_.at("help").as<bool>(); }

std::string Arguments::GetHelp() const {
    std::stringstream help;
    help << combined_args_;
    return help.str();
}

void Arguments::Check() {
    boost::program_options::notify(vm_);
    if ((vm_.count("b_ref1") != vm_.count("b_ref2")) != (vm_.count("b_sec1") != vm_.count("b_sec2"))) {
        throw std::invalid_argument(
            "All burst indexes must be either supplied or left undefined. "
            "Use -a [ --aoi ] to skip defining burst indexes.");
    }

    if ((!vm_.count("orbit_ref") || !vm_.count("orbit_sec")) && !vm_.count("orbit_dir")) {
        throw std::invalid_argument(
            "Orbit files must be supplied for both scenes. "
            "Use --orbit_dir for determining the right one during processing.");
    }

    alus_args_.Check();
}

std::optional<std::string> Arguments::GetAoi() const {
    if (!vm_.count("aoi")) {
        return std::nullopt;
    }

    return std::make_optional(aoi_);
}

std::optional<std::tuple<size_t, size_t>> Arguments::GetBurstIndexesReference() const {
    if (!vm_.count("b_ref1") || !vm_.count("b_ref2")) {
        return std::nullopt;
    }

    return std::make_tuple(burst_start_index_reference_, burst_last_index_reference_);
}

std::optional<std::tuple<size_t, size_t>> Arguments::GetBurstIndexesSecondary() const {
    if (!vm_.count("b_sec1") || !vm_.count("b_sec2")) {
        return std::nullopt;
    }

    return std::make_tuple(burst_start_index_secondary_, burst_last_index_secondary_);
}

}  // namespace alus::coherenceestimationroutine