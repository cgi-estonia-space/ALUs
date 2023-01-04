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

#include <boost/algorithm/string.hpp>

namespace alus::palsar {

Arguments::Arguments() {
    // clang-format off
    alg_args_.add_options()
        ("help,h", po::bool_switch(&help_requested_),"Print help")
        ("input,i", po::value<std::string>(&directory_path_)->required(),"Input directory path")
        ("output,o", po::value<std::string>(&output_path_)->required(), "Output path")
        ("polarisation,p", po::value<std::string>(&polarisation_)->required(), "Polarisation - HH, HV, VH or VV")
        ("intensity", po::bool_switch(&output_intensity_), "Output images as Float32 intensity instead of CFloat32 complex")
        ("wif", po::bool_switch(&write_temp_files_), "Write intermediate files")
        ("metadata", po::bool_switch(&print_metadata_only_), "Only print metadata, skip processing");
    // clang-format on
    combined_args_.add(alg_args_).add(alus_args_.Get());
}

void Arguments::ParseArgs(int argc, char* argv[]) {
    po::store(po::parse_command_line(argc, argv, combined_args_), vm_);
}

void Arguments::Check() {
    po::notify(vm_);

    alus_args_.Check();

    polarisation_ = boost::to_upper_copy(polarisation_);
    if (polarisation_ != "HH" && polarisation_ != "HV" && polarisation_ != "VH" && polarisation_ != "VV") {
        throw std::invalid_argument("Unknown polarisation " + polarisation_);
    }
}
}  // namespace alus::palsar