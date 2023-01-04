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

#include <string>

#include <boost/program_options.hpp>

#include "app_utils.h"

namespace alus::palsar {
namespace po = boost::program_options;

class Arguments {
public:
    Arguments();

    void ParseArgs(int argc, char* argv[]);

    void Check();

    [[nodiscard]] const std::string& GetInputDirectory() const { return directory_path_; }
    [[nodiscard]] bool GetWriteIntermediateFiles() const { return write_temp_files_; }
    [[nodiscard]] bool GetOutputIntensity() const { return output_intensity_; }
    [[nodiscard]] const char* GetPolarisation() const { return polarisation_.c_str(); }
    [[nodiscard]] const std::string& GetOutputPath() const { return output_path_; };
    [[nodiscard]] bool GetPrintMetadataOnly() const { return print_metadata_only_; }
    [[nodiscard]] bool GetHelpRequested() const { return help_requested_; }
    [[nodiscard]] common::log::Level GetLogLevel() const { return alus_args_.GetLogLevel(); }
    [[nodiscard]] double GetGpuMemFraction() const {
        return static_cast<double>(alus_args_.GetGpuMemoryPercentage()) / 100.0;
    };
    [[nodiscard]] std::string GetHelp() const {
        std::stringstream help;
        help << combined_args_;
        return help.str();
    }

    std::string directory_path_;
    std::string output_path_;
    std::string polarisation_;
    bool output_intensity_ = false;
    bool write_temp_files_ = false;
    bool print_metadata_only_ = false;
    bool help_requested_ = false;
    boost::program_options::options_description alg_args_;
    app::Arguments alus_args_;
    boost::program_options::options_description combined_args_;
    po::variables_map vm_;
};
}  // namespace alus::palsar
