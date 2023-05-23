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
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

#include "alus_log.h"
#include "app_utils.h"

namespace alus::calibrationroutine {

class Arguments final {
public:
    Arguments();
    explicit Arguments(const std::vector<char*>& args);

    void Parse(const std::vector<char*>& args);

    void Check();

    bool IsHelpRequested() const;
    std::string GetHelp() const;
    std::string GetInput() const { return input_; }
    std::string GetSubswath() const { return subswath_; }
    std::string GetPolarisation() const { return polarisation_; }
    std::optional<std::string> GetAoi() const;
    std::optional<std::tuple<size_t, size_t>> GetBurstIndexes() const;
    std::string GetCalibrationType() const { return calibration_type_; }
    const std::vector<std::string>& GetDemFiles() const { return dem_files_; }
    std::string GetOutput() const { return output_; }
    bool DoSaveIntermediateResults() const { return wif_; };
    bool OutputValuesInDb() const { return db_values_; }
    size_t GetGpuMemoryPercentage() const { return alus_args_.GetGpuMemoryPercentage(); }
    common::log::Level GetLogLevel() const { return alus_args_.GetLogLevel(); }

    ~Arguments() = default;

private:
    void Construct();

    boost::program_options::variables_map vm_;
    boost::program_options::options_description alg_args_{""};
    app::Arguments alus_args_;
    boost::program_options::options_description combined_args_{"Arguments"};

    std::string input_;
    std::string subswath_;
    std::string polarisation_;
    std::string aoi_;
    size_t burst_start_index_;
    size_t burst_last_index_;
    std::string calibration_type_;
    std::vector<std::string> dem_files_;
    bool db_values_{false};
    std::string output_;
    bool wif_{false};
};
}  // namespace alus::calibrationroutine