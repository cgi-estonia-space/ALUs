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
#include <vector>

#include <boost/program_options.hpp>

#include "alus_log.h"
#include "app_utils.h"

namespace alus::coherenceestimationroutine {
class Arguments final {
public:
    Arguments();
    explicit Arguments(const std::vector<char*>& args);

    void Parse(const std::vector<char*>& args);

    void Check();

    bool IsHelpRequested() const;
    std::string GetHelp() const;
    std::string GetInputReference() const { return input_reference_; }
    std::string GetInputSecondary() const { return input_secondary_; }
    std::string GetSubswath() const { return subswath_; }
    std::string GetPolarisation() const { return polarisation_; }
    std::optional<std::string> GetAoi() const;
    std::optional<std::tuple<size_t, size_t>> GetBurstIndexesReference() const;
    std::optional<std::tuple<size_t, size_t>> GetBurstIndexesSecondary() const;
    const std::vector<std::string>& GetDemFiles() const { return dem_files_; }
    std::string GetOutput() const { return output_; }
    bool DoSaveIntermediateResults() const { return wif_; };
    std::string GetOrbitFileReference() const { return orbit_file_reference_; }
    std::string GetOrbitFileSecondary() const { return orbit_file_secondary_; }
    std::string GetOrbitDirectory() const { return orbit_file_dir_; }
    size_t GetSrpNumberPoints() const { return srp_number_points_; }
    size_t GetSrpPolynomialDegree() const { return srp_polynomial_degree_; }
    size_t GetOrbitDegree() const { return orbit_degree_; }
    bool DoSubtractFlatEarthPhase() const { return subtract_flat_earth_phase_; };
    size_t GetRangeWindow() const { return range_window_; }
    size_t GetAzimuthWindow() const { return az_window_; }
    size_t GetGpuMemoryPercentage() const { return alus_args_.GetGpuMemoryPercentage(); }
    common::log::Level GetLogLevel() const { return alus_args_.GetLogLevel(); }

    ~Arguments() = default;

private:
    void Construct();

    boost::program_options::variables_map vm_;
    boost::program_options::options_description alg_args_{""};
    app::Arguments alus_args_;
    boost::program_options::options_description combined_args_{"Arguments"};

    std::string input_reference_;
    std::string input_secondary_;
    std::string subswath_;
    std::string polarisation_;
    std::string aoi_;
    size_t burst_start_index_reference_;
    size_t burst_last_index_reference_;
    size_t burst_start_index_secondary_;
    size_t burst_last_index_secondary_;
    std::vector<std::string> dem_files_;
    std::string output_;
    bool wif_{false};
    size_t srp_number_points_;
    size_t srp_polynomial_degree_;
    bool subtract_flat_earth_phase_{true};
    size_t range_window_;
    size_t az_window_;
    size_t orbit_degree_;
    std::string orbit_file_reference_;
    std::string orbit_file_secondary_;
    std::string orbit_file_dir_;
};

}  // namespace alus::coherenceestimationroutine