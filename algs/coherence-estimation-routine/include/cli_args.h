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
    explicit Arguments(bool timeline_args);

    void Parse(const std::vector<char*>& args);

    void Check();

    [[nodiscard]] bool IsHelpRequested() const;
    [[nodiscard]] std::string GetHelp() const;
    [[nodiscard]] std::string GetTimelineStart() const { return timeline_start_; }
    [[nodiscard]] std::string GetTimelineEnd() const { return timeline_end_; }
    [[nodiscard]] std::string GetTimelineInput() const { return timeline_input_; }
    [[nodiscard]] std::string GetTimelineMission() const { return timeline_mission_; }
    [[nodiscard]] std::string GetInputReference() const { return input_reference_; }
    [[nodiscard]] std::string GetInputSecondary() const { return input_secondary_; }
    [[nodiscard]] std::optional<std::tuple<size_t, size_t>> GetBurstIndexesReference() const;
    [[nodiscard]] std::optional<std::tuple<size_t, size_t>> GetBurstIndexesSecondary() const;
    [[nodiscard]] std::string GetSubswath() const { return subswath_; }
    [[nodiscard]] std::string GetPolarisation() const { return polarisation_; }
    [[nodiscard]] std::optional<std::string> GetAoi() const;
    [[nodiscard]] const std::vector<std::string>& GetDemFiles() const { return dem_files_; }
    [[nodiscard]] std::string GetOutput() const { return output_; }
    [[nodiscard]] bool DoSaveIntermediateResults() const { return wif_; };
    [[nodiscard]] std::string GetOrbitFileReference() const { return orbit_file_reference_; }
    [[nodiscard]] std::string GetOrbitFileSecondary() const { return orbit_file_secondary_; }
    [[nodiscard]] std::string GetOrbitDirectory() const { return orbit_file_dir_; }
    [[nodiscard]] size_t GetSrpNumberPoints() const { return srp_number_points_; }
    [[nodiscard]] size_t GetSrpPolynomialDegree() const { return srp_polynomial_degree_; }
    [[nodiscard]] size_t GetOrbitDegree() const { return orbit_degree_; }
    [[nodiscard]] bool DoSubtractFlatEarthPhase() const { return subtract_flat_earth_phase_; };
    [[nodiscard]] size_t GetRangeWindow() const { return range_window_; }
    [[nodiscard]] size_t GetAzimuthWindow() const { return az_window_; }
    [[nodiscard]] bool DoMaskOutAreaWithoutElevation() const { return !disable_coregistration_elevation_mask_; }
    [[nodiscard]] size_t GetGpuMemoryPercentage() const { return alus_args_.GetGpuMemoryPercentage(); }
    [[nodiscard]] common::log::Level GetLogLevel() const { return alus_args_.GetLogLevel(); }

    ~Arguments() = default;

private:
    void Construct();

    boost::program_options::variables_map vm_;
    boost::program_options::options_description alg_args_{""};
    app::Arguments alus_args_;
    boost::program_options::options_description combined_args_{"Arguments"};

    bool timeline_args_;
    std::string input_reference_;
    std::string input_secondary_;
    size_t burst_start_index_reference_;
    size_t burst_last_index_reference_;
    size_t burst_start_index_secondary_;
    size_t burst_last_index_secondary_;

    std::string timeline_start_;
    std::string timeline_end_;
    std::string timeline_input_;
    std::string timeline_mission_;

    std::string subswath_;
    std::string polarisation_;
    std::string aoi_;
    std::vector<std::string> dem_files_;
    std::string output_;
    bool wif_{false};
    size_t srp_number_points_;
    size_t srp_polynomial_degree_;
    bool subtract_flat_earth_phase_{true};
    bool disable_coregistration_elevation_mask_{false};
    size_t range_window_;
    size_t az_window_;
    size_t orbit_degree_;
    std::string orbit_file_reference_;
    std::string orbit_file_secondary_;
    std::string orbit_file_dir_;
};

}  // namespace alus::coherenceestimationroutine