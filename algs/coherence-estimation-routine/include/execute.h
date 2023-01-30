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

#include <memory>
#include <string>
#include <vector>

#include "constants.h"
#include "cuda_device_init.h"
#include "dem_assistant.h"
#include "topsar_split.h"

namespace alus::coherenceestimationroutine {

class Execute final {
public:
    struct Parameters {
        std::string subswath;
        std::string polarisation;
        std::string aoi;

        std::string timeline_start;
        std::string timeline_end;
        std::string timeline_input;
        std::string timeline_mission;

        std::string input_reference;
        std::string input_secondary;
        size_t burst_index_start_reference;
        size_t burst_index_last_reference;
        size_t burst_index_start_secondary;
        size_t burst_index_last_secondary;
        std::string orbit_dir;
        std::string orbit_reference;
        std::string orbit_secondary;
        size_t srp_number_points;
        size_t srp_polynomial_degree;
        bool subtract_flat_earth;
        size_t rg_window;
        size_t az_window;
        size_t orbit_degree;
        bool wif;
        std::string output;
        bool mask_out_area_without_elevation;
    };

    Execute() = delete;
    Execute(Parameters params, const std::vector<std::string>& dem_files);

    void RunSinglePair(alus::cuda::CudaInit& cuda_init, size_t gpu_mem_percentage) const;
    void RunTimeline(alus::cuda::CudaInit& cuda_init, size_t gpu_mem_percentage) const;

    Execute(const Execute& other) = delete;
    Execute& operator=(const Execute& other) = delete;

    ~Execute();

private:
    void CalcSingleCoherence(const std::vector<std::shared_ptr<alus::topsarsplit::TopsarSplit>>& reference_splits,
                             const std::vector<std::shared_ptr<alus::topsarsplit::TopsarSplit>>& secondary_splits,
                             const std::vector<std::string>& reference_swath_selection,
                             const std::vector<std::string>& secondary_swath_selection,
                             const std::string& reference_name, dem::Assistant* dem_assistant) const;

    void SplitApplyOrbit(const std::string& path, size_t burst_index_start, size_t burst_index_stop,
                         std::vector<std::shared_ptr<alus::topsarsplit::TopsarSplit>>& slave_splits,
                         std::vector<std::string>& swath_selection) const;
    std::string ConditionAoi(const std::string& aoi) const;

    Parameters params_;
    const std::vector<std::string>& dem_files_;
};

}  // namespace alus::coherenceestimationroutine
