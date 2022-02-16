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
#include "dem_assistant.h"

namespace alus::coherenceestimationroutine {

class Execute final {
public:
    struct Parameters {
        std::string input_reference;
        std::string input_secondary;
        std::string subswath;
        std::string polarisation;
        std::string aoi;
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
    };

    Execute() = delete;
    Execute(Parameters params, const std::vector<std::string>& dem_files);

    void Run() const;

    Execute(const Execute& other) = delete;
    Execute& operator=(const Execute& other) = delete;

    ~Execute();

private:
    const Parameters params_;
    std::shared_ptr<app::DemAssistant> dem_assistant_;
};

}  // namespace alus::coherenceestimationroutine
