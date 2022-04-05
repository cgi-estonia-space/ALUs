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

#include <map>
#include <string>
#include <vector>

namespace alus::tnr {
struct TimeMaps {
    std::map<std::string, double> t_0_map;
    std::map<std::string, double> delta_t_map;
    std::map<std::string, std::vector<double>> swath_start_end_times_map;
};
}