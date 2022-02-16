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
#include <string_view>

#include <boost/program_options.hpp>

#include "alus_log.h"

namespace alus::app {

std::string GenerateHelpMessage(std::string_view alg_name, std::string_view args_help);

class Arguments {
public:
    Arguments() = default;

    boost::program_options::options_description Get();
    void Check();
    [[nodiscard]] common::log::Level GetLogLevel() const { return log_level_; }
    [[nodiscard]] size_t GetGpuMemoryPercentage() const { return gpu_mem_percentage_; }

private:
    void CheckLogLevel();
    void CheckGpuMemoryPercentage() const;

    static constexpr size_t MAX_GPU_MEMORY_PERCENTAGE{100};
    std::string log_level_str_;
    common::log::Level log_level_{common::log::Level::VERBOSE};
    size_t gpu_mem_percentage_;
};
}  // namespace alus::app