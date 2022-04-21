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

#include "app_utils.h"

#include <sstream>
#include <stdexcept>
#include <string_view>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "../../VERSION"

namespace alus::app {

std::string GenerateHelpMessage(std::string_view alg_name, std::string_view args_help) {
    std::stringstream msg;
    msg << "ALUs - " << alg_name << std::endl;
    msg << "Version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
    msg << std::endl;
    msg << args_help << std::endl;
    msg << std::endl;
    msg << "https://bitbucket.org/cgi-ee-space/alus" << std::endl;

    return msg.str();
}

boost::program_options::options_description Arguments::Get() {
    boost::program_options::options_description args{""};

    // clang-format off
    args.add_options()
        ("ll", boost::program_options::value<std::string>(&log_level_str_)->default_value("verbose"),
            "Log level, one of the following - verbose|debug|info|warning|error")
        ("gpu_mem",
            boost::program_options::value<size_t>(&gpu_mem_percentage_)->default_value(MAX_GPU_MEMORY_PERCENTAGE),
            "Percentage of how much GPU memory can be used for processing");
    // clang-format on

    return args;
}

void Arguments::Check() {
    CheckLogLevel();
    CheckGpuMemoryPercentage();
}

void Arguments::CheckLogLevel() {
    constexpr std::array<std::string_view, 5> ALLOWED_LEVELS{"verbose", "debug", "info", "warning", "error"};
    constexpr std::array<common::log::Level, 5> LOG_LEVELS{common::log::Level::VERBOSE, common::log::Level::DEBUG,
                                                           common::log::Level::INFO, common::log::Level::WARNING,
                                                           common::log::Level::ERROR};

    size_t level_index{0};
    for (const auto level_str : ALLOWED_LEVELS) {
        if (boost::iequals(log_level_str_, level_str)) {
            log_level_ = LOG_LEVELS.at(level_index);
            return;
        }
        level_index++;
    }

    throw std::invalid_argument("'" + log_level_str_ +
                                "' is not a valid log level. Valid ones are - verbose|debug|info|warning|error");
}

void Arguments::CheckGpuMemoryPercentage() const {
    if (gpu_mem_percentage_ > MAX_GPU_MEMORY_PERCENTAGE) {
        throw std::invalid_argument("GPU memory percentage shall be defined as a number in range 1-100(including).");
    }
}

}  // namespace alus::app