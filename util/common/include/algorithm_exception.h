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

#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

namespace alus::common {
class AlgorithmException final : public std::runtime_error {
public:
    AlgorithmException(std::string_view algorithm_name, std::string_view reason)
        : AlgorithmException(algorithm_name, reason, "", -1) {}
    AlgorithmException(std::string_view algorithm_name, std::string_view reason, std::string_view file, int line)
        : std::runtime_error(std::string(algorithm_name) + " exception - '" + std::string{reason} + "' at " +
                             std::string{file} + ":" + std::to_string(line)),
          algorithm_name_{algorithm_name},
          reason_{reason},
          file_{file},
          line_{line} {}

    [[nodiscard]] std::string_view GetAlgorithmName() const { return algorithm_name_; }
    [[nodiscard]] std::string_view GetReason() const { return reason_; }
    [[nodiscard]] std::tuple<std::string_view, int> GetLocation() const { return {file_, line_}; }

private:
    const std::string algorithm_name_;
    const std::string reason_;
    const std::string file_;
    const int line_;
};

}  // namespace alus::common

#define THROW_ALGORITHM_EXCEPTION(algorithm_name, reason) \
    throw alus::common::AlgorithmException(algorithm_name, reason, __FILE__, __LINE__)
