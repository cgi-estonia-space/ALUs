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

namespace alus::sentinel1calibrate {
class Sentinel1CalibrateException : public std::runtime_error {
public:
    explicit Sentinel1CalibrateException(const char* message)
        : std::runtime_error(std::string("Error has occurred in Sentinel1 Calibration Operator: ") + message) {}
    explicit Sentinel1CalibrateException(const std::string& message)
        : std::runtime_error("Error has occurred in Sentinel1 Calibration Operator: " + message) {}

    Sentinel1CalibrateException() = delete;
};
}  // namespace alus::sentinel1calibrate