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
#include <unordered_map>

namespace alus::common::metadata {

namespace sentinel1 {
constexpr std::string_view AREA_SELECTION{"AREA_SELECTION"};
constexpr std::string_view ORBIT_SOURCE{"ORBIT_SOURCE"};
constexpr std::string_view BACKGEOCODING_NO_ELEVATION_MASK{"BACKGEOCODING_NO_ELEVATION_MASK"};
constexpr std::string_view COH_WIN_AZ{"COH_WIN_AZ"};
constexpr std::string_view COH_WIN_RG{"COH_WIN_RG"};
constexpr std::string_view SRP_POLYNOMIAL_DEGREE{"SRP_POLYNOMIAL_DEGREE"};
constexpr std::string_view SRP_NUMBER_POINTS{"SRP_NUMBER_POINTS"};
constexpr std::string_view ORBIT_DEGREE{"ORBIT_DEGREE"};
constexpr std::string_view SUBTRACT_FLAT_EARTH_PHASE{"SUBTRACT_FLAT_EARTH_PHASE"};
}  // namespace sentinel1

inline std::string CreateBooleanValue(bool value) {
    return value ? "YES" : "NO";
}

class Container final {
public:
    Container();

    void Add(std::string_view key, std::string value);
    void AddOrAppend(std::string_view key, std::string value);
    void AddWhenMissing(std::string_view key, std::string value);

    const std::unordered_map<std::string, std::string>& GetValues() const { return metadata_; }

    ~Container() = default;

private:

    std::unordered_map<std::string, std::string> metadata_;
};

}  // namespace alus::common::metadata