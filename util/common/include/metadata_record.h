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

#include <string_view>
#include <unordered_map>

namespace alus::common::metadata {

namespace sentinel1 {
constexpr std::string_view AREA_SELECTION{"AREA_SELECTION"};
constexpr std::string_view ORBIT_SOURCE{"ORBIT_SOURCE"};
}  // namespace sentinel1

class Container final {
public:
    Container() = default;

    void Add(std::string_view key, std::string value);
    void AddOrAppend(std::string_view key, std::string value);
    void AddWhenMissing(std::string_view key, std::string value);

    const std::unordered_map<std::string, std::string>& GetValues() const { return metadata_; }

    ~Container() = default;

private:
    std::unordered_map<std::string, std::string> metadata_;
};

}  // namespace alus::common::metadata