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


namespace alus::common {

class ProductName final {
public:
    ProductName(std::string_view path_or_filename, char delimiter);
    ProductName(std::string_view path_or_filename);

    bool IsFinal() const { return !filename_.empty(); }
    std::string GetDirectory() const { return root_dir_ + path_separator_; };
    std::string Construct(std::string extension = "") const {
        return root_dir_ + path_separator_ + filename_ + extension;
    }
    void Add(std::string_view part_of_a_name);

    ~ProductName() = default;

private:
    const char path_separator_;
    char delimiter_{'_'};
    std::string root_dir_{};
    std::string filename_{};
};

}  // namespace alus::common