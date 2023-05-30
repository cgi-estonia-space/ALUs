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

#include "product_name.h"

#include <filesystem>

namespace alus::common {

ProductName::ProductName(std::string_view path_or_filename)
    : path_separator_{std::filesystem::path::preferred_separator} {
    auto initial_path = std::filesystem::path(path_or_filename);
    if (!std::filesystem::is_directory(initial_path)) {
        filename_ = initial_path.filename();
        root_dir_ = initial_path.parent_path().string();
    } else {
        root_dir_ = path_or_filename;
    }
}

ProductName::ProductName(std::string_view path_or_filename, char delimiter) : ProductName(path_or_filename) {
    delimiter_ = delimiter;
}

void ProductName::Add(std::string_view part_of_a_name) {
    if (!filename_.empty()) {
        filename_ += delimiter_;
    }
    filename_ += part_of_a_name;
}

}  // namespace alus::common