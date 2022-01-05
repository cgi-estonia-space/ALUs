/**
 * This file is a filtered duplicate of a SNAP's
 * com.bc.ceres.core.VirtualDir.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "ceres-core/core/dir.h"

#include <string>

namespace alus::ceres {

bool ceres::Dir::IsCompressed() { return false; }

std::vector<std::string> Dir::List(std::string_view path) {
    //    todo: this must be checked over later
    auto child = GetFile(path);
    std::vector<std::string> name_set;
    if (boost::filesystem::exists(child) && boost::filesystem::is_directory(child)) {
        for (auto& entry : boost::filesystem::directory_iterator(child)) {
            name_set.emplace_back(entry.path().filename().string());
        }
    }
    return name_set;
}

boost::filesystem::path Dir::GetFile(std::string_view path) {
    //    todo: this must be checked over later
    boost::filesystem::path child{std::string(path)};
    return boost::filesystem::canonical(child, dir_);
}

bool Dir::Exists(std::string_view path) {
    //    todo: this must be checked over later
    boost::filesystem::path child{std::string(path)};
    try {
        auto check_path = boost::filesystem::canonical(child, dir_);
        return true;
    } catch (const boost::filesystem::filesystem_error& ex) {
        return false;
    }
}

void Dir::GetInputStream([[maybe_unused]] std::string_view path, [[maybe_unused]] std::fstream& stream) {
    // todo: add gzip support if used (boost has some good gzip options for this)
    stream.open(GetFile(path).generic_path().string(), std::ifstream::in);
}

void Dir::Close() {
    // this does nothing
}
}  // namespace alus::ceres