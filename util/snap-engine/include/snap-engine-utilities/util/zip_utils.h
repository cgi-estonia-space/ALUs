/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.engine_utilities.util.ZipUtils.java
 * ported for native code.
 * Copied from a snap-engine's (https://github.com/senbox-org/snap-engine) repository originally stated:
 *
 * Copyright (C) 2015 by Array Systems Computing Inc. http://www.array.ca
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
#pragma once

#include <array>
#include <string>
#include <string_view>

#include <boost/filesystem.hpp>

namespace alus {
namespace snapengine {

/**
 * For zipping and unzipping compressed files
 */
class ZipUtils {
private:
    static constexpr std::array<std::string_view, 3> EXT_LIST{".zip", ".gz", ".z"};

public:
    static bool IsZipped(const boost::filesystem::path& file);

    static bool IsZip(const boost::filesystem::path& input_path);

    static std::string GetRootFolder(const boost::filesystem::path& file, std::string_view header_file_name);

    static bool FindInZip(const boost::filesystem::path& file, std::string_view prefix, std::string_view suffix);
};

}  // namespace snapengine
}  // namespace alus
