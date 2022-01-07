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

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

#include "zipper/unzipper.h"

namespace alus::common::zip {

constexpr std::string_view ZIP_EXTENSION = ".zip";

/**
 * Gets the list of files inside the given archive.
 *
 * @param archive_path Path to the archive.
 * @return Vector of strings where each string denotes a file inside the archive.
 */
inline std::vector<std::string> GetZipContents(std::string_view archive_path) {
    zipper::Unzipper archive(archive_path.data());
    const auto entries = archive.entries();
    std::vector<std::string> zip_content(entries.size());
    std::transform(entries.begin(), entries.end(), zip_content.begin(),
                   [](const auto& entry) { return entry.name; });
    archive.close();

    return zip_content;
}

/**
 * Checks whether the file is a zip archive using its file extension.
 *
 * @param archive_path Path to the archive
 * @return True if the file is a zip archive.
 */
inline bool IsFileAnArchive(std::string_view archive_path) {
    const auto path = boost::filesystem::path(archive_path.data());
    const auto extension = path.extension().string();
    return extension == ZIP_EXTENSION;
}

}  // namespace alus::common::zip