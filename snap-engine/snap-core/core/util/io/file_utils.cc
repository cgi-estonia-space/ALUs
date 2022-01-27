/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.io.FileUtils.java
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
#include "snap-core/core/util/io/file_utils.h"

#include <boost/algorithm/string/predicate.hpp>

#include "../guardian.h"

namespace alus::snapengine {

std::string FileUtils::ExchangeExtension(std::string_view path, std::string_view extension) {
    Guardian::AssertNotNullOrEmpty("path", path);
    Guardian::AssertNotNull("extension", extension);
    if (extension.length() > 0 && boost::algorithm::ends_with(path, extension)) {
        return std::string(path);
    }
    auto extension_dot_pos = GetExtensionDotPos(path);
    if (extension_dot_pos > 0) {
        // replace existing extension
        return std::string(path.substr(0, extension_dot_pos)) + std::string(extension);
    }
    // append extension
    return std::string(path) + std::string(extension);
}

int FileUtils::GetExtensionDotPos(std::string_view path) {
    Guardian::AssertNotNullOrEmpty("path", path);
    //    todo: might need to rethink this logic
    std::size_t extension_dot_pos = 0;
    if (path.find_last_of('.') != std::string_view::npos) {
        extension_dot_pos = path.find_last_of('.');
    }
    if (extension_dot_pos > 0) {
        std::size_t last_separator_pos = 0;
        if (path.find_last_of('/') != std::string_view::npos) {
            last_separator_pos = path.find_last_of('/');
        }
        if (path.find_last_of('\\') != std::string_view::npos) {
            last_separator_pos = std::max(last_separator_pos, path.find_last_of('\\'));
        }
        if (path.find_last_of(':') != std::string_view::npos) {
            last_separator_pos = std::max(last_separator_pos, path.find_last_of(':'));
        }
        if (last_separator_pos < extension_dot_pos - 1) {
            return extension_dot_pos;
        }
    }
    return -1;
}
}  // namespace alus::snapengine