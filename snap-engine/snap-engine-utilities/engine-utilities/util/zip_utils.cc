/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.util.ZipUtils.java
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
#include "snap-engine-utilities/engine-utilities/util/zip_utils.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "zipper/unzipper.h"
#include "zipper/zipper.h"

#include "alus_log.h"

namespace alus::snapengine {

bool ZipUtils::IsZipped(const boost::filesystem::path& file) {
    std::string name = file.filename().string();
    boost::to_lower(name);
    for (auto const& ext : EXT_LIST) {
        if (boost::algorithm::ends_with(name, ext)) {
            return true;
        }
    }
    return false;
}
bool ZipUtils::IsZip(const boost::filesystem::path& input_path) {
    //    auto path = input_path.filename().string();
    auto path = boost::filesystem::canonical(input_path).string();
    boost::to_lower(path);
    return boost::algorithm::ends_with(path, ".zip");
}

// todo: this needs further testing
std::string ZipUtils::GetRootFolder(const boost::filesystem::path& file, std::string_view header_file_name) {
    zipper::Unzipper unzipper(boost::filesystem::canonical(file).string());
    std::vector<zipper::ZipEntry> entries = unzipper.entries();
    unzipper.close();
    auto found_entry = std::find_if(std::begin(entries), std::end(entries), [header_file_name](const auto& entry) {
        if (!boost::ends_with(entry.name, "/")) {
            std::string entry_name_lower = entry.name;
            boost::to_lower(entry_name_lower);
            return boost::ends_with(entry_name_lower, header_file_name);
        }
        return false;
    });

    if (found_entry != entries.end()) {
        std::string found_entry_name = found_entry->name;
        auto sep_index = found_entry_name.rfind('/');
        if (sep_index > 0) {
            return found_entry_name.substr(0, sep_index) + "/";
        }
        return "";
    }
    return "";
}

bool ZipUtils::FindInZip(const boost::filesystem::path& file, std::string_view prefix, std::string_view suffix) {
    try {
        std::string lowercase_suffix(suffix);
        boost::algorithm::to_lower(lowercase_suffix);

        std::string lowercase_prefix(prefix);
        boost::algorithm::to_lower(lowercase_prefix);
        zipper::Unzipper product_zip(boost::filesystem::canonical(file).string());
        auto entries = product_zip.entries();
        auto filtered_entries =
            std::find_if(entries.begin(), entries.end(), [lowercase_suffix, lowercase_prefix](auto const& entry) {
                std::string lowercase_name = entry.name;
                boost::algorithm::to_lower(lowercase_name);
                return !boost::ends_with(entry.name, "/") &&
                       boost::algorithm::ends_with(lowercase_name, lowercase_suffix) &&
                       boost::algorithm::starts_with(lowercase_name, lowercase_prefix);
            });
        if (filtered_entries != entries.end()) {
            return true;
        }
    } catch (const std::exception& e) {
        LOGE << "unable to read zip file " << file.filename().string() << ": " << e.what();
    }
    return false;
}

}  // namespace alus::snapengine