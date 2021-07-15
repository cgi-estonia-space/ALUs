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
#include "ceres-core/core/i_virtual_dir.h"

#include <boost/filesystem.hpp>

#include "ceres-core/core/ceres_assert.h"
#include "ceres-core/core/dir.h"
#include "ceres-core/core/zip.h"

namespace alus {
namespace ceres {

std::shared_ptr<IVirtualDir> IVirtualDir::Create(const boost::filesystem::path& file) {
    if (boost::filesystem::is_directory(file)) {
        return std::make_shared<Dir>(file);
    }
    try {
        return std::make_shared<Zip>(file);
    } catch (const std::exception& ignored) {
        return nullptr;
    }
}

boost::filesystem::path IVirtualDir::CreateUniqueTempDir() {
    boost::filesystem::path base_dir = GetBaseTempDir();

    // todo: move to some utility, when refactoring
    auto time = std::chrono::system_clock::now();  // get the current time
    auto since_epoch = time.time_since_epoch();    // get the duration since epoch
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch);
    auto now = millis.count();

    std::string base_name = std::to_string(now) + "-";

    for (int counter = 0; counter < TEMP_DIR_ATTEMPTS; counter++) {
        boost::filesystem::path temp_dir(boost::filesystem::canonical(base_dir).string() +
                                         boost::filesystem::path::preferred_separator + base_name +
                                         std::to_string(counter));
        if (boost::filesystem::create_directories(temp_dir)) {
            return temp_dir;
        }
    }
    throw std::runtime_error("Failed to create directory within " + std::to_string(TEMP_DIR_ATTEMPTS) +
                             " attempts (tried " + base_name + "0 to " + base_name +
                             std::to_string((TEMP_DIR_ATTEMPTS - 1)) + ')');
}

boost::filesystem::path IVirtualDir::GetBaseTempDir() {
    return boost::filesystem::temp_directory_path();
    //    boost::filesystem::path temp_dir = "temp_alus_dir";
    //    // this is custom solution, until we decide how we use external configuration
    //    // todo: need configuration files to provide paths
    //    if (!boost::filesystem::is_directory(temp_dir)) {
    //        boost::filesystem::create_directories(temp_dir);
    //    }
    //    if (!boost::filesystem::exists(temp_dir)) {
    //        throw std::runtime_error("Temporary directory not available: " + temp_dir.filename().string());
    //    }
    //    return temp_dir;
}
void IVirtualDir::DeleteFileTree(const boost::filesystem::path& tree_root) {
    if (!tree_root.empty()) {
        boost::filesystem::remove_all(tree_root);
    } else {
        throw std::invalid_argument("provided root directory path is empty ");
    }
}

}  // namespace ceres
}  // namespace alus
