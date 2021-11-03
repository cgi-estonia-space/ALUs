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
#pragma once

#include <fstream>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

#include "zipper/unzipper.h"
#include "zipper/zipper.h"

#include "ceres-core/core/i_virtual_dir.h"

namespace alus {
namespace ceres {
class Zip : public IVirtualDir {
private:
    boost::filesystem::path zip_file_;
    boost::filesystem::path temp_zip_file_dir_;

    zipper::ZipEntry GetEntry(std::string_view path);
    void Unzip(const zipper::ZipEntry& zip_entry, const boost::filesystem::path& temp_file);
    void GetInputStream(const zipper::ZipEntry& zip_entry, std::fstream& stream);
    void Cleanup();

public:
    explicit Zip(const boost::filesystem::path& file) : zip_file_(file) {}
    bool IsCompressed() override { return true; }
    std::vector<std::string> List(std::string_view path) override;
    boost::filesystem::path GetFile(std::string_view path) override;
    bool Exists(std::string_view path) override;
    void GetInputStream(std::string_view path, std::fstream& stream) override;
    void Close() override;
};
}  // namespace ceres
}  // namespace alus
