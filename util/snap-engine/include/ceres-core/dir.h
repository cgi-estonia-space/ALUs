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

#include <boost/filesystem.hpp>

#include "ceres-core/i_virtual_dir.h"

namespace alus {
namespace ceres {
class Dir : public virtual IVirtualDir {
private:
    boost::filesystem::path dir_;

public:
    explicit Dir(const boost::filesystem::path& file) : IVirtualDir(){
        dir_ = file;
    }
    bool IsCompressed() override;
    std::vector<std::string> List(std::string_view path) override;
    boost::filesystem::path GetFile(std::string_view path) override;
    bool Exists(std::string_view path) override;
    void GetInputStream([[maybe_unused]] std::string_view path, [[maybe_unused]] std::fstream& stream) override;
    void Close() override;
};
}  // namespace ceres
}  // namespace alus
