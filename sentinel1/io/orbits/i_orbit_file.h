/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.orbits.OrbitFile.java
 * ported and modified for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

#include "snap-engine-utilities/engine-utilities/datamodel/orbit_vector.h"

namespace alus {
namespace s1tbx {
class IOrbitFile {
public:
    virtual std::vector<std::string> GetAvailableOrbitTypes() = 0;

    /**
     * download, find and read orbit file
     */
    virtual boost::filesystem::path RetrieveOrbitFile(std::string_view orbit_type) = 0;

    /**
     * Get orbit information for given time.
     *
     * @param utc The UTC in days.
     * @return The orbit information.
     */
    virtual std::shared_ptr<snapengine::OrbitVector> GetOrbitData(double utc) = 0;

    /**
     * Get the orbit file used
     *
     * @return the new orbit file
     */
    virtual boost::filesystem::path GetOrbitFile() = 0;

    IOrbitFile() = default;
    IOrbitFile(const IOrbitFile&) = delete;
    IOrbitFile& operator=(const IOrbitFile&) = delete;
    virtual ~IOrbitFile() = default;
};
}  // namespace s1tbx
}  // namespace alus