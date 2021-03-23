/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.SystemUtils.java
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

#include <string_view>

#include <boost/filesystem.hpp>

namespace alus {
namespace snapengine {
/**
 * A collection of (BEAM-) system level functions.
 * <p>
 * <p> All functions have been implemented with extreme caution in order to provide a maximum performance.
 *
 * java version original authors Norman Fomferra, Sabine Embacher
 */
class SystemUtils {
   public:
    /**
     * Name of SNAP's auxdata directory.
     */
    static constexpr std::string_view AUXDATA_DIR_NAME{"auxdata"};
    /**
     * Gets the auxdata directory which stores dems, orbits, rgb profiles, etc.
     *
     * @return the auxiliary data directory
     * @since SNAP 2.0
     */
    static boost::filesystem::path GetAuxDataPath() {
        return GetApplicationDataDir().toPath().resolve(AUXDATA_DIR_NAME);
    }

    /**
     * Gets the current user's application data directory.
     *
     * @return the current user's application data directory
     * @since BEAM 4.2
     */
    static boost::filesystem::path GetApplicationDataDir() { return GetApplicationDataDir(false); }

    /**
     * Optionally creates and returns the current user's application data directory.
     *
     * @param force if true, the directory will be created if it didn't exist before
     * @return the current user's application data directory
     * @since BEAM 4.2
     */
    static boost::filesystem::path GetApplicationDataDir(bool force) {
        boost::filesystem::path dir = Config.instance().userDir().toFile();
        if (force && !(boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir))) {
            boost::filesystem::create_directory(dir);
        }
        return dir;
    }
};

}  // namespace snapengine
}  // namespace alus