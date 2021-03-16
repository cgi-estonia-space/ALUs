/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1Constants.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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

namespace alus::s1tbx {
class Sentinel1Constants {
private:
    static constexpr std::array<std::string_view, 1> FORMAT_NAMES = {"SENTINEL-1"};
    static constexpr std::array<std::string_view, 2> FORMAT_FILE_EXTENSIONS = {"safe", "zip"};
    static constexpr std::string_view PLUGIN_DESCRIPTION{"SENTINEL-1 Products"}; /*I18N*/
    static constexpr std::string_view INDICATION_KEY{"SAFE"};

public:
    static constexpr std::string_view PRODUCT_HEADER_PREFIX{"MANIFEST"};
    static constexpr std::string_view PRODUCT_HEADER_NAME{"manifest.safe"};

    //    final static Class[] VALID_INPUT_TYPES = new Class[]{Path.class, File.class, String.class};

    static std::string GetIndicationKey() { return std::string(INDICATION_KEY); }

    static std::string GetPluginDescription() { return std::string(PLUGIN_DESCRIPTION); }

    static auto GetFormatNames() { return FORMAT_NAMES; }

    static auto GetFormatFileExtensions() { return FORMAT_FILE_EXTENSIONS; }
};

}  // namespace alus::s1tbx
