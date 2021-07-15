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
#pragma once

#include <string>
#include <string_view>

namespace alus {
namespace snapengine {
/**
 * This class provides additional functionality in handling with files. All methods in this class dealing with
 * extensions, expect that an extension is the last part of a file name starting with the dot '.' character.
 *
 * original java version authors:Tom Block, Sabine Embacher, Norman Fomferra
 */
class FileUtils {
public:
    /**
     * Returns the file string with the given new extension. If the given file string have no extension, the given
     * extension will be added.
     * <p>
     * Example1:
     * <pre> "tie.point.grids\tpg1.hdr" </pre>
     * results to
     * <pre> "tie.point.grids\tpg1.raw" </pre>
     * <p>
     * Example2:
     * <pre> "tie.point.grids\tpg1" </pre>
     * results to
     * <pre> "tie.point.grids\tpg1.raw" </pre>
     *
     * @param path      the string to change the extension
     * @param extension the new file extension including a leading dot (e.g. <code>".raw"</code>).
     * @throws java.lang.IllegalArgumentException if one of the given strings are null or empty.
     */
    static std::string ExchangeExtension(std::string_view path, std::string_view extension);

    static int GetExtensionDotPos(std::string_view path);
};
}  // namespace snapengine
}  // namespace alus
