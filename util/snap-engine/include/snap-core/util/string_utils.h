/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.StringUtils.java
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
#include <vector>

namespace alus {
namespace snapengine {

/**
 * consider using boost instead!
 */
class StringUtils {
public:
    static std::vector<std::string> StringToVectorByDelimiter(std::string_view str, std::string_view delimiter);

    /**
     * Adds padding to an integer
     * 1 becomes 001 or __1
     * @param num the integer value
     * @param max the desired string length
     * @param c the inserted character
     * @return padded number as string
     */
    static std::string PadNum(int num, int max, char c);

    /**
     * Converts an std::vector into a string.
     *
     * @param array the array object
     * @param s     the separator string, e.g. ","
     * @return a string represenation of the array
     * @throws std::invalid_argument if the given Object is not an <code>array</code> or <code>null</code>.
     */
    static std::string ArrayToString(std::vector<std::string>, std::string_view delimiter);

    /**
     * Creates a valid name for the given source name. The method returns a string which is the given name where each
     * occurence of a character which is not a letter, a digit or one of the given valid characters is replaced by the
     * given replace character. The returned string always has the same length as the source name.
     *
     * @param name        the source name, must not be  <code>null</code>
     * @param valid_chars  the array of valid characters
     * @param replace_char the replace character
     */
    static std::string CreateValidName(std::string name, std::string valid_chars, char replace_char);
};

}  // namespace snapengine
}  // namespace alus