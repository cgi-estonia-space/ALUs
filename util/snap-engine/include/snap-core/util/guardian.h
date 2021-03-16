/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.Guardian.java
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

#include <any>
#include <memory>
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {
class Guardian {
public:
    static void AssertNotNullOrEmpty(std::string_view expr_text, std::string_view text);
    template <typename T>
    static void AssertNotNull(std::string_view expr_text, T expr_value) {
        if (expr_value == nullptr) {
            throw std::invalid_argument(std::string(expr_text) + " argument is nullptr");
        }
    }

    /**
     * Checks if the given values are equal. If not, an <code>std::invalid_argument</code> is thrown with a
     * standardized message text using the supplied message.
     * <p>This utility method is used to check arguments passed into methods:
     * <pre>
     * WriteDataAtRegion(int x, int y, int w, int h, std::vector<byte> data) {
     *     Guardian::AssertEquals("data.length",
     *                           data.length, w * h);
     *     ...
     * }
     * </pre>
     *
     * @param expr_text      the test expression as text
     * @param expr_value     the test expression result
     * @param expected_value the expected value
     * @throws std::invalid_argument if the <code>expr_value</code> is not equal to <code>expected_value</code>
     */
    static void AssertEquals(std::string_view expr_text, long expr_value, long expected_value);

    //     todo:might need to use std::any or use T and override equality...  check Product::CheckGeoCoding for usage
    //    static void AssertSame(std::string_view expr_text, std::any expr_value, std::any expected_value);

    /**
     * Checks if the given value are in the given range. If not, an <code>IllegalArgumentException</code> is thrown with
     * a standardized message text using the supplied value name.
     * <p>This utility method is used to check arguments passed into methods:
     * <pre>
     * public void writeDataAtRegion(int x, inty, int w, int h, byte[] data) {
     *     Guardian.assertWithinRange("w", w, 0, data.length -1);
     *     ...
     * }
     * </pre>
     *
     * @param exprText  the test expression as text
     * @param exprValue the expression result
     * @param rangeMin  the range lower limit
     * @param rangeMax  the range upper limit
     * @throws IllegalArgumentException if the <code>exprValue</code> is less than <code>rangeMin</code> or greater than
     * <code>rangeMax</code>
     */
    static void AssertWithinRange(std::string_view expr_text, long expr_value, long range_min, long range_max);
};

}  // namespace snapengine
}  // namespace alus
