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
#include "guardian.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace alus {
namespace snapengine {

void Guardian::AssertNotNullOrEmpty(std::string_view expr_text, std::string_view expr_value) {
    if (expr_value.empty()) {
        throw std::invalid_argument(std::string(expr_text) + " argument is empty");
    }
}
void Guardian::AssertEquals(std::string_view expr_text, long expr_value, long expected_value) {
    if (expected_value != expr_value) {
        std::stringstream sb;
        sb << "[" << expr_text << "]"
           << " is [" << expr_value << "] but should be equal to [" << expected_value << "]";
        throw std::invalid_argument(sb.str());
    }
}
void Guardian::AssertWithinRange(std::string_view expr_text, long expr_value, long range_min, long range_max) {
    if (expr_value < range_min || expr_value > range_max) {
        std::stringstream sb;
        sb << "[" << expr_text << "]"
           << " is [" << expr_value << "]  but should be in the range [" << range_min << "] to [" << range_max << "]";
        throw std::invalid_argument(sb.str());
    }
}

// void Guardian::AssertSame(std::string_view expr_text, std::any expr_value, std::any expected_value) {
//    if (expected_value != expr_value) {
//        std::stringstream sb;
//        sb << "[" << expr_text << "]"
//           << " is [" << expr_value << "] but should be same as [" << expected_value << "]";
//        throw std::invalid_argument(sb.str());
//    }
//}

}  // namespace snapengine
}  // namespace alus
