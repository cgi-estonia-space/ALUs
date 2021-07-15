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
#include "snap-core/core/util/string_utils.h"

#include <iterator>
#include <sstream>
#include <algorithm>
#include <cctype>

#include "guardian.h"

namespace alus {
namespace snapengine {

std::vector<std::string> StringUtils::StringToVectorByDelimiter(std::string_view strv, std::string_view delimiter) {
    std::string str(strv);
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string_view token;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        tokens.emplace_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.size());
    }
    tokens.emplace_back(str);
    return tokens;
}

std::string StringUtils::PadNum(int num, int max, char c) {
    std::stringstream str;
    size_t len = std::to_string(num).length();
    while (len < static_cast<size_t>(max)) {
        str << c;
        len++;
    }
    str << num;
    return str.str();
}

std::string StringUtils::ArrayToString(std::vector<std::string> vec, std::string_view delimiter) {
    std::ostringstream oss;
    if (!vec.empty()) {
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<std::string>(oss, std::string(delimiter).c_str()));
        // last element with no delimiter
        oss << vec.back();
    }
    return oss.str();
}

std::string StringUtils::CreateValidName(std::string name, std::string valid_chars, char replace_char) {
    //Guardian.assertNotNull("name", name);
    std::string sorted_valid_chars;
    if (valid_chars.empty()) {
        sorted_valid_chars.push_back(0);
    } else {
        sorted_valid_chars = valid_chars;
    }
    std::sort(sorted_valid_chars.begin(), sorted_valid_chars.end());
    std::string valid_name;
    for (size_t i = 0; i < name.size(); i++) {
        char ch = name.at(i);
        if (std::isdigit(ch) || std::isalpha(ch)) {
            valid_name.push_back(ch);
        } else if (std::binary_search(sorted_valid_chars.begin(), sorted_valid_chars.end(), ch)) {
            valid_name.push_back(ch);
        } else {
            valid_name.push_back(replace_char);
        }
    }
    return valid_name;
}

}  // namespace snapengine
}  // namespace alus