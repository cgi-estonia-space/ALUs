/**
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

#include <chrono>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>

#include "alus_log.h"

namespace alus::utils::general {

// TODO: maybe remove inline and create a library?
inline void MeasureExecutionTime(std::string_view message, const std::function<void()>& code_block) {
    const auto start = std::chrono::steady_clock::now();
    code_block();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    LOGV << message << " " << elapsed_time << " ms";
}

inline void DisplayProgressBar(size_t counter, size_t total_size, const std::function<void()>& code_block) {
    code_block();
    const float progress{static_cast<float>(counter) / (total_size - 1)};
    const int bar_width_in_symbols{70};
    auto position = static_cast<int>(bar_width_in_symbols * progress);

    LOGV << "[";
    for (int i = 0; i < bar_width_in_symbols; ++i) {
        if (i < position) {
            LOGV << "=";
        } else if (i == position) {
            LOGV << ">";
        } else {
            LOGV << " ";
        }
    }
    LOGV << "] " << static_cast<int>(progress * 100.0) << "%\r";
    LOGV.flush();
}

inline bool DoesStringContain(std::string_view string, std::string_view key) {
    return string.find(key) != std::string::npos;
}

template <typename T>
inline bool DoesVectorContain(const std::vector<T>& vector, T key) {
    return std::find(vector.begin(), vector.end(), key) != vector.end();
}

/**
 * Joins vector of string into one string using the provided delimiter. This function is similar to Java's
 * String.join().
 *
 * @param delimiter delimiter which is used to join strings.
 * @param string_sequence vector of strings which will be joined.
 * @return strings from the vector joined with the given delimiter into one string.
 */
inline std::string JoinStrings(std::string_view delimiter, const std::vector<std::string>& string_sequence) {
    std::string result;
    for (size_t i = 0; i < string_sequence.size(); ++i) {
        result += string_sequence.at(i);
        if (i != string_sequence.size() - 1) {
            result += delimiter;
        }
    }

    return result;
}
}  // namespace alus::utils::general