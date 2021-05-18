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
#include <string_view>

namespace alus::utils::general {

// TODO: maybe remove inline and create a library?
inline void MeasureExecutionTime(std::string_view message, const std::function<void()>& code_block) {
    const auto start = std::chrono::steady_clock::now();
    code_block();
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << message << " " << elapsed_time << " ms" << std::endl;
}

inline void DisplayProgressBar(size_t counter, size_t total_size, const std::function<void()>& code_block) {
    code_block();
    const float progress{static_cast<float>(counter) / (total_size - 1)};
    const int bar_width_in_symbols{70};
    auto position = static_cast<int>(bar_width_in_symbols * progress);

    std::cout << "[";
    for (int i = 0; i < bar_width_in_symbols; ++i) {
        if (i < position) {
            std::cout << "=";
        } else if (i == position) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << static_cast<int>(progress * 100.0) << "%\r";
    std::cout.flush();
}

inline bool DoesStringContain(std::string_view string, std::string_view key) {
    return string.find(key) != std::string::npos;
}

template <typename T>
inline bool DoesVectorContain(const std::vector<T>& vector, T key) {
    return std::find(vector.begin(), vector.end(), key) != vector.end();
}
}  // namespace alus::utils::general