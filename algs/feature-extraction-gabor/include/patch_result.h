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

#include <array>
#include <cstddef>
#include <map>
#include <vector>

#include "alus_log.h"

namespace alus ::featurextractiongabor {

struct PatchResult {
    size_t x;
    size_t y;
    float mean;
    float std_dev;
};

class Result {
public:
    [[nodiscard]] const std::vector<PatchResult>& GetPatchesReadOnly(size_t band, size_t orientation,
                                                                     size_t frequency) const {
        return results_.at(band).at(orientation).at(frequency);
    }

    [[nodiscard]] std::vector<PatchResult>& GetResultRef(size_t band, size_t orientation, size_t frequency) {
        return results_[band][orientation][frequency];
    }

    void LogConsoleResult() const {
        LOGI << "Results:";

        for (const auto& band : results_) {
            LOGI << "band = " << band.first;

            for (const auto& orientation : band.second) {
                LOGI << "orientation index = " << orientation.first;
                for (const auto& frequency : orientation.second) {
                    LOGI << "frequency index = " << frequency.first;
                    for (const auto& patch_result : frequency.second) {
                        LOGI << "patch = (" << patch_result.x << ", " << patch_result.y << ")"
                             << " mean = " << patch_result.mean << " std dev = " << patch_result.std_dev;
                    }
                }
            }
        }
    }

private:
    // band -> orientation -> frequency -> results of individual patches
    std::map<size_t, std::map<size_t, std::map<size_t, std::vector<PatchResult>>>> results_;
};

}  // namespace alus::featurextractiongabor