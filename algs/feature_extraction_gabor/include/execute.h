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

#include <cstddef>
#include <string_view>
#include <vector>

#include "filter_bank.h"
#include "patched_image.h"

namespace alus::featurextractiongabor {

class Execute final {

public:

    Execute() = delete;
    Execute(size_t orientation_count, size_t frequency_count, size_t patch_edge_dimension, std::string_view input);

    Execute(const Execute& other) = delete;
    Execute& operator=(const Execute& other) = delete;

    void GenerateInputs();
    void SaveGaborInputsTo(std::string_view path) const;
    void CalculateGabor();
    void SaveResultsTo(std::string_view path) const { (void)path; }

    ~Execute();

private:

    std::vector<size_t> ExtractfilterEdgeSizes() const;
    void SaveFilterBanks(std::string_view path) const;
    void SavePatchedImages(std::string_view path) const;

    size_t orientation_count_;
    size_t frequency_count_;
    size_t patch_edge_dimension_;
    std::vector<FilterBankItem> filter_banks_;
    PatchedImage patched_image_;
};

}