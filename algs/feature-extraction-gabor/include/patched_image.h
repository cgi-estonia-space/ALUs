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

#include <map>
#include <string_view>
#include <vector>

#include "patch_assembly.h"

namespace alus::featurextractiongabor {

class PatchedImage final {
public:
    PatchedImage() = delete;
    explicit PatchedImage(std::string_view input_path);

    void CreatePatchedImagesFor(const std::vector<size_t>& filter_edge_sizes, size_t patch_edge_size);
    size_t GetBandCount() const { return band_count_; }
    const std::vector<size_t>& GetFilterEdgeSizes() const { return filter_edge_sizes_; }

    struct Item {
        PaddedPatchParameters padding;
        size_t width;
        size_t height;
        size_t filter_edge_size;
        std::vector<float> buffer;
    };
    const Item& GetPatchedImageFor(size_t band, size_t filter_edge_size) const;

    ~PatchedImage() = default;

private:
    std::string_view input_path_;
    size_t band_count_;
    std::vector<size_t> filter_edge_sizes_;
    // per band, vector contains all patches with different filter borders.
    std::map<size_t, std::vector<Item>> patched_images_;
};
}  // namespace alus::featurextractiongabor