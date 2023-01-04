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

#include <ogr_spatialref.h>

#include "patch_assembly.h"
#include "raster_properties.h"

namespace alus::featurextractiongabor {

class PatchedImage final {
public:
    PatchedImage() = delete;
    explicit PatchedImage(std::string_view input_path);

    void CreatePatchedImagesFor(const std::vector<size_t>& filter_edge_sizes, size_t patch_edge_size);
    [[nodiscard]] size_t GetBandCount() const { return band_count_; }
    [[nodiscard]] const std::vector<size_t>& GetFilterEdgeSizes() const { return filter_edge_sizes_; }

    struct Item {
        PaddedPatchParameters padding;
        size_t width;
        size_t height;
        size_t filter_edge_size;
        std::vector<float> buffer;
    };
    [[nodiscard]] const Item& GetPatchedImageFor(size_t band, size_t filter_edge_size) const;
    /**
     * Should be called after CreatePatchedImageFor()
     */
    [[nodiscard]] GeoTransformParameters GetPatchesGeoTransform() const { return patches_gt_; }
    /**
     * Should be called after CreatePatchedImageFor()
     */
    [[nodiscard]] OGRSpatialReference GetPatchesSrs() const { return patches_srs_; }
    /**
     * Should be called after CreatePatchedImageFor()
     */
    [[nodiscard]] RasterDimension GetPatchesAggregateDimension() const { return patches_aggregate_dimension_; };

    ~PatchedImage() = default;

private:
    std::string_view input_path_;
    size_t band_count_;
    std::vector<size_t> filter_edge_sizes_;
    // per band, vector contains all patches with different filter borders.
    std::map<size_t, std::vector<Item>> patched_images_;
    GeoTransformParameters patches_gt_{};
    OGRSpatialReference patches_srs_{};
    RasterDimension patches_aggregate_dimension_{};
};
}  // namespace alus::featurextractiongabor