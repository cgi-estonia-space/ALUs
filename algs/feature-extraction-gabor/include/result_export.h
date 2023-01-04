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

#include <cstddef>
#include <map>
#include <string_view>
#include <utility>
#include <vector>

#include <gdal_priv.h>
#include <ogrsf_frmts.h>

#include "patch_result.h"
#include "raster_properties.h"

namespace alus::featurextractiongabor {

class ResultExport final {
public:
    ResultExport() = delete;
    ResultExport(std::string_view filename, OGRSpatialReference* srs, GeoTransformParameters patches_corner_gt,
                 size_t orientation_count, size_t frequency_count);

    void Add(const PatchResult& result);
    void StoreFeatures();

    ~ResultExport();

private:
    void SetupAttributeTable();

    GeoTransformParameters patches_corner_gt_;
    GDALDataset* feature_ds_{};
    OGRLayer* layer_{};
    // patch x index -> patch y index -> array of mean and std dev values.
    std::map<std::pair<size_t, size_t>, std::vector<std::pair<float, float>>> features_x_y_{};
    const std::string input_parameters_attribute_value_;
};

}  // namespace alus::featurextractiongabor
