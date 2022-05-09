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

#include <s1tbx-commons/sentinel1_utils.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "c16_dataset.h"
#include "s1tbx-commons/sentinel1_utils.h"
#include "s1tbx-commons/subswath_info.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader.h"
#include "snap-core/core/dataio/i_product_reader.h"
#include "snap-core/core/datamodel/product.h"
#include "split_product_subset_builder.h"

namespace alus::topsarsplit {

class TopsarSplit {
public:
    TopsarSplit(std::string_view filename, std::string_view selected_subswath, std::string_view selected_polarisation);
    TopsarSplit(std::string_view filename, std::string_view selected_subswath, std::string_view selected_polarisation,
                size_t first_burst, size_t last_burst);
    TopsarSplit(std::string_view filename, std::string_view selected_subswath, std::string_view selected_polarisation,
                std::string_view aoi_polygon_wkt);

    TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                std::string_view selected_polarisation);
    TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                std::string_view selected_polarisation, std::string_view aoi_polygon_wkt);
    TopsarSplit(std::shared_ptr<snapengine::Product> source_product, std::string_view selected_subswath,
                std::string_view selected_polarisation, size_t first_burst, size_t last_burst);

    void Initialize();
    [[nodiscard]] std::shared_ptr<snapengine::Product> GetTargetProduct() const { return target_product_; }
    void OpenPixelReader(std::string_view filename);
    [[nodiscard]] const std::shared_ptr<C16Dataset<int16_t>>& GetPixelReader() const { return pixel_reader_; }

    constexpr static int BURST_INDEX_OFFSET{1};

private:
    std::shared_ptr<snapengine::Product> source_product_;
    std::shared_ptr<snapengine::Product> target_product_;
    std::unique_ptr<s1tbx::Sentinel1Utils> s1_utils_;
    std::shared_ptr<snapengine::SplitProductSubsetBuilder> subset_builder_;
    std::shared_ptr<C16Dataset<int16_t>> pixel_reader_;

    std::string subswath_;
    std::vector<std::string> selected_polarisations_;

    int first_burst_index_{BURST_INDEX_OFFSET};
    int last_burst_index_ = 9999;
    std::string burst_aoi_wkt_{};
    s1tbx::SubSwathInfo* selected_subswath_info_ = nullptr;
    alus::Rectangle split_reading_area_;

    void LoadInputDataset(std::string_view filename);
    void UpdateAbstractedMetadata();
    void RemoveBursts(std::shared_ptr<snapengine::MetadataElement>& orig_meta);
    void UpdateImageInformation(std::shared_ptr<snapengine::MetadataElement>& orig_meta);
    void UpdateOriginalMetadata();
    void UpdateTargetProductMetadata();
    void RemoveElements(std::shared_ptr<snapengine::MetadataElement>& orig_meta, std::string parent);
};
}  // namespace alus::topsarsplit
