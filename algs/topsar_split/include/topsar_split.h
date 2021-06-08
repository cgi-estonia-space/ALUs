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

#include <sentinel1_utils.h>
#include <memory>
#include <string>

#include "c16_dataset.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader.h"
#include "sentinel1_utils.h"
#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/datamodel/product.h"
#include "split_product_subset_builder.h"
#include "subswath_info.h"

namespace alus::topsarsplit {

class TopsarSplit {
public:
    TopsarSplit(std::string filename, std::string selected_subswath, std::string selected_polarisation);
    void initialize();

    std::shared_ptr<snapengine::Product> GetTargetProduct() { return target_product_; }
    [[nodiscard]] const std::shared_ptr<C16Dataset<double>>& GetPixelReader() const { return pixel_reader_; }

private:
    std::shared_ptr<snapengine::IProductReader> reader_;
    std::shared_ptr<snapengine::Product> source_product_;
    std::shared_ptr<snapengine::Product> target_product_;
    std::unique_ptr<s1tbx::Sentinel1Utils> s1_utils_;
    std::unique_ptr<snapengine::SplitProductSubsetBuilder> subset_builder_;
    std::shared_ptr<C16Dataset<double>> pixel_reader_;

    std::string subswath_;
    std::vector<std::string> selected_polarisations_;

    int first_burst_index_ = 1;
    int last_burst_index_ = 9999;

    s1tbx::SubSwathInfo* selected_subswath_info_ = nullptr;

    void UpdateAbstractedMetadata();
    void RemoveBursts(std::shared_ptr<snapengine::MetadataElement>& orig_meta);
    void UpdateImageInformation(std::shared_ptr<snapengine::MetadataElement>& orig_meta);
    void UpdateOriginalMetadata();
    void UpdateTargetProductMetadata();
    void RemoveElements(std::shared_ptr<snapengine::MetadataElement>& orig_meta, std::string parent);
};

}  // namespace alus::topsarsplit
