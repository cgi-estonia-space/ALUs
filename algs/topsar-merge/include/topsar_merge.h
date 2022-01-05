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

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gdal_priv.h>
#include <boost/bimap.hpp>

#include "gdal_util.h"
#include "general_constants.h"
#include "s1tbx-commons/sentinel1_utils.h"
#include "s1tbx-commons/subswath_info.h"
#include "shapes.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-engine-utilities/engine-utilities/gpf/tile_index.h"
#include "topsar_merge_utils.h"

namespace alus::topsarmerge {

class TopsarMergeOperator {
public:
    TopsarMergeOperator(std::vector<std::shared_ptr<snapengine::Product>> source_products,
                        std::vector<std::string> selected_polarisations, int tile_width, int tile_height,
                        std::string_view output_path);
    void Compute();
    std::shared_ptr<snapengine::Product> GetTargetProduct() const;
    TopsarMergeOperator() = delete;
    TopsarMergeOperator(const TopsarMergeOperator&) = delete;
    TopsarMergeOperator& operator=(const TopsarMergeOperator&) = delete;

private:
    static constexpr size_t MINIMAL_AMOUNT_OF_PRODUCTS{2};

    int tile_width_;
    int tile_height_;

    MergeOperatorParameters parameters_;

    std::vector<std::shared_ptr<snapengine::Product>> source_products_;
    boost::bimap<product_map_index, sub_swath_map_index> source_product_index_to_sub_swath_index_map_;
    std::map<sub_swath_map_index, std::shared_ptr<s1tbx::Sentinel1Utils>> sentinel_utils_;
    std::map<sub_swath_map_index, std::shared_ptr<s1tbx::SubSwathInfo>> sub_swath_info_;
    std::vector<std::string> selected_polarisations_;
    std::shared_ptr<snapengine::Product> target_product_{nullptr};
    std::vector<Rectangle> target_rectangles_;
    std::shared_ptr<GDALDataset> target_dataset_;
    std::string output_path_;
    std::map<sub_swath_map_index, SubSwathMergeInfo> sub_swath_merge_info_;

    static constexpr std::string_view PRODUCT_SUFFIX{"_mrg"};
    static constexpr int TIE_POINT_GRID_WIDTH{20};
    static constexpr int TIE_POINT_GRID_HEIGHT{5};
    static constexpr double TIE_POINT_GRID_OFFSET_X{0};
    static constexpr double TIE_POINT_GRID_OFFSET_Y{0};

    [[nodiscard]] std::shared_ptr<snapengine::Band> GetSourceBandFromTargetBandName(
        std::string_view target_band_name, std::string_view acquisition_mode, std::string_view swath_index_string);
    [[nodiscard]] std::shared_ptr<snapengine::ITile> GetSourceTile(
        std::shared_ptr<snapengine::RasterDataNode> raster_data_node, Rectangle region, int band_index,
        int source_product_index);

    void ComputeTileInOneSwathFloat(Rectangle target_rectangle, int first_sub_swath_index, Rectangle source_rectangle,
                                    std::string_view target_band_name, snapengine::ITile* target_tile);
    void ComputeMultipleSubSwathFloat(Rectangle target_rectangle, int first_sub_swath_index, int last_sub_swath_index,
                                      const std::vector<Rectangle>& source_rectangles,
                                      std::string_view target_band_name, snapengine::ITile* target_tile);
    void ComputeTileStack(
        std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>>& target_tiles,
        const Rectangle& target_rectangle);
    void CreateTargetTiePointGrids();
    void UpdateTargetProductMetadata() const;
    void UpdateTargetProductAbstractedMetadata() const;
    void UpdateSubSwathParameters();
    void CreateTargetProduct();
    void Initialise();
    void CheckSourceProductsValidity();
    void CheckAreSubSwathsFromSameProduct();
    void PrepareBeamDimapSourceProducts() const;
    void ComputeTileInOneSwathShort(Rectangle target_rectangle, int first_sub_swath_index, Rectangle source_rectangle,
                                    std::string_view target_band_name, const snapengine::ITile* target_tile) const;
    void PopulateSourceBandNameVectors(int product_index) const;
};
}  // namespace alus::topsarmerge