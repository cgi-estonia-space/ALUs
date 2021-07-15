/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.ProductSubsetBuilder.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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

#include "abstract_product_builder.h"

namespace alus::snapengine {

class ProductSubsetBuilder : public AbstractProductBuilder {
public:
    ProductSubsetBuilder();

protected:
    std::shared_ptr<Product> ReadProductNodesImpl();
    void AddBandsToProduct(std::shared_ptr<Product> product);
    /*void ReadBandRasterDataImpl(int sourceOffsetX, int sourceOffsetY, int sourceWidth, int sourceHeight,
                                int sourceStepX, int sourceStepY, std::shared_ptr<Band> destBand, int destOffsetX,
                                int destOffsetY, int destWidth, int destHeight,
                                const std::shared_ptr<ProductData>& dest_buffer,
                                std::shared_ptr<ceres::IProgressMonitor> pm);*/

private:
    static void UpdateMetadata(std::shared_ptr<Product> source_product, std::shared_ptr<Product> target_product,
                               std::shared_ptr<ProductSubsetDef> subset_def);
    static bool IsNearRangeOnLeft(std::shared_ptr<Product>& product);
    static void SetSubsetSRGRCoefficients(std::shared_ptr<Product>& source_product,
                                          std::shared_ptr<Product>& target_product,
                                          std::shared_ptr<ProductSubsetDef>& subset_def,
                                          std::shared_ptr<MetadataElement>& abs_root, bool near_range_on_left);
    static void SetLatLongMetadata(std::shared_ptr<Product>& product, std::shared_ptr<MetadataElement>& abs_root,
                                   std::string tag_lat, std::string tag_lon, float x, float y);

    std::shared_ptr<Product> CreateProduct();
};

}  // namespace alus::snapengine