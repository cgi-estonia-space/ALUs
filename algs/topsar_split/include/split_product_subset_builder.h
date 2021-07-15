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

#include "snap-core/core/dataio/product_subset_builder.h"

namespace alus::snapengine{
/**
 * Works exactly like ProductSubsetBuilder, but without the raster data copy
 */
class SplitProductSubsetBuilder : public ProductSubsetBuilder{
public:
protected:
    void ReadBandRasterDataImpl(int /*sourceOffsetX*/, int /*sourceOffsetY*/, int /*sourceWidth*/, int /*sourceHeight*/,
                                int /*sourceStepX*/, int /*sourceStepY*/, std::shared_ptr<Band> /*destBand*/, int /*destOffsetX*/,
                                int /*destOffsetY*/, int /*destWidth*/, int /*destHeight*/,
                                const std::shared_ptr<ProductData>& /*dest_buffer*/,
                                std::shared_ptr<ceres::IProgressMonitor> /*pm*/) {};
private:
};
}
