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

#include <custom/rectangle.h>
#include <memory>

#include "../i_tile.h"
#include "shapes.h"
#include "snap-core/core/datamodel/product_data.h"
#include "snap-core/core/datamodel/raster_data_node.h"

namespace alus::snapengine {

class TileImpl : public ITile {
private:
    int min_x_;
    int min_y_;
    int max_x_;
    int max_y_;
    int width_;
    int height_;
    int scanline_offset_{0};
    int scanline_stride_;
    // this instance should be created using reader?
    std::shared_ptr<snapengine::ProductData> data_buffer_;  // NOLINT
    std::vector<float> simple_data_buffer_;
    const std::shared_ptr<RasterDataNode>& raster_data_node_;  // NOLINT

public:
    inline TileImpl(const std::shared_ptr<RasterDataNode>& raster_data_node, const custom::Rectangle& rectangle)
        : min_x_{rectangle.x},
          min_y_{rectangle.y},
          max_x_{rectangle.x + rectangle.width - 1},
          max_y_{rectangle.y + rectangle.height - 1},
          width_{rectangle.width},
          height_{rectangle.height},
          raster_data_node_{raster_data_node} {
        [[maybe_unused]] int sm_x0 =
            rectangle.x;  //- raster.getSampleModelTranslateX(); (currently not supporting sampling)
        [[maybe_unused]] int sm_y0 =
            rectangle.y;            //- raster.getSampleModelTranslateY(); (currently not supporting sampling)
        scanline_stride_ = width_;  // keeping it here in case we add more logic in the future (e.g subsampling)
    }
    inline TileImpl(const std::shared_ptr<RasterDataNode>& raster_data_node, const alus::Rectangle& rectangle)
        : min_x_{rectangle.x},
          min_y_{rectangle.y},
          max_x_{rectangle.x + rectangle.width - 1},
          max_y_{rectangle.y + rectangle.height - 1},
          width_{rectangle.width},
          height_{rectangle.height},
          raster_data_node_{raster_data_node} {
        [[maybe_unused]] int sm_x0 =
            rectangle.x;  //- raster.getSampleModelTranslateX(); (currently not supporting sampling)
        [[maybe_unused]] int sm_y0 =
            rectangle.y;            //- raster.getSampleModelTranslateY(); (currently not supporting sampling)
        scanline_stride_ = width_;  // keeping it here in case we add more logic in the future (e.g subsampling)
    }
    ~TileImpl() override = default;

    [[nodiscard]] inline int GetMinX() const override { return min_x_; }
    [[nodiscard]] inline int GetMaxX() const override { return max_x_; }
    [[nodiscard]] inline int GetMinY() const override { return min_y_; }
    [[nodiscard]] inline int GetMaxY() const override { return max_y_; }
    [[nodiscard]] inline int GetScanlineOffset() const override { return scanline_offset_; }
    [[nodiscard]] inline int GetScanlineStride() const override { return scanline_stride_; }

    inline std::vector<float>& GetSimpleDataBuffer() override {
        simple_data_buffer_.resize(width_ * height_);
        return simple_data_buffer_;
    }
    inline int GetDataBufferIndex(int x, int y) override {
        return scanline_offset_ + (x - min_x_) + (y - min_y_) * scanline_stride_;
    }
};

}  // namespace alus::snapengine