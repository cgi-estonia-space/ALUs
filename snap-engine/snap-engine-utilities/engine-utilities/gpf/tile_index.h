/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.ReaderUtils.java
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

#include <memory>

#include "snap-gpf/gpf/i_tile.h"

namespace alus::snapengine {

/**
 * calculates the index into a tile
 */
class TileIndex {
private:
    int tile_offset_;
    int tile_stride_;
    int tile_min_x_;
    int tile_min_y_;

    int offset_{0};

public:
    explicit TileIndex(const std::shared_ptr<ITile>& tile) {
        tile_offset_ = tile->GetScanlineOffset();
        tile_stride_ = tile->GetScanlineStride();
        tile_min_x_ = tile->GetMinX();
        tile_min_y_ = tile->GetMinY();
    }
    explicit TileIndex(const ITile* tile)
        : tile_offset_(tile->GetScanlineOffset()),
          tile_stride_(tile->GetScanlineStride()),
          tile_min_x_(tile->GetMinX()),
          tile_min_y_(tile->GetMinY()) {}

    /**
     * calculates offset
     *
     * @param ty y pos
     * @return offset
     */
    inline int CalculateStride(int ty) {
        offset_ = tile_min_x_ - (((ty - tile_min_y_) * tile_stride_) + tile_offset_);
        return offset_;
    }

    [[nodiscard]] inline int GetOffset() const { return offset_; }

    [[nodiscard]] inline int GetIndex(int tx) const { return tx - offset_; }
};

}  // namespace alus::snapengine
