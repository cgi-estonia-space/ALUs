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

#include "io_tile.h"
#include "tile.h"

#include <string>

namespace alus {
namespace coherence_cuda {

class CohTile : public IoTile {
private:
    // padding is added for the edge tiles, and also helps to calculate overlapping data regions
    int y_min_pad_{}, y_max_pad_{}, x_min_pad_{}, x_max_pad_{};
    int burst_index_;
    int burst_size_;

public:
    CohTile() = default;
    CohTile(int tile_x, int tile_y, const Tile& tile_in, const Tile& tile_out, int y_min_pad, int y_max_pad,
            int x_min_pad, int x_max_pad, int burst_index, int burst_size)
        : IoTile(tile_x, tile_y, tile_in, tile_out),
          y_min_pad_{y_min_pad},
          y_max_pad_{y_max_pad},
          x_min_pad_{x_min_pad},
          x_max_pad_{x_max_pad},
          burst_index_{burst_index},
          burst_size_{burst_size} {}

    [[nodiscard]] int GetYMinPad() const;
    [[nodiscard]] int GetYMaxPad() const;
    [[nodiscard]] int GetXMinPad() const;
    [[nodiscard]] int GetXMaxPad() const;
    [[nodiscard]] int GetWw() const;
    [[nodiscard]] int GetHh() const;
    [[nodiscard]] int GetCalcXMin() const;
    [[nodiscard]] int GetCalcXMax() const;
    [[nodiscard]] int GetCalcYMin() const;
    [[nodiscard]] int GetCalcYMax() const;
    [[nodiscard]] int GetBurstIndex() const { return burst_index_; }
    [[nodiscard]] int GetBurstSize() const { return burst_size_; }
    ~CohTile() = default;

    [[nodiscard]] std::string Str() const {
        char format_buf[120];
        snprintf(format_buf, std::size(format_buf), "tile in = %d %d %d %d, n br = %d | ", tile_in_.GetXMin(),
                 tile_in_.GetXMax(), tile_in_.GetYMin(), tile_in_.GetYMax(), burst_index_);
        std::string s(format_buf);
        snprintf(format_buf, std::size(format_buf), "tile out = %d %d %d %d", tile_out_.GetXMin(), tile_out_.GetXMax(),
                 tile_out_.GetYMin(), tile_out_.GetYMax());
        return s + format_buf;
    }
};
}  // namespace coherence_cuda
}  // namespace alus