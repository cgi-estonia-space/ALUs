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
public:
    int burst_index_;

public:
    CohTile() = default;
    CohTile(int tile_x, int tile_y, const Tile& tile_in, const Tile& tile_out, int y_min_pad, int y_max_pad,
            int x_min_pad, int x_max_pad, int burst_index);
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
    ~CohTile() = default;

    std::string str()
    {
        char buf[1000];
        sprintf(buf, "tile in = %d %d %d %d, n br = %d\n", tile_in_.GetXMin(), tile_in_.GetXMax(), tile_in_.GetYMin(), tile_in_.GetYMax(), burst_index_);

        char buf2[1000];
        sprintf(buf2, "tile out = %d %d %d %d\n", tile_out_.GetXMin(), tile_out_.GetXMax(), tile_out_.GetYMin(), tile_out_.GetYMax());
        return std::string(buf) + buf2;
    }
};
}  // namespace coherence-cuda
}  // namespace alus