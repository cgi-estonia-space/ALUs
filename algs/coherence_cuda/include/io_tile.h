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

#include "tile.h"

namespace alus {
namespace coherence_cuda {
class IoTile {
protected:
    // tile location In cartesian system tile_x_ tile_y_
    int tile_x_{};
    int tile_y_{};
    // tile read from source using provided reader
    Tile tile_in_;
    // tile written out using provided writer
    Tile tile_out_;

public:
    IoTile() = default;
    IoTile(int tile_x, int tile_y, const Tile& tile_in, const Tile& tile_out);
    [[nodiscard]] const Tile& GetTileIn() const;
    [[nodiscard]] const Tile& GetTileOut() const;
    ~IoTile() = default;
};

}  // namespace coherence_cuda
}  // namespace alus
