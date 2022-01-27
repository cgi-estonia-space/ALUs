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

#include <memory>

#include "snap-gpf/gpf/i_tile.h"

namespace alus::snapengine::custom {
class IoTile {
private:
    // tile location In cartesian system tile_x_ tile_y_
    int tile_x_{};  // NOLINT
    int tile_y_{};  // NOLINT
    // tile read from source using provided reader
    std::shared_ptr<snapengine::ITile> tile_in_;
    // tile written out using provided writer
    std::shared_ptr<snapengine::ITile> tile_out_;

public:
    IoTile(int tile_x, int tile_y, std::shared_ptr<snapengine::ITile> tile_in,
           std::shared_ptr<snapengine::ITile> tile_out);
    [[nodiscard]] const std::shared_ptr<snapengine::ITile>& GetTileIn() const;
    [[nodiscard]] const std::shared_ptr<snapengine::ITile>& GetTileOut() const;
    ~IoTile() = default;
};
}  // namespace alus::snapengine::custom