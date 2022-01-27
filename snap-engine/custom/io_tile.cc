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
#include "io_tile.h"

#include <memory>
#include <utility>

#include "snap-gpf/gpf/i_tile.h"

namespace alus::snapengine::custom {

IoTile::IoTile(int tile_x, int tile_y, std::shared_ptr<snapengine::ITile> tile_in,
               std::shared_ptr<snapengine::ITile> tile_out)
    : tile_x_{tile_x}, tile_y_{tile_y}, tile_in_{std::move(tile_in)}, tile_out_{std::move(tile_out)} {}
const std::shared_ptr<snapengine::ITile>& IoTile::GetTileIn() const { return tile_in_; }
const std::shared_ptr<snapengine::ITile>& IoTile::GetTileOut() const { return tile_out_; }

}  // namespace alus::snapengine::custom