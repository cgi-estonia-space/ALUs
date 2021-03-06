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

#include <string_view>
#include <vector>

#include "i_data_tile_read_write_base.h"
#include "tile.h"

/**
 * reads I tiles from tiles
 */
namespace alus {
class IDataTileReader : public IDataTileReadWriteBase {
public:
    IDataTileReader() = delete;
    // todo:check if this works like expected
    IDataTileReader(std::string_view file_name, const std::vector<int>& band_map, int band_count)
        : IDataTileReadWriteBase(file_name, band_map, band_count) {}
    virtual ~IDataTileReader() = default;
    virtual void ReadTile(const Tile& tile) = 0;
    virtual void ReadTile(const Tile& tile, float* data) = 0;
    [[nodiscard]] virtual const std::vector<float>& GetData() const = 0;
    [[nodiscard]] virtual double GetValueAtXy(int x, int y) const = 0;
};
}  // namespace alus