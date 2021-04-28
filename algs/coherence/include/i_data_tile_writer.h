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
#include <utility>
#include <vector>

#include "band_params.h"
#include "i_data_tile_read_write_base.h"
#include "tile.h"

namespace alus {
class IDataTileWriter : public IDataTileReadWriteBase {
public:
    IDataTileWriter() = delete;
    IDataTileWriter(std::string_view file_name, const BandParams& band_params,
                    const std::vector<double>& affine_geo_transform_out, std::string_view data_projection_out)
        : IDataTileReadWriteBase(file_name, band_params, data_projection_out, affine_geo_transform_out) {}
    virtual void WriteTile(const Tile& tile, float* tile_data, std::size_t tile_data_size) = 0;
    IDataTileWriter(const IDataTileWriter&) = delete;
    IDataTileWriter& operator=(const IDataTileWriter&) = delete;
    virtual ~IDataTileWriter() = default;
};
}  // namespace alus