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

#include <vector>

#include "coh_tile.h"
#include "i_data_tile_reader.h"

namespace alus {
namespace coherence_cuda {
class IAlgoCuda {
public:
    IAlgoCuda() = default;
    IAlgoCuda(const IAlgoCuda&) = delete;
    IAlgoCuda& operator=(const IAlgoCuda&) = delete;
    virtual ~IAlgoCuda() = default;
    // todo: probably needs T instead of float (starting simple) and need some more generic tile
    virtual void TileCalc(CohTile& tile, const std::vector<float>& data, std::vector<float>& data_out) = 0;
    virtual void PreTileCalc() = 0;
    virtual void Cleanup() = 0;
};
}  // namespace coherence_cuda
}  // namespace alus