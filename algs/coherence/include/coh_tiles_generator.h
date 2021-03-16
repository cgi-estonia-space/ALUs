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

#include <cmath>
#include <stdexcept>
#include <vector>

#include "coh_tile.h"
#include "i_tile_provider.h"

namespace alus {

class CohTilesGenerator : virtual public ITileProvider {
private:
    int band_x_size_, band_y_size_, tile_x_size_, tile_y_size_, coh_win_rg_, coh_win_az_;

    [[nodiscard]] static int GetNumberOfTilesDim(int band_size_dim, int tile_size_dim);
    [[nodiscard]] static short GetCohWinDim(int coh_win_dim);
    [[nodiscard]] std::vector<CohTile> GenerateTiles() const;

public:
    CohTilesGenerator(int band_x_size, int band_y_size, int tile_x_size, int tile_y_size, int coh_win_rg,
                      int coh_win_az);
    [[nodiscard]] std::vector<CohTile> GetTiles() const override;
};
}  // namespace alus