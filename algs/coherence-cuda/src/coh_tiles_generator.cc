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
#include "coh_tiles_generator.h"

#include <cmath>

namespace alus {
namespace coherence_cuda {
CohTilesGenerator::CohTilesGenerator(int band_x_size, int band_y_size, int tile_x_size, int tile_y_size, int coh_win_rg,
                                     int coh_win_az)
    : band_x_size_{band_x_size},
      band_y_size_{band_y_size},
      tile_x_size_{tile_x_size},
      tile_y_size_{tile_y_size},
      coh_win_rg_{coh_win_rg},
      coh_win_az_{coh_win_az} {}

// pre-generate tile attributes
std::vector<CohTile> CohTilesGenerator::GenerateTiles() const {
    int coh_x = GetCohWinDim(coh_win_rg_);
    int coh_y = GetCohWinDim(coh_win_az_);

    // quick fix to use asymmetric coherence window (adding asymetry to right and bottom side)
    auto x_asymmetry = static_cast<int>(coh_win_rg_ % 2 == 0);
    auto y_asymmetry = static_cast<int>(coh_win_az_ % 2 == 0);

    if (2 * coh_y > tile_y_size_ || 2 * coh_x > tile_x_size_) {
        throw std::invalid_argument("GenerateTiles: coherence window proportions to tile proportions are not logical");
    }

    int x_tiles = GetNumberOfTilesDim(band_x_size_, tile_x_size_);
    int y_tiles = GetNumberOfTilesDim(band_y_size_, tile_y_size_);
    std::vector<CohTile> tiles;
    //    todo::investigate
    tiles.reserve(static_cast<unsigned long>(y_tiles * x_tiles));

    int y_min_pad, y_max_pad, x_min_pad, x_max_pad;
    int data_y_max, data_x_max, data_x_min, data_y_min;
    int tile_x_size_out, tile_y_size_out;

    for (auto tile_y = 0; tile_y < y_tiles; tile_y++) {
        if (tile_y_size_ >= band_y_size_) {
            y_min_pad = coh_y;
            y_max_pad = coh_y + y_asymmetry;
            data_y_min = 0;
            data_y_max = band_y_size_ - 1;
        } else {
            if (tile_y == 0) {
                y_min_pad = coh_y;
                data_y_min = 0;
                // we just need to know size for data from gdal
            } else {
                y_min_pad = 0;
                data_y_min = tile_y_size_ * tile_y - coh_y;
            }

            if (tile_y == y_tiles - 1) {
                y_max_pad = coh_y + y_asymmetry;
                data_y_max = band_y_size_ - 1;
            } else if (tile_y == 0) {
                data_y_max = ((data_y_min + tile_y_size_) - 1) + coh_y + y_asymmetry;
                y_max_pad = 0;
            } else {
                y_max_pad = 0;
                data_y_max = ((data_y_min + tile_y_size_) - 1) + 2 * coh_y + y_asymmetry;
            }
        }
        for (auto tile_x = 0; tile_x < x_tiles; tile_x++) {
            if (tile_x_size_ >= band_x_size_) {
                x_min_pad = coh_x;
                x_max_pad = coh_x + x_asymmetry;
                data_x_min = 0;
                data_x_max = band_x_size_ - 1;
            } else {
                if (tile_x == 0) {
                    x_min_pad = coh_x;
                    data_x_min = 0;
                } else {
                    x_min_pad = 0;
                    data_x_min = tile_x_size_ * tile_x - coh_x;
                }

                if (tile_x == x_tiles - 1) {
                    x_max_pad = coh_x + x_asymmetry;
                    data_x_max = band_x_size_ - 1;
                } else if (tile_x == 0) {
                    data_x_max = ((data_x_min + tile_x_size_) - 1) + coh_x + x_asymmetry;
                    x_max_pad = 0;
                } else {
                    x_max_pad = 0;
                    data_x_max = ((data_x_min + tile_x_size_) - 1) + 2 * coh_x + x_asymmetry;
                }
            }

            // last tile might be shorter
            if (tile_x == x_tiles - 1) {
                tile_x_size_out = band_x_size_ - tile_x_size_ * tile_x;
            } else {
                tile_x_size_out = tile_x_size_;
            }
            if (tile_y == y_tiles - 1) {
                tile_y_size_out = band_y_size_ - tile_y_size_ * tile_y;
            } else {
                tile_y_size_out = tile_y_size_;
            }

            tiles.emplace_back(
                tile_x, tile_y, Tile{data_x_max, data_y_max, data_x_min, data_y_min},
                Tile{tile_x_size_out - 1 + (tile_x_size_ * tile_x), tile_y_size_out - 1 + (tile_y_size_ * tile_y),
                     tile_x_size_ * tile_x, tile_y_size_ * tile_y},
                y_min_pad, y_max_pad, x_min_pad, x_max_pad);
        }
    }

    return tiles;
}

short CohTilesGenerator::GetCohWinDim(int coh_win_dim) {
    return static_cast<short>((static_cast<float>(coh_win_dim) - 1.f) / 2.f);
}

int CohTilesGenerator::GetNumberOfTilesDim(int band_size_dim, int tile_size_dim) {
    return static_cast<short>(std::ceil(static_cast<float>(band_size_dim) / static_cast<float>(tile_size_dim)));
}

std::vector<CohTile> CohTilesGenerator::GetTiles() const { return this->GenerateTiles(); }

}  // namespace coherence-cuda
}  // namespace alus