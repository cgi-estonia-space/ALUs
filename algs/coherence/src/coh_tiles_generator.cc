#include "coh_tiles_generator.h"

namespace alus {

CohTilesGenerator::CohTilesGenerator(
    int band_x_size, int band_y_size, int tile_x_size, int tile_y_size, int coh_win_rg, int coh_win_az)
    : band_x_size_{band_x_size},
      band_y_size_{band_y_size},
      tile_x_size_{tile_x_size},
      tile_y_size_{tile_y_size},
      coh_win_rg_{coh_win_rg},
      coh_win_az_{coh_win_az} {}

// pre-generate tile attributes
std::vector<CohTile> CohTilesGenerator::GenerateTiles() const {
    int coh_x = GetCohWinDim(this->coh_win_rg_);
    int coh_y = GetCohWinDim(this->coh_win_az_);

    if (2 * coh_y > this->tile_y_size_ || 2 * coh_x > this->tile_x_size_) {
        throw std::invalid_argument("GenerateTiles: coherence window proportions to tile proportions are not logical");
    }

    int x_tiles = GetNumberOfTilesDim(this->band_x_size_, this->tile_x_size_);
    int y_tiles = GetNumberOfTilesDim(this->band_y_size_, this->tile_y_size_);

    std::vector<CohTile> tiles;
    //    todo::investigate
    tiles.reserve(static_cast<unsigned long>(y_tiles * x_tiles));

    int y_min_pad, y_max_pad, x_min_pad, x_max_pad;
    int data_y_max, data_x_max, data_x_min, data_y_min;
    int tile_x_size_out, tile_y_size_out;

    for (auto tile_y = 0; tile_y < y_tiles; tile_y++) {
        if (this->tile_y_size_ >= this->band_y_size_) {
            y_min_pad = coh_y;
            y_max_pad = coh_y;
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
                y_max_pad = coh_y;
                data_y_max = band_y_size_ - 1;
            } else if (tile_y == 0) {
                data_y_max = ((data_y_min + tile_y_size_) - 1) + coh_y;
                y_max_pad = 0;
            } else {
                y_max_pad = 0;
                data_y_max = ((data_y_min + tile_y_size_) - 1) + 2 * coh_y;
            }
        }
        for (auto tile_x = 0; tile_x < x_tiles; tile_x++) {
            if (tile_x_size_ >= band_x_size_) {
                x_min_pad = coh_x;
                x_max_pad = coh_x;
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
                    x_max_pad = coh_x;
                    data_x_max = band_x_size_ - 1;
                } else if (tile_x == 0) {
                    data_x_max = ((data_x_min + tile_x_size_) - 1) + coh_x;
                    x_max_pad = 0;
                } else {
                    x_max_pad = 0;
                    data_x_max = ((data_x_min + tile_x_size_) - 1) + 2 * coh_x;
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

            tiles.emplace_back(tile_x,
                               tile_y,
                               Tile{data_x_max, data_y_max, data_x_min, data_y_min},
                               Tile{tile_x_size_out - 1 + (tile_x_size_ * tile_x),
                                    tile_y_size_out - 1 + (tile_y_size_ * tile_y),
                                    tile_x_size_ * tile_x,
                                    tile_y_size_ * tile_y},
                               y_min_pad,
                               y_max_pad,
                               x_min_pad,
                               x_max_pad);
        }
    }

    return tiles;
}

short CohTilesGenerator::GetCohWinDim(int coh_win_dim) {
    return static_cast<short>((static_cast<float>(coh_win_dim) - 1.f) / 2.f);
}
int CohTilesGenerator::GetNumberOfTilesDim(int band_size_dim, int tile_size_dim) {
    return static_cast<short>(ceil(static_cast<float>(band_size_dim) / static_cast<float>(tile_size_dim)));
}
std::vector<CohTile> CohTilesGenerator::GetTiles() const { return this->GenerateTiles(); }

}  // namespace alus