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
#include "coh_tile.h"

namespace alus {
namespace coherence_cuda {

CohTile::CohTile(int tile_x, int tile_y, const Tile& tile_in, const Tile& tile_out, int y_min_pad, int y_max_pad,
                 int x_min_pad, int x_max_pad, int burst_index)
    : IoTile(tile_x, tile_y, tile_in, tile_out),
      y_min_pad_{y_min_pad},
      y_max_pad_{y_max_pad},
      x_min_pad_{x_min_pad},
      x_max_pad_{x_max_pad},
      burst_index_{burst_index}
{}

int CohTile::GetYMinPad() const { return y_min_pad_; }
int CohTile::GetYMaxPad() const { return y_max_pad_; }
int CohTile::GetXMinPad() const { return x_min_pad_; }
int CohTile::GetXMaxPad() const { return x_max_pad_; }

int CohTile::GetWw() const {
    return this->tile_in_.GetXMax() + this->GetXMinPad() + this->GetXMaxPad() - this->tile_in_.GetXMin() + 1;
}
int CohTile::GetHh() const {
    return this->tile_in_.GetYMax() + this->GetYMinPad() + this->GetYMaxPad() - this->tile_in_.GetYMin() + 1;
}

// calc also uses paddings (actual calculation area min max vs data take from gdal Min max)
int CohTile::GetCalcXMin() const { return this->tile_in_.GetXMin(); }
int CohTile::GetCalcXMax() const { return this->tile_in_.GetXMax() + this->GetXMinPad() + this->GetXMaxPad(); }
int CohTile::GetCalcYMin() const { return this->tile_in_.GetYMin(); }
int CohTile::GetCalcYMax() const { return this->tile_in_.GetYMax() + this->GetYMinPad() + this->GetYMaxPad(); }

}  // namespace coherence-cuda
}  // namespace alus