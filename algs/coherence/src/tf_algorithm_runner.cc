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
#include "tf_algorithm_runner.h"

#include <iostream>
#include <vector>

namespace alus {

void TFAlgorithmRunner::Run() {
    // todo:resolve parameters in a better way here (maybe add user provided options during construction?)
    auto tiles = tile_provider_->GetTiles();
    algo_->PreTileCalc(*scope_);
    size_t i{0};
    const size_t tiles_to_process{tiles.size()};
    for (auto tile : tiles) {
        tile_reader_->ReadTile(tile.GetTileIn());
        // todo:move to tensors
        algo_->DataToTensors(tile.GetTileIn(), *tile_reader_);
        std::vector<tensorflow::Tensor> outputs_101;
        TF_CHECK_OK(session_->Run(algo_->GetInputs(),
                                  // todo:do both here using abstract function
                                  std::vector<tensorflow::Output>{algo_->TileCalc(*scope_, tile)}, &outputs_101));
        tile_writer_->WriteTile(tile.GetTileOut(), static_cast<float*>(outputs_101.at(0).data()), outputs_101.size());
        ++i;
        std::cout << '\r' << "Tile " << i << "/" << tiles.size() << " processed" << std::endl;
        if (i >= tiles_to_process) {
            break;
        }
    }
}

TFAlgorithmRunner::TFAlgorithmRunner(IDataTileReader* tile_reader, IDataTileWriter* tile_writer,
                                     ITileProvider* tile_provider, IAlgo* algorithm, tensorflow::ClientSession* session,
                                     tensorflow::Scope* scope)
    : tile_reader_{tile_reader},
      tile_writer_{tile_writer},
      tile_provider_{tile_provider},
      algo_{algorithm},
      session_{session},
      scope_{scope} {}

}  // namespace alus