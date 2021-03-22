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
#include <tf_algorithm_runner.h>

#include <iostream>
#include <vector>

namespace alus {

void TFAlgorithmRunner::Run() {
    // todo:resolve parameters in a better way here (maybe add user provided options during construction?)
    auto tiles = this->tile_provider_->GetTiles();
    this->algo_->PreTileCalc(*this->scope_);
    size_t i{0};
    const size_t TILES_TO_PROCESS{tiles.size()};
    for (auto tile : tiles) {
        this->tile_reader_->ReadTile(tile.GetTileIn());
        // todo:move to tensors
        this->algo_->DataToTensors(tile.GetTileIn(), *this->tile_reader_);
        // to prepare variables for actual tile manipulations
        std::vector<tensorflow::Output> algo_graph_tile{this->algo_->TileCalc(*this->scope_, tile)};
        tensorflow::ClientSession::FeedType& inputs = this->algo_->GetInputs();
        std::vector<tensorflow::Tensor> outputs_101;
        TF_CHECK_OK(this->session_->Run(inputs,
                                        // todo:do both here using abstract function
                                        algo_graph_tile, &outputs_101));
        this->tile_writer_->WriteTile(tile.GetTileOut(), outputs_101[0].data());

        ++i;
        std::cout << '\r' << "Tile " << i << "/" << tiles.size() << " processed" << std::endl;
        if (i >= TILES_TO_PROCESS) {
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