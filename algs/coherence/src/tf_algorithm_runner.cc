#include <tf_algorithm_runner.h>

namespace alus {

void TFAlgorithmRunner::Run() {
    // todo:resolve parameters in a better way here (maybe add user provided options during construction?)
    auto tiles = this->tile_provider_->GetTiles();
    this->algo_->PreTileCalc(*this->scope_);
    for (auto tile : tiles) {
        this->tile_reader_->ReadTile(tile.GetTileIn());
        // todo:move to tensors
        this->algo_->DataToTensors(tile.GetTileIn(), *this->tile_reader_);
        // to prepare variables for actual tile manipulations
        std::vector<tensorflow::Output> algo_graph_tile{this->algo_->TileCalc(*this->scope_, tile)};
        tensorflow::ClientSession::FeedType &inputs = this->algo_->GetInputs();
        std::vector<tensorflow::Tensor> outputs_101;
        TF_CHECK_OK(this->session_->Run(inputs,
                                        // todo:do both here using abstract function
                                        algo_graph_tile,
                                        &outputs_101));
        this->tile_writer_->WriteTile(tile.GetTileOut(), outputs_101[0].data());
    }
}

TFAlgorithmRunner::TFAlgorithmRunner(IDataTileReader *tile_reader,
                                     IDataTileWriter *tile_writer,
                                     ITileProvider *tile_provider,
                                     IAlgo *algorithm,
                                     tensorflow::ClientSession *session,
                                     tensorflow::Scope *scope)
    : tile_reader_{tile_reader},
      tile_writer_{tile_writer},
      tile_provider_{tile_provider},
      algo_{algorithm},
      session_{session},
      scope_{scope} {}

}  // namespace alus