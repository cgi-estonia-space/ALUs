#pragma once

#include <tensorflow/cc/client/client_session.h>

#include "coh_tile.h"
#include "i_data_tile_reader.h"

namespace alus {
class IAlgo {
   protected:
    tensorflow::ClientSession::FeedType inputs_;

   public:
    // todo:CohTile needs replacement with more general specification
    virtual tensorflow::Output TileCalc(tensorflow::Scope &scope, CohTile &tile) = 0;
    virtual void PreTileCalc(tensorflow::Scope &scope) = 0;
    virtual void DataToTensors(const Tile &tile, const IDataTileReader &reader) = 0;
    [[nodiscard]] virtual tensorflow::ClientSession::FeedType &GetInputs() = 0;
    virtual ~IAlgo() = default;
};
}  // namespace alus