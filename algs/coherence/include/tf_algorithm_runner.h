#pragma once

#include <vector>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>

#include "i_algo.h"
#include "i_data_tile_reader.h"
#include "i_data_tile_writer.h"
#include "i_tile_provider.h"

namespace alus {
class TFAlgorithmRunner {
   private:
    IDataTileReader *tile_reader_;
    IDataTileWriter *tile_writer_;
    ITileProvider *tile_provider_;
    IAlgo *algo_;
    tensorflow::ClientSession *session_;
    tensorflow::Scope *scope_;

   public:
    TFAlgorithmRunner() = delete;
    TFAlgorithmRunner(IDataTileReader *tile_reader,
                      IDataTileWriter *tile_writer,
                      ITileProvider *tile_provider,
                      IAlgo *algorithm,
                      tensorflow::ClientSession *session,
                      tensorflow::Scope *scope);
    void Run();
};
}  // namespace alus