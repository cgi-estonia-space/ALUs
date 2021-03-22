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

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/framework/tensor.h>

#include "i_algo.h"
#include "i_data_tile_reader.h"
#include "i_data_tile_writer.h"
#include "i_tile_provider.h"

namespace alus {
class TFAlgorithmRunner {
private:
    IDataTileReader* tile_reader_;
    IDataTileWriter* tile_writer_;
    ITileProvider* tile_provider_;
    IAlgo* algo_;
    tensorflow::ClientSession* session_;
    tensorflow::Scope* scope_;

public:
    TFAlgorithmRunner() = delete;
    TFAlgorithmRunner(IDataTileReader* tile_reader, IDataTileWriter* tile_writer, ITileProvider* tile_provider,
                      IAlgo* algorithm, tensorflow::ClientSession* session, tensorflow::Scope* scope);
    void Run();
};
}  // namespace alus