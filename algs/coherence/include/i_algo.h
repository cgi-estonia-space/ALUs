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

#include "coh_tile.h"
#include "i_data_tile_reader.h"

namespace alus {
class IAlgo {
protected:
    tensorflow::ClientSession::FeedType inputs_;

public:
    // todo:CohTile needs replacement with more general specification
    virtual tensorflow::Output TileCalc(const tensorflow::Scope& scope, CohTile& tile) = 0;
    virtual void PreTileCalc(const tensorflow::Scope& scope) = 0;
    virtual void DataToTensors(const Tile& tile, const IDataTileReader& reader) = 0;
    const tensorflow::ClientSession::FeedType& GetInputs() const { return inputs_; }
    IAlgo() = default;
    IAlgo(const IAlgo&) = delete;
    IAlgo& operator=(const IAlgo&) = delete;
    virtual ~IAlgo() = default;
};
}  // namespace alus