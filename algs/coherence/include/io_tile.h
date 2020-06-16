#pragma once

#include "tile.h"

namespace alus {

class IoTile {
   protected:
    // tile location In cartesian system tile_x_ tile_y_
    int tile_x_{};
    int tile_y_{};
    // tile read from source using provided reader
    Tile tile_in_;
    // tile written out using provided writer
    Tile tile_out_;

   public:
    IoTile(int tile_x, int tile_y, const Tile &tile_in, const Tile &tile_out);
    [[nodiscard]] const Tile &GetTileIn() const;
    [[nodiscard]] const Tile &GetTileOut() const;
    ~IoTile() = default;
};

}  // namespace alus
