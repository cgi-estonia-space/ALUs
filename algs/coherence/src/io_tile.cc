#include "io_tile.h"

namespace alus {

IoTile::IoTile(int tile_x, int tile_y, const Tile &tile_in, const Tile &tile_out)
    : tile_x_{tile_x}, tile_y_{tile_y}, tile_in_{tile_in}, tile_out_{tile_out} {}
const Tile &IoTile::GetTileIn() const { return tile_in_; }
const Tile &IoTile::GetTileOut() const { return tile_out_; }

}  // namespace alus