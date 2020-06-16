#pragma once

#include "io_tile.h"
#include "tile.h"

namespace alus {

class CohTile : virtual public IoTile {
   private:
    // padding is added for the edge tiles, and also helps to calculate overlapping data regions
    int y_min_pad_{}, y_max_pad_{}, x_min_pad_{}, x_max_pad_{};

   public:
    CohTile(int tile_x,
            int tile_y,
            const Tile &tile_in,
            const Tile &tile_out,
            int y_min_pad,
            int y_max_pad,
            int x_min_pad,
            int x_max_pad);
    [[nodiscard]] int GetYMinPad() const;
    [[nodiscard]] int GetYMaxPad() const;
    [[nodiscard]] int GetXMinPad() const;
    [[nodiscard]] int GetXMaxPad() const;
    [[nodiscard]] int GetWw() const;
    [[nodiscard]] int GetHh() const;
    [[nodiscard]] int GetCalcXMin() const;
    [[nodiscard]] int GetCalcXMax() const;
    [[nodiscard]] int GetCalcYMin() const;
    [[nodiscard]] int GetCalcYMax() const;
    ~CohTile() = default;
};

}  // namespace alus