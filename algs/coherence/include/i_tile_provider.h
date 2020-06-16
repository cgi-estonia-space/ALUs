#pragma once

#include <vector>

#include "coh_tile.h"

namespace alus {
class ITileProvider {
   public:
    [[nodiscard]] virtual std::vector<CohTile> GetTiles() const = 0;
    virtual ~ITileProvider() = default;
};
}  // namespace alus
