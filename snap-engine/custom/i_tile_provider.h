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

#include <memory>
#include <vector>

#include "custom/io_tile.h"
#include "snap-gpf/gpf/i_tile.h"

namespace alus {
namespace snapengine {
namespace custom {
class ITileProvider {
public:
    [[nodiscard]] virtual std::vector<std::shared_ptr<IoTile>> GetTiles() const = 0;
    virtual ~ITileProvider() = default;
};
}  // namespace custom
}  // namespace snapengine
}  // namespace alus