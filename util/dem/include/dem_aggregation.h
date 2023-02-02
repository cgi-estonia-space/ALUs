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

#include <cstddef>
#include <vector>

#include "dem_property.h"
#include "dem_type.h"
#include "pointer_holders.h"

namespace alus::dem {

class Aggregation {
public:

    //   virtual Type GetType() const = 0;
    virtual void LoadTiles() = 0;
    virtual size_t GetTileCount() = 0;
    virtual const PointerHolder* GetBuffers() = 0;
    virtual const Property* GetProperties() = 0;
    virtual const std::vector<Property>& GetPropertiesValue() = 0;
    virtual void TransferToDevice() = 0;
    virtual void ReleaseFromDevice() = 0;
    virtual ~Aggregation() = default;
};

}