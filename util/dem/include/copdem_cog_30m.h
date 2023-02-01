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

#include <future>
#include <string>
#include <vector>

#include "dataset.h"
#include "dem_aggregation.h"
#include "dem_property.h"
#include "pointer_holders.h"

namespace alus::dem {

class CopDemCog30m : public Aggregation {
public:
    CopDemCog30m() = delete;
    CopDemCog30m(std::vector<std::string> filenames);

    void LoadTiles() override;
    size_t GetTileCount() override;
    const PointerHolder* GetBuffers() override;
    const Property* GetProperties() override;
    const std::vector<Property>& GetPropertiesValue() override;
    void TransferToDevice() override;
    void ReleaseFromDevice() override;

    ~CopDemCog30m();

private:

    void LoadTilesThread();
    void WaitLoadTilesAndCheckErrors();

    std::vector<std::string> filenames_;
    std::vector<Dataset<float>> datasets_;

    std::vector<float*> device_formated_buffers_;
    PointerHolder* device_formated_buffers_table_{nullptr};
    std::vector<dem::Property> host_dem_properties_;
    dem::Property* device_dem_properties_{nullptr};

    std::future<void> load_tiles_future_;
};

}  // namespace alus::dem