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
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "dataset.h"
#include "dem_management.h"
#include "dem_property.h"
#include "pointer_holders.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"

namespace alus::dem {

class CopDemCog30m : public Management {
public:
    CopDemCog30m() = delete;
    CopDemCog30m(std::vector<std::string> filenames);
    CopDemCog30m(std::vector<std::string> filenames, std::shared_ptr<snapengine::EarthGravitationalModel96> egm96);

    void LoadTiles() override;
    size_t GetTileCount() override;
    const PointerHolder* GetBuffers() override;
    const Property* GetProperties() override;
    const std::vector<Property>& GetPropertiesValue() override;
    void TransferToDevice() override;
    void ReleaseFromDevice() noexcept override;

    static int ComputeId(double lon_origin, double lat_origin);

    ~CopDemCog30m();

private:
    void LoadTilesImpl();
    void TransferToDeviceImpl();
    static void WaitFutureAndCheckErrorsDefault(std::future<void>& f, size_t tile_count, std::string_view ex_msg);
    void WaitLoadTilesAndCheckErrors();
    void WaitTransferDeviceAndCheckErrors();
    void VerifyProperties(const Property& prop, const Dataset<float>& ds, std::string_view filename);

    std::vector<std::string> filenames_;
    std::vector<Dataset<float>> datasets_;

    std::vector<float*> device_formated_buffers_{};
    PointerHolder* device_formated_buffers_table_{nullptr};
    std::vector<Property> host_dem_properties_{};
    dem::Property* device_dem_properties_{nullptr};
    std::shared_ptr<snapengine::EarthGravitationalModel96> egm96_;

    std::future<void> load_tiles_future_;
    std::future<void> transfer_to_device_future_;
};

}  // namespace alus::dem