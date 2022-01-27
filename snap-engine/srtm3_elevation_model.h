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

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "cuda_friendly_object.h"
#include "dataset.h"
#include "pointer_holders.h"
#include "shapes.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model_constants.h"
#include "srtm3_format_computation.h"

namespace alus::snapengine {

/*
    Majority of this class is from SRTM3GeoTiffElevationModel.java and BaseElevationModel.java.
    But this also incorporates parts from SRTM3GeoTiffTile.java, SRTM3GeoTiffFile.java and their base classes,
    as it made sense to pile all of this together due to having to be available from the gpu.

    This class is given the tile coordinates using a map and then it finds those tiles, formats them and
    makes them available on the gpu.
*/
class Srtm3ElevationModel : public cuda::CudaFriendlyObject {
private:
    std::vector<std::string> file_names_;
    std::vector<Dataset<float>> srtms_;
    // use this for cudaFree once you are done with image.
    std::vector<float*> device_formated_srtm_buffers_;
    PointerHolder* device_formated_srtm_buffers_info_{nullptr};
    size_t device_srtm3_tiles_count_;
    std::vector<Srtm3FormatComputation> srtm_format_info_;

    bool is_inited_{false};
    bool is_on_device_{false};
    std::mutex init_mutex_;
    std::mutex info_mutex_;
    std::mutex buffer_mutex_;
    std::condition_variable init_var_;
    std::condition_variable copy_var_;

    std::thread init_thread_;
    std::thread copy_thread_;

    std::shared_ptr<EarthGravitationalModel96> egm_96_;
    std::exception_ptr elevation_exception_{nullptr};

    void ReadSrtmTilesThread();
    void HostToDeviceThread();

public:
    explicit Srtm3ElevationModel(std::vector<std::string> file_names);
    virtual ~Srtm3ElevationModel();

    void ReadSrtmTiles(std::shared_ptr<EarthGravitationalModel96>& egm_96);
    PointerHolder* GetSrtmBuffersInfo();
    size_t GetDeviceSrtm3TilesCount();

    void HostToDevice() override;
    void DeviceToHost() override;
    void DeviceFree() override;

    constexpr static int GetTileWidthInDegrees() { return srtm3elevationmodel::DEGREE_RES; }
    constexpr static int GetTileWidth() { return srtm3elevationmodel::TILE_WIDTH_PIXELS; }

    Srtm3ElevationModel(const Srtm3ElevationModel&) = delete;  // class does not support copying(and moving)
    Srtm3ElevationModel& operator=(const Srtm3ElevationModel&) = delete;
};

}  // namespace alus::snapengine