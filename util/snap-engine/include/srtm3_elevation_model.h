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

#include <string>
#include <vector>

#include "CudaFriendlyObject.h"
#include "dataset.h"
#include "earth_gravitational_model96.h"
#include "pointer_holders.h"
#include "srtm3_format_computation.h"

namespace alus {
namespace snapengine {

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

    std::vector<Dataset> srtms_;
    // use this for cudaFree once you are done with image.
    std::vector<float*> device_formated_srtm_buffers_;
    PointerHolder* device_formated_srtm_buffers_info_{nullptr};
    std::vector<Srtm3FormatComputation> srtm_format_info_;

    void DeviceFree() override;

public:
    Srtm3ElevationModel(std::vector<std::string> file_names);

    std::vector<Srtm3FormatComputation> SrtmPlaceholderData(EarthGravitationalModel96* egm_96);
    void ReadSrtmTiles(EarthGravitationalModel96* egm_96);
    PointerHolder* GetSrtmBuffersInfo() const { return device_formated_srtm_buffers_info_; }

    void HostToDevice() override;
    void DeviceToHost() override;

    ~Srtm3ElevationModel();
};

}  // namespace snapengine
}  // namespace alus
