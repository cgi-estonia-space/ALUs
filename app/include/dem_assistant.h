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
#include <memory>
#include <string>
#include <vector>

#include "pointer_holders.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"

namespace alus::app {

class DemAssistant final {
public:
    class ArgumentsExtract final {
    public:
        bool static IsValid(std::string_view dem_file);
        std::vector<std::string> static ExtractSrtm3Files(const std::vector<std::string>& cmd_line_arguments);

    private:
        static std::string AdjustSrtm3Path(std::string_view path);
    };

    DemAssistant() = delete;
    explicit DemAssistant(std::vector<std::string> srtm3_files);

    static std::shared_ptr<DemAssistant> CreateFormattedSrtm3TilesOnGpuFrom(
        const std::vector<std::string>& cmd_line_arguments);

    snapengine::Srtm3ElevationModel* GetSrtm3Manager() { return &model_; }
    snapengine::EarthGravitationalModel96* GetEgm96Manager() { return egm96_.get(); }

    ~DemAssistant() = default;

private:
    snapengine::Srtm3ElevationModel model_;
    std::shared_ptr<snapengine::EarthGravitationalModel96> egm96_;
};
}  // namespace alus::app