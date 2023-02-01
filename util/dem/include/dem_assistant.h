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

#include "dem_aggregation.h"
#include "dem_type.h"
#include "pointer_holders.h"
#include "snap-dem/dem/dataio/earth_gravitational_model96.h"
#include "srtm3_elevation_model.h"

namespace alus::dem {

class Assistant final {
public:
    class ArgumentsExtract final {
    public:
        bool static IsValidSrtm3Filename(std::string_view dem_file);
        bool static IsValidCopDemCog30mFilename(std::string_view filename);
        std::vector<std::string> static ExtractSrtm3Files(const std::vector<std::string>& cmd_line_arguments);
        std::vector<std::string> static ExtractCopDem30mFiles(const std::vector<std::string>& cmd_line_arguments);
        std::vector<std::string> static PrepareArgs(const std::vector<std::string>& cmd_line_arguments);

    private:
        static std::string AdjustZipPathForSrtm3(std::string_view path);
    };

    Assistant() = delete;
    explicit Assistant(std::vector<std::string> filenames, Type type);

    static std::shared_ptr<Assistant> CreateFormattedDemTilesOnGpuFrom(
        const std::vector<std::string>& cmd_line_arguments);

    Type GetType() const { return type_; }

    std::shared_ptr<Aggregation> GetElevationManager() { return model_; }
    std::shared_ptr<snapengine::EarthGravitationalModel96> GetEgm96Manager() { return egm96_; }

    ~Assistant() = default;

private:
    static std::shared_ptr<Assistant> TryCreateCopDem30mFrom(const std::vector<std::string>& cmd_line_arguments);
    static std::shared_ptr<Assistant> TryCreateSrtm3From(const std::vector<std::string>& cmd_line_arguments);

    Type type_;
    std::shared_ptr<Aggregation> model_;
    std::shared_ptr<snapengine::EarthGravitationalModel96> egm96_;
};
}  // namespace alus::dem