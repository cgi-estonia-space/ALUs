/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.orbits.sentinel1.SentinelPODOrbitFile.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
 *
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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

#include "pugixml.hpp"

#include "io/orbits/i_orbit_file.h"
#include "metadata_element.h"
#include "orbit_vector.h"

namespace alus {
namespace s1tbx {

class FixedHeader {
private:
    std::string mission_;
    std::string file_type_;
    std::string validity_start_;
    std::string validity_stop_;

public:
    friend class SentinelPODOrbitFile;
    FixedHeader(std::string_view mission, std::string_view file_type, std::string_view validity_start,
                std::string_view validity_stop)
        : mission_(mission), file_type_(file_type), validity_start_(validity_start), validity_stop_(validity_stop){};
};

class SentinelPODOrbitFile : virtual public IOrbitFile {
private:
    int poly_degree_;
    std::unique_ptr<FixedHeader> fixed_header_ = nullptr;
    std::shared_ptr<snapengine::MetadataElement> abs_root_;
    std::optional<boost::filesystem::path> orbit_file_;
    std::vector<std::shared_ptr<snapengine::OrbitVector>> osv_list_;
    static constexpr std::string_view DATE_FORMAT{"%Y%m%d-%H%M%S"};
    static constexpr std::string_view ORBIT_DATE_FORMAT{"%Y-%m-%d %H:%M:%S"};

    static std::string GetMissionPrefix(std::shared_ptr<snapengine::MetadataElement> abs_root);
    static boost::filesystem::path GetDestFolder(std::string_view mission_prefix, std::string_view orbit_type, int year,
                                                 int month);

    static std::optional<boost::filesystem::path> FindOrbitFile(std::string_view mission_prefix,
                                                                std::string_view orbit_type, double state_vector_time,
                                                                int year, int month);

    static bool IsWithinRange(std::string_view filename, double state_vector_time);

    static std::shared_ptr<snapengine::Utc> GetValidityStartFromFilenameUTC(std::string_view filename);
    static std::shared_ptr<snapengine::Utc> GetValidityStopFromFilenameUTC(std::string_view filename);
    static std::string ExtractTimeFromFilename(std::string_view filename, int offset);
    static std::string ConvertUtc(std::string_view utc);
    static std::shared_ptr<snapengine::OrbitVector> ReadOneOSV(const pugi::xml_node& osv_node);
    static std::vector<std::shared_ptr<snapengine::OrbitVector>> ReadOSVList(pugi::xml_node& list_of_o_s_v_s_node);
    static std::optional<boost::filesystem::path> DownloadFromQCRestAPI(std::string_view mission_prefix,
                                                                        std::string_view orbit_type, int year,
                                                                        int month, int day, int hour, int minute,
                                                                        int second, double state_vector_time);
    void ReadFixedHeader(pugi::xml_node& fixed_header_node);

protected:
    static std::shared_ptr<snapengine::Utc> ToUtc(std::string_view str);

    std::string GetValidityStopFromHeader();
    std::string GetValidityStartFromHeader();

    /**
     * Check if product acquisition time is within the validity period of the orbit file.
     *
     * @throws Exception
     */
    void CheckOrbitFileValidity();
    void ReadOrbitFile();

public:
    static constexpr std::string_view RESTITUTED = "Sentinel Restituted";
    static constexpr std::string_view PRECISE = "Sentinel Precise";

    SentinelPODOrbitFile(int poly_degree, std::shared_ptr<snapengine::MetadataElement> abs_root)
        : poly_degree_(poly_degree), abs_root_(abs_root) {}

    std::vector<std::string> GetAvailableOrbitTypes() override {
        return std::vector<std::string>{std::string(RESTITUTED), std::string(PRECISE)};
    }

    boost::filesystem::path RetrieveOrbitFile(std::string_view orbit_type) override;
    boost::filesystem::path GetOrbitFile() override { return *orbit_file_; }

    std::vector<std::shared_ptr<snapengine::OrbitVector>> GetOrbitData(double start_utc, double end_utc);

    /**
     * Get orbit state vector for given time using polynomial fitting.
     *
     * @param utc The UTC in days.
     * @return The orbit state vector.
     * @throws Exception The exceptions.
     */
    std::shared_ptr<snapengine::OrbitVector> GetOrbitData(double utc) override;
};
}  // namespace s1tbx
}  // namespace alus