/** Sentinel-1 SLC to ETAD burst association and host preparation. SPDX-License-Identifier: GPL-3.0-or-later */
#include "etad_preparation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "general_utils.h"

namespace {
constexpr double BURST_START_TOLERANCE_SECONDS = 0.1;
constexpr double ACQUISITION_TIME_TOLERANCE_SECONDS = 1.0;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("ETAD preparation: " + message);
    }
}

const alus::s1tbx::etad::Acquisition& MatchSlcAcquisition(const alus::s1tbx::slc::Metadata& slc_metadata,
                                                          const alus::s1tbx::etad::Metadata& etad_metadata) {
    static const std::regex SENSING_TIMES(R"(\d{8}T\d{6}_\d{8}T\d{6}(?:_\d{6})?)");
    std::smatch name_match;
    if (std::regex_search(slc_metadata.product_name, name_match, SENSING_TIMES)) {
        const std::string token = alus::utils::general::ToLower(std::string_view(name_match.str()));
        const alus::s1tbx::etad::Acquisition* found = nullptr;
        for (const auto& acquisition : etad_metadata.acquisitions) {
            if (alus::utils::general::ToLower(std::string_view(acquisition.product_id)).find(token) !=
                std::string::npos) {
                Require(found == nullptr, "ambiguous acquisition match for '" + slc_metadata.product_name + "'");
                found = &acquisition;
            }
        }
        if (found != nullptr) {
            return *found;
        }
    }

    std::ostringstream orbit;
    orbit << '_' << std::setfill('0') << std::setw(6) << slc_metadata.absolute_orbit << '_';
    const std::string mission = alus::utils::general::ToLower(std::string_view(slc_metadata.mission));
    const alus::s1tbx::etad::Acquisition* found = nullptr;
    for (const auto& acquisition : etad_metadata.acquisitions) {
        const std::string product_id = alus::utils::general::ToLower(std::string_view(acquisition.product_id));
        if (slc_metadata.absolute_orbit > 0 && !mission.empty() && product_id.rfind(mission, 0) == 0 &&
            product_id.find(orbit.str()) != std::string::npos &&
            acquisition.start_time - ACQUISITION_TIME_TOLERANCE_SECONDS <= slc_metadata.start_time &&
            slc_metadata.stop_time <= acquisition.stop_time + ACQUISITION_TIME_TOLERANCE_SECONDS) {
            Require(found == nullptr, "ambiguous acquisition metadata for '" + slc_metadata.product_name + "'");
            found = &acquisition;
        }
    }
    Require(found != nullptr, "no ETAD acquisition matches SLC metadata for '" + slc_metadata.product_name + "'");
    return *found;
}
}  // namespace

namespace alus::s1tbx::etad {

AcquisitionAssociation AssociateBursts(const slc::Metadata& slc_metadata, BurstSelection selection,
                                       const Metadata& etad_metadata) {
    Require(selection.burst_count > 0, "burst selection is empty");
    Require(selection.first_burst <= slc_metadata.bursts.size() &&
                selection.burst_count <= slc_metadata.bursts.size() - selection.first_burst,
            "burst selection is outside the SLC annotation");
    Require(slc_metadata.raster.lines_per_burst > 0 && slc_metadata.raster.samples_per_burst > 0,
            "invalid SLC burst dimensions");

    const auto& acquisition = MatchSlcAcquisition(slc_metadata, etad_metadata);
    const auto swath = std::find_if(acquisition.swaths.begin(), acquisition.swaths.end(), [&](const auto& candidate) {
        return candidate.swath_id == slc_metadata.swath_id;
    });
    Require(swath != acquisition.swaths.end(), "ETAD acquisition does not contain swath " + slc_metadata.swath_id);
    Require(selection.burst_count <= std::numeric_limits<size_t>::max() / slc_metadata.raster.lines_per_burst,
            "selected output line offsets are not representable");

    AcquisitionAssociation result;
    result.etad_p_index = acquisition.p_index;
    result.bursts.reserve(selection.burst_count);
    std::set<int> used_etad_bursts;
    for (size_t split_index = 0; split_index < selection.burst_count; ++split_index) {
        const size_t source_index = selection.first_burst + split_index;
        const auto& source_burst = slc_metadata.bursts[source_index];
        const Burst* match = nullptr;
        for (const int b_index : swath->burst_indices) {
            const auto& candidate = GetBurst(etad_metadata, b_index);
            if (std::abs(source_burst.first_line_time - candidate.coverage.azimuth_min) <
                BURST_START_TOLERANCE_SECONDS) {
                Require(match == nullptr, "ambiguous ETAD burst for SLC burst " + std::to_string(source_index));
                match = &candidate;
            }
        }
        Require(match != nullptr, "no ETAD burst for SLC burst " + std::to_string(source_index));

        const double last_line_time =
            source_burst.first_line_time + static_cast<double>(slc_metadata.raster.lines_per_burst - 1) *
                                               slc_metadata.raster.azimuth_time_interval;
        const double last_range_time =
            slc_metadata.raster.first_range_time + static_cast<double>(slc_metadata.raster.samples_per_burst - 1) *
                                                       slc_metadata.raster.range_time_interval;
        const auto& coverage = match->coverage;
        Require(source_burst.first_line_time > coverage.azimuth_min && last_line_time < coverage.azimuth_max &&
                    slc_metadata.raster.first_range_time > coverage.range_min && last_range_time < coverage.range_max,
                "ETAD burst " + std::to_string(match->b_index) + " does not cover SLC burst " +
                    std::to_string(source_index));
        Require(used_etad_bursts.insert(match->b_index).second,
                "ETAD burst " + std::to_string(match->b_index) + " matched more than one SLC burst");
        result.bursts.push_back(
            {source_index, split_index, split_index * slc_metadata.raster.lines_per_burst, match->b_index});
    }
    return result;
}

PreparedAcquisition PrepareInSar(const slc::Metadata& slc_metadata, BurstSelection selection,
                                 const Sentinel1EtadProduct& etad_product) {
    const auto association = AssociateBursts(slc_metadata, selection, etad_product.GetMetadata());
    PreparedAcquisition result{slc_metadata.product_name, slc_metadata.swath_id, slc_metadata.polarisation,
                               association.etad_p_index, slc_metadata.radar_frequency_hz, {}};
    result.bursts.reserve(association.bursts.size());
    for (const auto& mapping : association.bursts) {
        const auto& etad_burst = GetBurst(etad_product.GetMetadata(), mapping.etad_b_index);
        const SlcBurstGeometry slc_geometry{
            slc_metadata.bursts[mapping.slc_burst_index].first_line_time,
            slc_metadata.raster.azimuth_time_interval,
            slc_metadata.raster.first_range_time,
            slc_metadata.raster.range_time_interval,
            slc_metadata.raster.lines_per_burst,
            slc_metadata.raster.samples_per_burst,
        };
        auto layers = etad_product.LoadInSarBurstLayers(mapping.etad_b_index, slc_metadata.polarisation,
                                                        slc_metadata.radar_frequency_hz);
        result.bursts.push_back({mapping, slc_geometry, Geometry(etad_burst), std::move(layers)});
    }
    return result;
}

}  // namespace alus::s1tbx::etad
