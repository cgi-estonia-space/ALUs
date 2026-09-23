/**
 * Native Sentinel-1 SLC metadata needed for ETAD preparation.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace alus::s1tbx::slc {

/** A native SLC burst's zero-Doppler first-line time, in UTC seconds since 2000-01-01. */
struct Burst {
    double first_line_time{};
};

/** Shared raster geometry from one swath/polarisation annotation. Range times are two-way. */
struct RasterGeometry {
    double azimuth_time_interval{};
    double first_range_time{};
    double range_time_interval{};
    size_t lines_per_burst{};
    size_t samples_per_burst{};
    size_t total_lines{};
    size_t total_samples{};
};

static_assert(std::is_standard_layout_v<Burst> && std::is_trivially_copyable_v<Burst>);
static_assert(std::is_standard_layout_v<RasterGeometry> && std::is_trivially_copyable_v<RasterGeometry>);

/** Owned metadata independent of SNAP Product, MetadataElement and Sentinel1Utils. */
struct Metadata {
    std::string product_name;
    std::string mission;
    std::string mode;
    std::string swath_id;
    std::string polarisation;
    int absolute_orbit{};
    double start_time{};
    double stop_time{};
    double radar_frequency_hz{};
    RasterGeometry raster;
    std::vector<Burst> bursts;
};

/** Parse one native SLC product annotation. product_name is the containing SAFE identity. */
Metadata ParseAnnotation(std::string_view xml, std::string_view product_name);

/**
 * Read exactly one swath/polarisation annotation from an unpacked SAFE directory, manifest.safe or ZIP.
 * Returned metadata is fully owned and outlives the virtual directory used while reading.
 */
Metadata ReadMetadata(const std::filesystem::path& path, std::string_view swath_id, std::string_view polarisation);

}  // namespace alus::s1tbx::slc
