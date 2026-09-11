/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "ceres-core/core/i_virtual_dir.h"
#include "etad_grid.h"

namespace alus::s1tbx::etad {

/** Contiguous azimuth-major grid. Invalid samples are quiet NaNs; zero is valid. */
struct Grid {
    size_t rows{};
    size_t columns{};
    std::vector<double> values;
};

/** Time bounds in seconds; azimuth times use UTC seconds since 2000-01-01, range times are two-way. */
struct Coverage {
    double azimuth_min{};
    double azimuth_max{};
    double range_min{};
    double range_max{};
};

/** Burst identity uses the XML indices, never vector offsets or local SLC burst numbers. */
struct Burst {
    int b_index{};
    int p_index{};
    int s_index{};
    std::string swath_id;
    Coverage coverage;
    // XML grid starts are offsets relative to product coverage, retained for provenance.
    double grid_start_azimuth{};
    double grid_start_range{};
    double sampling_azimuth{};
    double sampling_range{};
    size_t rows{};
    size_t columns{};
    // Early-near, early-far, late-near, late-far; each pair is latitude, longitude in degrees.
    std::array<std::array<double, 2>, 4> corners{};
};

struct Swath {
    std::string swath_id;
    int s_index{};
    std::vector<int> burst_indices;
};

struct Acquisition {
    std::string product_id;
    int p_index{};
    double start_time{};
    double stop_time{};
    std::vector<Swath> swaths;
};

struct Calibration {
    std::string swath_id;
    std::string polarisation;
    double range_seconds{};
    double azimuth_seconds{};
    double range_offset_seconds{};   /**< Channel offset, separate from the product reference calibration. */
    double azimuth_offset_seconds{}; /**< Channel offset, separate from the product reference calibration. */
};

/** Plain metadata, independent of SNAP Product/MetadataElement and GDAL datasets. */
struct Metadata {
    std::string mission;
    std::string mode;
    Coverage coverage;
    double carrier_frequency_hz{};
    std::vector<Acquisition> acquisitions;
    std::vector<Burst> bursts;
    std::vector<Calibration> calibrations;
};

struct InSarLayers {
    Grid phase;    /**< radians */
    Grid height;   /**< metres */
    Grid gradient; /**< radians/metre */
};

/** Borrow a host grid for the shared inline calculations. */
inline GridView View(const Grid& grid) { return {grid.values.data(), grid.rows, grid.columns}; }

/** Extract kernel-ready geometry from burst metadata. */
inline GridGeometry Geometry(const Burst& burst) {
    return {burst.coverage.azimuth_min, burst.coverage.range_min, burst.sampling_azimuth, burst.sampling_range};
}

/** Parse a native ETAD annotation XML document, validating required fields and index references. */
Metadata ParseAnnotation(std::string_view xml);

/** Match sensing timestamps and optional absolute orbit in the original SLC name; throw if unmatched/ambiguous. */
const Acquisition& MatchAcquisition(const Metadata& metadata, std::string_view slc_product_name);

/** First acquisition containing azimuth_time, inclusive at both ends; nullptr if outside coverage. */
const Acquisition* FindAcquisition(const Metadata& metadata, double azimuth_time);

/** Find by XML global bIndex; throws if absent. */
const Burst& GetBurst(const Metadata& metadata, int b_index);

/** First listed burst in the specified acquisition/swath with strict azimuth and range bounds, or nullptr. */
const Burst* FindBurst(const Metadata& metadata, int p_index, std::string_view swath_id, double azimuth_time,
                       double range_time);

/**
 * Reference range calibration for the swath, excluding channel offsets. First matching record; absent swath is zero.
 * Channel-dependent timing corrections need polarisation-aware selection and are not applied by this function.
 */
double RangeCalibration(const Metadata& metadata, std::string_view swath_id);

/**
 * Bilinear sample using burst coverage minima and sampling intervals.
 * Returns NaN outside the grid's four-neighbour interpolation domain or for invalid neighbours.
 * Grid and burst dimensions must agree; malformed inputs throw.
 */
double Interpolate(const Grid& grid, const Burst& burst, double azimuth_time, double range_time);

/** Compute -2*pi*frequency_hz*(troposphere + geodetic - ionosphere + calibration_seconds), in radians. */
Grid ComputePhase(const Grid& troposphere, const Grid& geodetic, const Grid& ionosphere, double frequency_hz,
                  double calibration_seconds);

/**
 * SNAP 14.x 25x25-window regression of range differences in troposphere against height differences, seconds/metre.
 * Retains SNAP's exclusive right difference boundary and zero final column. Singular/invalid windows yield NaN.
 */
Grid ComputeTroposphericGradient(const Grid& troposphere, const Grid& height);

/** Derive phase, height and phase/metre gradient. No raster I/O, caching, or implicit frequency conversion. */
InSarLayers ComputeInSarLayers(const Grid& troposphere, const Grid& geodetic, const Grid& ionosphere,
                               const Grid& height, double frequency_hz, double calibration_seconds);

}  // namespace alus::s1tbx::etad

namespace alus::s1tbx {

/**
 * Host-only ETAD SAFE/ZIP reader. Open parses metadata; layer reads are explicit and return owned arrays.
 * Copies share ownership of extracted ZIP files. Returned arrays outlive the reader.
 * Perform I/O during host preparation, before launching workers; no shared GDAL handles or global grid cache.
 */
class Sentinel1EtadProduct {
public:
    /** Accept an unpacked SAFE directory, its manifest.safe, or ZIP. Currently IW SLC ETAD only. */
    static Sentinel1EtadProduct Open(const std::filesystem::path& path);

    [[nodiscard]] const etad::Metadata& GetMetadata() const { return metadata_; }
    [[nodiscard]] const std::string& GetName() const { return name_; }

    /** Read /<swath>/BurstNNNN/<layer> in native NetCDF order, validating dimensions and converting fill to NaN. */
    [[nodiscard]] etad::Grid LoadLayer(int b_index, std::string_view layer) const;

    /**
     * Load four InSAR layers and calculate corrections with the SLC frequency in Hz.
     * Input samples and phase/gradient results retain double precision, with no intermediate Float32 conversion.
     * Numerical differences against SNAP's Float32 reader/exports are documented in docs/etad-precision.md.
     * Uses reference range calibration; channel-dependent calibration offsets are retained as metadata only.
     */
    [[nodiscard]] etad::InSarLayers LoadInSarBurstLayers(int b_index, double frequency_hz) const;

private:
    Sentinel1EtadProduct() = default;
    std::shared_ptr<ceres::IVirtualDir> directory_;
    std::filesystem::path measurement_path_;
    std::string name_;
    etad::Metadata metadata_;
};

}  // namespace alus::s1tbx
