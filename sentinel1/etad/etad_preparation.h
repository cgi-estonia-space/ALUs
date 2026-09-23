/**
 * Sentinel-1 SLC to ETAD burst association and host preparation.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "sentinel1_etad_product.h"
#include "slc_metadata.h"

namespace alus::s1tbx::etad {

/** Zero-based contiguous selection in the source SLC annotation. */
struct BurstSelection {
    size_t first_burst{};
    size_t burst_count{};
};

/** Identity mapping from one source SLC burst to one global ETAD burst. */
struct BurstAssociation {
    size_t slc_burst_index{};
    size_t split_burst_index{};
    size_t first_output_line{};
    int etad_b_index{};
};

/** Calculation-ready SLC burst timing. Azimuth is MJD2000 UTC; range is two-way time. */
struct SlcBurstGeometry {
    double first_line_time{};
    double azimuth_time_interval{};
    double first_range_time{};
    double range_time_interval{};
    size_t lines{};
    size_t samples{};
};

/** Non-owning view of one prepared burst. Pointers borrow the corresponding PreparedBurst. */
struct PreparedBurstView {
    BurstAssociation association;
    SlcBurstGeometry slc_geometry;
    GridGeometry etad_geometry;
    GridView phase;
    GridView height;
    GridView gradient;
};

static_assert(std::is_standard_layout_v<BurstSelection> && std::is_trivially_copyable_v<BurstSelection>);
static_assert(std::is_standard_layout_v<BurstAssociation> && std::is_trivially_copyable_v<BurstAssociation>);
static_assert(std::is_standard_layout_v<SlcBurstGeometry> && std::is_trivially_copyable_v<SlcBurstGeometry>);
static_assert(std::is_standard_layout_v<PreparedBurstView> && std::is_trivially_copyable_v<PreparedBurstView>);

struct AcquisitionAssociation {
    int etad_p_index{};
    std::vector<BurstAssociation> bursts;
};

struct PreparedBurst {
    BurstAssociation association;
    SlcBurstGeometry slc_geometry;
    GridGeometry etad_geometry;
    InSarLayers layers;
};

struct PreparedAcquisition {
    std::string slc_product_name;
    std::string swath_id;
    std::string polarisation;
    int etad_p_index{};
    double radar_frequency_hz{};
    std::vector<PreparedBurst> bursts;
};

/** Independently prepared reference and secondary acquisitions for one coregistration pair. */
struct PreparedPair {
    std::shared_ptr<const PreparedAcquisition> reference;
    std::shared_ptr<const PreparedAcquisition> secondary;
};

/** Associate selected SLC bursts without reading correction arrays. */
AcquisitionAssociation AssociateBursts(const slc::Metadata& slc_metadata, BurstSelection selection,
                                       const Metadata& etad_metadata);

/** Associate selected bursts, load their correction arrays and compute phase/height/gradient. */
PreparedAcquisition PrepareInSar(const slc::Metadata& slc_metadata, BurstSelection selection,
                                 const Sentinel1EtadProduct& etad_product);

/** Construct a borrowing descriptor after the owning PreparedBurst has reached its final location. */
inline PreparedBurstView View(const PreparedBurst& burst) {
    return {burst.association, burst.slc_geometry, burst.etad_geometry, View(burst.layers.phase),
            View(burst.layers.height), View(burst.layers.gradient)};
}

}  // namespace alus::s1tbx::etad
