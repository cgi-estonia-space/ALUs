/**
 * ETAD grid views shared by host code and CUDA kernels.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include <cstddef>

namespace alus::s1tbx::etad {

/** Non-owning azimuth-major grid. The pointer must address memory accessible to the caller. */
struct GridView {
    const double* values{};
    size_t rows{};
    size_t columns{};
};

/**
 * Interpolation origin and steps, in seconds. Azimuth uses UTC seconds since 2000-01-01; range is two-way time.
 * SNAP 14.x uses burst coverage minima as origins, not the XML's product-relative gridStart offsets.
 */
struct GridGeometry {
    double azimuth_origin{};
    double range_origin{};
    double azimuth_step{};
    double range_step{};
};

}  // namespace alus::s1tbx::etad
